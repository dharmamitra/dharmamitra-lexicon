#!/usr/bin/env python3
"""Held-out teacher labels for student-teacher agreement (precision / recall / F1).

Mines N_KEEP fresh seeds per direction with EXACTLY the protocol that produced the
training seeds, on sentence pairs the teacher was never called on before:

  sa-bo, sa-zh : generate_seeds.py protocol  (gemini-3.6-flash, thinking budget 16384,
                 strict prompt, reservoir sample over the substrate, gates: verbatim,
                 grounded, <=90% tagged). Wylie -> Unicode conversion + re-grounding
                 as in seeds_to_unicode.py.
  sa-en        : generate_seeds_saen.py protocol (gemini-3.8-flash, thinking HIGH,
                 the unused half of sa-en/seed_candidates.ndjson, gates: verbatim,
                 >=2 spans, <=90% tagged).

Exclusion: every unit whose key appears in the ORIGINAL <dir>/gemini_raw.ndjson
(i.e. every pair the teacher was called on, kept or not) is skipped, so the new
rows are disjoint from train / val / test of both SFT datasets.

Outputs (resumable):
  heldout/<dir>/gemini_raw.ndjson  raw teacher calls (same schema as before)
  heldout/<dir>/seeds.ndjson       kept seeds in the strict-prompt format
  heldout/<dir>/test.jsonl         kept seeds in the STUDENT (minimal-prompt) format
  ~/data/qwen-dictionaries-heldout500/test.jsonl   all directions concatenated

  python generate_heldout.py --direction sa-bo --smoke 3
  python generate_heldout.py --direction sa-bo
  python generate_heldout.py --assemble          # build the concatenated test.jsonl
"""
import argparse, glob, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries"
sys.path.insert(0, HERE)
import generate_seeds as G
import generate_seeds_saen as S
import seeds_to_unicode as SU

API_KEY = S.API_KEY                     # env or ~/code/mitra-evaluation/.env
OUT_BASE = Path(HERE) / "heldout"
DATA_OUT = Path(os.path.expanduser("~/data/qwen-dictionaries-heldout500"))
SEED = int(os.environ.get("HELDOUT_SEED", "2026"))
N_KEEP = int(os.environ.get("N_KEEP", "500"))
N_SAMPLE = {"sa-bo": 1500, "sa-zh": 1200}   # reservoir size; keep rate was 56% / 79%
WAVE = int(os.environ.get("WAVE", "300"))
WORKERS = int(os.environ.get("GEMINI_WORKERS", "40"))
INSTRUCTION = ("Insert the numbered tags into the target line, marking the "
               "word(s) that translate each numbered source word. Copy the "
               "target verbatim and omit numbers without a clear equivalent.")
TEACHER = {"sa-bo": "gemini-3.6-flash", "sa-zh": "gemini-3.6-flash", "sa-en": "gemini-3.8-flash"}


def minimal_row(input_text, target_text, lang, **meta):
    block = input_text.split("NOW DO THIS ONE\n", 1)[1].rsplit("OUTPUT:", 1)[0].strip()
    return {"input_text": f"{INSTRUCTION}\n\n{block}", "target_text": target_text,
            "lang": lang, **meta}


def call_with_retry(client, model, prompt, gcfg, st, lock):
    last_err = None
    for a in range(8):
        try:
            return (client.models.generate_content(
                model=model, contents=prompt, config=gcfg).text or "").strip()
        except Exception as e:
            last_err = str(e)[:160]
            wait = 15 * (a + 1) if ("429" in last_err or "RESOURCE_EXHAUSTED" in last_err) else 2 ** a
            if a == 7:
                with lock:
                    st["errs"] += 1
                    if st["errs"] % 10 == 1:
                        print(f"[err x{st['errs']}] {last_err}", file=sys.stderr)
                return None
            time.sleep(wait)


# ---------------------------------------------------------------- sa-bo / sa-zh
def units_pair(direction):
    cfg = G.DIRS[direction]
    files = sorted(glob.glob(str(G.BASE / cfg["data"] / cfg["glob"])))
    id_set = None
    if cfg.get("pairids"):
        id_set = set(open(G.BASE / cfg["pairids"], encoding="utf-8").read().split("\n"))
        id_set.discard("")
    rows, total = G.reservoir(files, N_SAMPLE[direction], random.Random(SEED), id_set)
    rows = G.length_filter(rows, cfg["q_field"], cfg["t_field"])
    units = [u for u in (G.build_unit(d, cfg) for d in rows) if u]
    key = lambda u: (str(u["src_segnr"]), str(u["tgt_segnr"]), u["target"])
    # exclude every pair the teacher was called on for the training seeds
    done = set()
    for l in open(Path(HERE) / direction / "gemini_raw.ndjson", encoding="utf-8"):
        d = json.loads(l)
        done.add((str(d.get("src_segnr")), str(d.get("tgt_segnr")), d.get("target")))
    seen, out = set(), []
    for u in units:
        k = key(u)
        if k in done or k in seen:
            continue
        seen.add(k); out.append(u)
    print(f"[{direction}] sampled {len(rows)} of {total}, {len(units)} usable, "
          f"{len(out)} after excluding {len(done)} training-time calls", file=sys.stderr)
    random.Random(SEED).shuffle(out)
    return cfg, out, key


def student_row_pair(direction, u, cfg, r):
    """Kept teacher row -> student-format row (Unicode Tibetan for sa-bo)."""
    grounded = {int(n): s for n, s in r["grounded"].items()}
    if direction == "sa-bo":
        t_uni = SU.U(u["target"])
        spans = {}
        for n, span in grounded.items():
            su = SU.U(span).strip("་")
            if not su or su not in t_uni:
                return None
            spans[n] = su
        tagged = G.reconstruct(t_uni, spans)
        input_text = SU.PROMPT_U["sa-bo"].format(
            target=t_uni, aid=u["aid"], sa_stop=G.SA_STOP, bo_stop=SU.BO_STOP_U,
            zh_stop=G.ZH_STOP, en_stop=G.EN_STOP, **u["q"])
    else:
        tagged = G.reconstruct(u["target"], grounded)
        input_text = G.fmt_strict(u, cfg)
    return minimal_row(input_text, tagged, direction, id=u["id"],
                       src_segnr=u["src_segnr"], tgt_segnr=u["tgt_segnr"])


def run_pair(direction, smoke):
    cfg, units, key = units_pair(direction)
    out_dir = OUT_BASE / direction; out_dir.mkdir(parents=True, exist_ok=True)
    rawp, seedp, testp = out_dir / "gemini_raw.ndjson", out_dir / "seeds.ndjson", out_dir / "test.jsonl"
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=API_KEY)
    gcfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=G.THINKING_BUDGET))
    done, kept0 = set(), 0
    if not smoke:
        if rawp.exists():
            for l in open(rawp, encoding="utf-8"):
                d = json.loads(l)
                done.add((str(d.get("src_segnr")), str(d.get("tgt_segnr")), d.get("target")))
        if testp.exists():
            kept0 = sum(1 for _ in open(testp, encoding="utf-8"))
        units = [u for u in units if key(u) not in done]
        print(f"[{direction}] resume: {len(done)} done, {kept0} kept, {len(units)} to run", file=sys.stderr)
    else:
        units = units[:smoke]
    lock = threading.Lock()
    st = {"done": 0, "kept": kept0, "errs": 0, "faithful": 0, "t0": time.time()}
    fhs = None if smoke else (open(rawp, "a", encoding="utf-8"), open(seedp, "a", encoding="utf-8"),
                              open(testp, "a", encoding="utf-8"))

    def work(u):
        out = call_with_retry(client, TEACHER[direction], G.fmt_strict(u, cfg), gcfg, st, lock)
        if out is None:
            return None
        grounded, faithful = G.parse_output(out, len(u["toks"]), u["target"])
        return u, out, grounded, faithful

    pos = 0
    while pos < len(units) and (smoke or st["kept"] < N_KEEP):
        wave = units[pos:pos + WAVE]; pos += len(wave)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for f in as_completed([ex.submit(work, u) for u in wave]):
                r = f.result()
                with lock:
                    st["done"] += 1
                    if r is None:
                        continue
                    u, out, grounded, faithful = r
                    st["faithful"] += int(faithful)
                    raw = {"id": u["id"], "src_segnr": u["src_segnr"], "tgt_segnr": u["tgt_segnr"],
                           "target": u["target"], "raw": out, "grounded": grounded,
                           "faithful": faithful, "n_toks": len(u["toks"]), "n_tagged": len(grounded)}
                    keep = faithful and grounded and len(grounded) <= 0.90 * len(u["toks"])
                    srow = student_row_pair(direction, u, cfg, raw) if keep else None
                    if smoke:
                        print(f"\n--- {direction} {u['id']}\nTARGET: {u['target'][:160]}\nRAW:    {out[:160]}\n"
                              f"tagged {len(grounded)}/{len(u['toks'])} faithful={faithful} keep={bool(keep)}")
                        if srow:
                            print("STUDENT INPUT tail:", srow["input_text"][-200:].replace("\n", " | "))
                            print("STUDENT TARGET:", srow["target_text"][:200])
                        continue
                    fhs[0].write(json.dumps(raw, ensure_ascii=False) + "\n")
                    if srow and st["kept"] < N_KEEP:
                        fhs[1].write(json.dumps(G.instruction_row(u, cfg, G.reconstruct(u["target"], grounded), direction),
                                                ensure_ascii=False) + "\n")
                        fhs[2].write(json.dumps(srow, ensure_ascii=False) + "\n")
                        st["kept"] += 1
                    if st["done"] % 50 == 0:
                        for fh in fhs: fh.flush()
                        print(f"[{direction}] {st['done']} calls kept={st['kept']} faithful={st['faithful']} "
                              f"errs={st['errs']} ({time.time()-st['t0']:.0f}s)", file=sys.stderr)
    if fhs:
        for fh in fhs: fh.close()
    print(f"[{direction}] DONE calls={st['done']} kept={st['kept']} faithful={st['faithful']} errs={st['errs']}",
          file=sys.stderr)


# ---------------------------------------------------------------------- sa-en
def run_saen(smoke):
    direction = "sa-en"
    out_dir = OUT_BASE / direction; out_dir.mkdir(parents=True, exist_ok=True)
    rawp, seedp, testp = out_dir / "gemini_raw.ndjson", out_dir / "seeds.ndjson", out_dir / "test.jsonl"
    called = set(json.loads(l)["id"] for l in open(Path(HERE) / "sa-en/gemini_raw.ndjson", encoding="utf-8"))
    cands = [json.loads(l) for l in open(S.CANDIDATES, encoding="utf-8")]
    units = [u for u in (S.build_unit(d) for d in cands) if u and u["id"] not in called]
    random.Random(SEED).shuffle(units)
    print(f"[{direction}] {len(units)} usable unused candidates (of {len(cands)}, {len(called)} called before)",
          file=sys.stderr)
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=API_KEY)
    gcfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=S.THINKING_LEVEL))
    done, kept0 = set(), 0
    if not smoke:
        if rawp.exists():
            done = set(json.loads(l)["id"] for l in open(rawp, encoding="utf-8"))
        if testp.exists():
            kept0 = sum(1 for _ in open(testp, encoding="utf-8"))
        units = [u for u in units if u["id"] not in done]
        print(f"[{direction}] resume: {len(done)} done, {kept0} kept, {len(units)} to run", file=sys.stderr)
    else:
        units = units[:smoke]
    lock = threading.Lock()
    st = {"done": 0, "kept": kept0, "errs": 0, "faithful": 0, "t0": time.time()}
    fhs = None if smoke else (open(rawp, "a", encoding="utf-8"), open(seedp, "a", encoding="utf-8"),
                              open(testp, "a", encoding="utf-8"))

    def work(u):
        out = call_with_retry(client, TEACHER[direction], S.fmt(u), gcfg, st, lock)
        if out is None:
            return None
        out = out.splitlines()[-1].strip() if "\n" in out.strip() else out
        grounded, faithful = S.parse_output(out, len(u["toks"]), u["target"])
        tagged = S.positional_tagged(out, u["target"], grounded) or S.reconstruct(u["target"], grounded)
        return u, out, grounded, faithful, tagged

    pos = 0
    while pos < len(units) and (smoke or st["kept"] < N_KEEP):
        wave = units[pos:pos + WAVE]; pos += len(wave)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for f in as_completed([ex.submit(work, u) for u in wave]):
                r = f.result()
                with lock:
                    st["done"] += 1
                    if r is None:
                        continue
                    u, out, grounded, faithful, tagged = r
                    st["faithful"] += int(faithful)
                    keep = faithful and len(grounded) >= 2 and len(grounded) <= 0.90 * len(u["toks"])
                    if smoke:
                        print(f"\n--- {u['id']}\nTARGET: {u['target'][:200]}\nRAW:    {out[:200]}\nTAGGED: {tagged[:200]}\n"
                              f"tagged {len(grounded)}/{len(u['toks'])} faithful={faithful} keep={bool(keep)}")
                        continue
                    fhs[0].write(json.dumps({
                        "id": u["id"], "file": u["file"], "target": u["target"], "raw": out,
                        "grounded": grounded, "faithful": faithful, "n_toks": len(u["toks"]),
                        "n_tagged": len(grounded)}, ensure_ascii=False) + "\n")
                    if keep and st["kept"] < N_KEEP:
                        srow = S.instruction_row(u, tagged)
                        fhs[1].write(json.dumps(srow, ensure_ascii=False) + "\n")
                        fhs[2].write(json.dumps(minimal_row(srow["input_text"], tagged, direction,
                                                            id=u["id"], file=u["file"], license=u["license"]),
                                                ensure_ascii=False) + "\n")
                        st["kept"] += 1
                    if st["done"] % 50 == 0:
                        for fh in fhs: fh.flush()
                        print(f"[{direction}] {st['done']} calls kept={st['kept']} faithful={st['faithful']} "
                              f"errs={st['errs']} ({time.time()-st['t0']:.0f}s)", file=sys.stderr)
    if fhs:
        for fh in fhs: fh.close()
    print(f"[{direction}] DONE calls={st['done']} kept={st['kept']} faithful={st['faithful']} errs={st['errs']}",
          file=sys.stderr)


def assemble():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    train = set()
    for ds in ("qwen-dictionaries-spanmine", "qwen-dictionaries-spanmine-saen"):
        for sp in ("train", "val", "test"):
            for l in open(os.path.expanduser(f"~/data/{ds}/{sp}.jsonl"), encoding="utf-8"):
                train.add(json.loads(l)["input_text"])
    n = 0
    with open(DATA_OUT / "test.jsonl", "w", encoding="utf-8") as out:
        for d in ("sa-bo", "sa-zh", "sa-en"):
            p = OUT_BASE / d / "test.jsonl"
            if not p.exists():
                print(f"[assemble] missing {p}", file=sys.stderr); continue
            rows = [l for l in open(p, encoding="utf-8") if l.strip()]
            # belt and braces: drop any row whose student input equals a training row
            rows = [l for l in rows if json.loads(l)["input_text"] not in train][:N_KEEP]
            out.writelines(rows); n += len(rows)
            print(f"[assemble] {d}: {len(rows)}", file=sys.stderr)
    print(f"[assemble] {n} rows -> {DATA_OUT/'test.jsonl'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["sa-bo", "sa-zh", "sa-en"])
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if a.assemble:
        return assemble()
    if not a.direction:
        ap.error("need --direction or --assemble")
    if a.direction == "sa-en":
        run_saen(a.smoke)
    else:
        run_pair(a.direction, a.smoke)


if __name__ == "__main__":
    main()
