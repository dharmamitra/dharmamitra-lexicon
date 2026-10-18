#!/usr/bin/env python3
"""Generalized Gemini span-projection mining for all 4 directions.

Query side = numbered tags; target side = the verbatim string with tags inserted
around the span translating each numbered query word. Per-span grounding gate:
a tag is kept only if its span is a verbatim substring of the target string.

  sa-bo : Sanskrit(unsandhied+lemma) -> Tibetan(wylie)
  sa-zh : Sanskrit(unsandhied+lemma) -> Chinese(original; segmented as aid)
  zh-sa : Chinese(segmented)         -> Sanskrit(raw, no segmentation)
  zh-bo : Chinese(segmented)         -> Tibetan(wylie)

  python mine_spans.py --direction sa-zh --probe 40        # cost estimate
  python mine_spans.py --direction sa-zh --smoke 5         # show live examples
  python mine_spans.py --direction sa-zh                   # full run
"""
import argparse, glob, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path("/home/sebastian/code/mitra-dictionaries/mitra-dictionaries-2026")
MODEL_ID = "gemini-3.1-flash-lite"
THINKING_BUDGET = 512
SEED = 42
N_TEST, N_VAL, N_TRAIN = 0, 500, 20000   # 20k train + 500 eval per direction
GEMINI_WORKERS = int(os.environ.get("GEMINI_WORKERS", "256"))
API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set GEMINI_API_KEY
PRICE_IN, PRICE_OUT = 0.10, 0.40

# direction -> data + which field/kind is query vs target
DIRS = {
    "sa-bo": dict(data="sa-bo/sentence-pairs-enriched", glob="sa-bo_matches_*.ndjson",
                  q_field="root", q_kind="sanskrit", t_field="par", t_kind="tibetan"),
    "sa-zh": dict(data="sa-zh/sentence-pairs", glob="sa-zh_matches_*.filtered.ndjson",
                  q_field="root", q_kind="sanskrit", t_field="par", t_kind="chinese",
                  pairids="final-output/sa-zh.productive_pairids.txt"),
    "zh-sa": dict(data="sa-zh/sentence-pairs", glob="sa-zh_matches_*.filtered.ndjson",
                  q_field="par", q_kind="chinese", t_field="root", t_kind="sanskrit",
                  pairids="final-output/sa-zh.productive_pairids.txt"),
    "zh-bo": dict(data="zh-bo/sentence-pairs", glob="bo-zh_matches_*.filtered.ndjson",
                  q_field="par", q_kind="chinese", t_field="root", t_kind="tibetan",
                  pairids="final-output/zh-bo.productive_pairids.txt"),
}

TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)

# ---- per-direction prompts (clear example, no wiggle room) ----
P_SA_TIB = """You are an expert Sanskrit-Tibetan philologist doing word alignment.
You are given a Sanskrit sentence with each word in a numbered tag <1>..</1> (SANSKRIT), the same words as lemmas with the SAME numbers (LEMMAS), and the Tibetan translation in Wylie (TIBETAN).
TASK: Reproduce the TIBETAN line EXACTLY and insert the numbered tags around the Tibetan word(s) translating each numbered Sanskrit word.
RULES: 1) Copy the Tibetan verbatim - never translate/reorder/add/delete it; insert ONLY tags. 2) <n> wraps the Tibetan translating Sanskrit word n. 3) A span may cover several consecutive Tibetan words. 4) Tags will be out of numerical order; never move Tibetan to make them sequential. 5) Each Tibetan word in at most one tag; no overlap/nesting. 6) Untranslated Tibetan and particles (ni, kyi, gyi, gi, du, tu, su, la, dang) stay untagged unless part of a span. 7) Omit numbers with no Tibetan equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged Tibetan line.
EXAMPLE
SANSKRIT: <1>cittaṃ</1> <2>kāśyapa</2> <3>na</3> <4>labhyate</4>
LEMMAS: <1>citta</1> <2>kāśyapa</2> <3>na</3> <4>labh</4>
TIBETAN: 'od srungs sems ni mi rnyed do
OUTPUT: <2>'od srungs</2> <1>sems</1> ni <3>mi</3> <4>rnyed do</4>
NOW DO THIS ONE
SANSKRIT: {sanskrit_tagged}
LEMMAS: {lemmas_tagged}
TIBETAN: {target}
OUTPUT:"""

P_SA_ZH = """You are an expert Sanskrit-Chinese philologist doing word alignment.
You are given a Sanskrit sentence with each word in a numbered tag <1>..</1> (SANSKRIT), the same words as lemmas with the SAME numbers (LEMMAS), the Chinese translation as written (CHINESE), and the same Chinese with word boundaries marked by spaces as an aid only (CHINESE_SEGMENTED).
TASK: Reproduce the CHINESE line EXACTLY and insert the numbered tags around the Chinese character(s) that translate each numbered Sanskrit word.
RULES: 1) Copy the original CHINESE verbatim - never translate/reorder/add/delete it; insert ONLY tags. Use CHINESE_SEGMENTED only as a hint for word boundaries. 2) <n> wraps the Chinese translating Sanskrit word n. 3) A span may cover one or several consecutive Chinese characters. 4) Tags will be out of numerical order; never move Chinese to make them sequential. 5) Each Chinese character in at most one tag; no overlap/nesting. 6) Chinese with no Sanskrit source, punctuation (，。、！？) and grammatical particles (之 其 者 而 也 矣) stay untagged. 7) Omit numbers with no Chinese equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged Chinese line.
EXAMPLE
SANSKRIT: <1>buddhaḥ</1> <2>dharmaṃ</2> <3>deśayati</3>
LEMMAS: <1>buddha</1> <2>dharma</2> <3>diś</3>
CHINESE: 佛為眾生說法。
CHINESE_SEGMENTED: 佛 為 眾生 說 法
OUTPUT: <1>佛</1>為眾生<3>說</3><2>法</2>。
NOW DO THIS ONE
SANSKRIT: {sanskrit_tagged}
LEMMAS: {lemmas_tagged}
CHINESE: {target}
CHINESE_SEGMENTED: {aid}
OUTPUT:"""

P_ZH_SA = """You are an expert Chinese-Sanskrit philologist doing word alignment.
You are given a Chinese sentence with each word in a numbered tag <1>..</1> (CHINESE), and the Sanskrit source as written, with sandhi, unsegmented (SANSKRIT).
TASK: Reproduce the SANSKRIT line EXACTLY and insert the numbered tags around the Sanskrit substring(s) that translate each numbered Chinese word.
RULES: 1) Copy the Sanskrit verbatim - identical characters including sandhi and order; insert ONLY tags; never change/reorder/add/delete it. 2) <n> wraps the Sanskrit translating Chinese word n. A span is any contiguous stretch of the Sanskrit string - because of compounds and sandhi a tag MAY cover only part of a written word. 3) A span may cover one or several characters. 4) Tags will be out of numerical order. 5) Each Sanskrit character in at most one tag; no overlap/nesting. 6) Sanskrit with no Chinese counterpart stays untagged. 7) Omit numbers with no Sanskrit equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged Sanskrit line.
EXAMPLE
CHINESE: <1>佛</1> <2>說</2> <3>法</3>
SANSKRIT: buddho dharmaṃ deśayati
OUTPUT: <1>buddho</1> <3>dharmaṃ</3> <2>deśayati</2>
NOW DO THIS ONE
CHINESE: {chinese_tagged}
SANSKRIT: {target}
OUTPUT:"""

P_ZH_BO = """You are an expert Chinese-Tibetan philologist doing word alignment.
You are given a Chinese sentence with each word in a numbered tag <1>..</1> (CHINESE), and the Tibetan translation in Wylie (TIBETAN).
TASK: Reproduce the TIBETAN line EXACTLY and insert the numbered tags around the Tibetan word(s) that translate each numbered Chinese word.
RULES: 1) Copy the Tibetan verbatim - never translate/reorder/add/delete it; insert ONLY tags. 2) <n> wraps the Tibetan translating Chinese word n. 3) A span may cover several consecutive Tibetan words. 4) Tags will be out of numerical order; never move Tibetan to make them sequential. 5) Each Tibetan word in at most one tag; no overlap/nesting. 6) Untranslated Tibetan and particles (ni, kyi, gyi, gi, du, tu, su, la, dang) stay untagged. 7) Omit numbers with no Tibetan equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged Tibetan line.
EXAMPLE
CHINESE: <1>佛</1> <2>說</2>
TIBETAN: sangs rgyas kyis gsungs so
OUTPUT: <1>sangs rgyas</1> kyis <2>gsungs</2> so
NOW DO THIS ONE
CHINESE: {chinese_tagged}
TIBETAN: {target}
OUTPUT:"""

PROMPTS = {("sanskrit", "tibetan"): P_SA_TIB, ("sanskrit", "chinese"): P_SA_ZH,
           ("chinese", "sanskrit"): P_ZH_SA, ("chinese", "tibetan"): P_ZH_BO}


def build_unit(d, cfg):
    qf, tf = cfg["q_field"], cfg["t_field"]
    target = (d.get(f"{tf}_string") or "").strip()
    if not target:
        return None
    if cfg["q_kind"] == "sanskrit":
        fa = d.get("full_analysis") or []
        toks = [(t.get("unsandhied", ""), t.get("lemma", "")) for t in fa if t.get("unsandhied")]
        if not toks:
            return None
        q = {"sanskrit_tagged": " ".join(f"<{i}>{s}</{i}>" for i, (s, l) in enumerate(toks, 1)),
             "lemmas_tagged": " ".join(f"<{i}>{l}</{i}>" for i, (s, l) in enumerate(toks, 1))}
    else:  # chinese query
        words = (d.get("par_zh_segmented") or "").split()
        toks = [(w, w) for w in words]
        if not toks:
            return None
        q = {"chinese_tagged": " ".join(f"<{i}>{w}</{i}>" for i, w in enumerate(words, 1))}
    aid = (d.get("par_zh_segmented") or "") if cfg["t_kind"] == "chinese" else ""
    return {"toks": toks, "q": q, "target": target, "aid": aid, "id": d.get("id"),
            "src_segnr": d.get(f"{qf}_segnr"), "tgt_segnr": d.get(f"{tf}_segnr")}


def fmt_prompt(u, cfg):
    p = PROMPTS[(cfg["q_kind"], cfg["t_kind"])]
    return p.format(target=u["target"], aid=u["aid"], **u["q"])


def parse_output(out, ntok, target):
    grounded = {}
    for m in TAG.finditer(out):
        n = int(m.group(1)); span = m.group(2).strip()
        if not (1 <= n <= ntok) or not span or n in grounded:
            continue
        if span in target:
            grounded[n] = span
        else:
            s2 = span.strip("/ ").strip()
            if s2 and s2 in target:
                grounded[n] = s2
    detag = re.sub(r"</?\d+>", "", out)
    fid = " ".join(detag.split()) == " ".join(target.split())
    return grounded, fid


def reconstruct(target, grounded):
    used = [False] * len(target); placed = []
    for n, span in sorted(grounded.items(), key=lambda kv: -len(kv[1])):
        start = 0
        while True:
            idx = target.find(span, start)
            if idx == -1:
                break
            if not any(used[idx:idx + len(span)]):
                for k in range(idx, idx + len(span)):
                    used[k] = True
                placed.append((idx, idx + len(span), n)); break
            start = idx + 1
    placed.sort(); out, pos = [], 0
    for s, e, n in placed:
        out.append(target[pos:s]); out.append(f"<{n}>{target[s:e]}</{n}>"); pos = e
    out.append(target[pos:]); return "".join(out)


def length_filter(rows, qf, tf, pct=0.90):
    """Drop pairs whose target/query char-length ratio is an outlier (mis-windowed)."""
    import statistics
    def lens(d):
        return len(d.get(f"{qf}_string") or ""), len(d.get(f"{tf}_string") or "")
    ratios = [b / a for a, b in (lens(d) for d in rows) if a and b]
    if len(ratios) < 50:
        return rows
    med = statistics.median(ratios)
    devs = sorted(abs(math.log(r / med)) for r in ratios)
    thr = devs[int(len(devs) * pct)]
    lo, hi = med * math.exp(-thr), med * math.exp(thr)
    out = []
    for d in rows:
        a, b = lens(d)
        if a and b and lo <= b / a <= hi:
            out.append(d)
    return out


def reservoir(files, k, rng, id_set=None):
    res, n = [], 0
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if id_set is not None:                 # restrict to productive pairs
                d = json.loads(line)
                if d.get("id") not in id_set:
                    continue
                item = d
            else:
                item = line
            n += 1
            if len(res) < k:
                res.append(item)
            else:
                j = rng.randint(0, n - 1)
                if j < k:
                    res[j] = item
    return [x if isinstance(x, dict) else json.loads(x) for x in res], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=list(DIRS))
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    cfg = DIRS[args.direction]
    files = sorted(glob.glob(str(BASE / cfg["data"] / cfg["glob"])))
    out_dir = BASE / cfg["data"].split("/")[0] / f"span-mine-{args.direction}"
    rng = random.Random(SEED)

    need = N_TEST + N_VAL + N_TRAIN
    want = max(need, args.probe, args.smoke) if (args.probe or args.smoke) else need
    id_set = None
    if cfg.get("pairids"):
        id_set = set(open(BASE / cfg["pairids"], encoding="utf-8").read().split("\n"))
        id_set.discard("")
        print(f"[{args.direction}] productive-pair filter: {len(id_set)} ids", file=sys.stderr)
    sample, scanned = reservoir(files, int(want * 1.4) if want == need else want * 5, rng, id_set=id_set)
    sample = length_filter(sample, cfg["q_field"], cfg["t_field"])
    units = [x for x in (build_unit(d, cfg) for d in sample) if x]
    print(f"[{args.direction}] files={len(files)} scanned={scanned} units={len(units)}", file=sys.stderr)

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=API_KEY)
    gcfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET))

    def call(u):
        prompt = fmt_prompt(u, cfg)
        for attempt in range(5):
            try:
                r = client.models.generate_content(model=MODEL_ID, contents=prompt, config=gcfg)
                um = r.usage_metadata
                usage = {"in": getattr(um, "prompt_token_count", 0) or 0,
                         "out": (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)}
                return (r.text or "").strip(), usage, prompt
            except Exception as e:
                if attempt == 4:
                    print(f"  fail: {str(e)[:120]}", file=sys.stderr); return None, None, prompt
                time.sleep(2 ** attempt)

    if args.smoke:
        for u in units[:args.smoke]:
            out, usage, prompt = call(u)
            if out is None:
                print("(call failed - likely out of API credits)"); continue
            grounded, fid = parse_output(out, len(u["toks"]), u["target"])
            print("=" * 100)
            print("PROMPT TAIL:\n" + prompt.split("NOW DO THIS ONE")[1][:600])
            print("RAW OUTPUT:", out)
            print(f"fidelity_ok={fid}  grounded={len(grounded)}/{len(u['toks'])}")
            for n in sorted(grounded):
                print(f"   {n:2} {u['toks'][n-1][0]:14} -> {grounded[n]}")
        return

    if args.probe:
        tin = tout = c = 0
        with ThreadPoolExecutor(max_workers=10) as ex:
            for out, usage, _ in ex.map(call, units[:args.probe]):
                if usage:
                    c += 1; tin += usage["in"]; tout += usage["out"]
        ai, ao = tin / c, tout / c
        cost1 = ai / 1e6 * PRICE_IN + ao / 1e6 * PRICE_OUT
        print(f"\n[{args.direction}] PROBE {c} calls: avg in={ai:.0f} out={ao:.0f} | "
              f"per-call ${cost1:.5f} | PROJECTED {need} = ${cost1*need:.2f}")
        return

    # full run
    out_dir.mkdir(parents=True, exist_ok=True)
    rng.shuffle(units); units = units[:need]
    for i, u in enumerate(units):
        u["split"] = "test" if i < N_TEST else "val" if i < N_TEST + N_VAL else "train"
    full_fh = (out_dir / "full_io.jsonl").open("w", encoding="utf-8")
    algn_fh = (out_dir / "alignments.jsonl").open("w", encoding="utf-8")
    sfh = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    lock = threading.Lock()
    st = {"calls": 0, "examples": 0, "aligns": 0, "fidelity_ok": 0, "in": 0, "out": 0, "t0": time.time()}
    SI = PROMPTS[(cfg["q_kind"], cfg["t_kind"])].split("\nTASK:")[0].strip()

    def work(u):
        out, usage, prompt = call(u)
        if usage is None:
            return
        grounded, fid = parse_output(out, len(u["toks"]), u["target"])
        rows = [{"src": u["toks"][n - 1][0], "src_lemma": u["toks"][n - 1][1], "tgt": grounded[n],
                 "n": n, "id": u["id"], "src_segnr": u["src_segnr"], "tgt_segnr": u["tgt_segnr"],
                 "direction": args.direction, "fidelity_ok": fid, "split": u["split"]} for n in sorted(grounded)]
        tagged = reconstruct(u["target"], grounded) if grounded else None
        user_content = prompt.split("NOW DO THIS ONE\n", 1)[1].rsplit("\nOUTPUT:", 1)[0]
        with lock:
            full_fh.write(json.dumps({"id": u["id"], "split": u["split"], "direction": args.direction,
                                      "prompt": prompt, "response": out, "fidelity_ok": fid,
                                      "n_grounded": len(grounded), "usage": usage}, ensure_ascii=False) + "\n")
            for r in rows:
                algn_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if tagged:
                ex = {"messages": [{"role": "system", "content": SI},
                                   {"role": "user", "content": user_content},
                                   {"role": "assistant", "content": tagged}],
                      "meta": {"id": u["id"], "split": u["split"], "fidelity_ok": fid, "n_grounded": len(grounded)}}
                sfh[u["split"]].write(json.dumps(ex, ensure_ascii=False) + "\n"); st["examples"] += 1
            full_fh.flush(); algn_fh.flush(); [f.flush() for f in sfh.values()]
            st["calls"] += 1; st["aligns"] += len(grounded); st["fidelity_ok"] += int(fid)
            st["in"] += usage["in"]; st["out"] += usage["out"]
            if st["calls"] % 200 == 0:
                st["usd"] = round(st["in"] / 1e6 * PRICE_IN + st["out"] / 1e6 * PRICE_OUT, 2)
                st["rate"] = round(st["calls"] / (time.time() - st["t0"]), 1)
                json.dump(st, open(out_dir / "progress.json", "w"))
                print(f"  calls={st['calls']} ex={st['examples']} aligns={st['aligns']} "
                      f"fid={st['fidelity_ok']} ${st['usd']} {st['rate']}/s", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as ex:
        list(as_completed([ex.submit(work, u) for u in units]))
    full_fh.close(); algn_fh.close(); [f.close() for f in sfh.values()]
    st["usd"] = round(st["in"] / 1e6 * PRICE_IN + st["out"] / 1e6 * PRICE_OUT, 2)
    json.dump(st, open(out_dir / "progress.json", "w"))
    print(f"[{args.direction}] DONE calls={st['calls']} examples={st['examples']} "
          f"aligns={st['aligns']} fid_ok={st['fidelity_ok']} cost=${st['usd']}")


if __name__ == "__main__":
    main()
