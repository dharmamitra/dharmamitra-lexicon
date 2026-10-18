#!/usr/bin/env python3
"""Teacher self-agreement on the held-out teacher labels.

Calls the teacher a SECOND time on exactly the 1,489 held-out sentence pairs of
~/data/qwen-dictionaries-heldout500/test.jsonl, with the prompt, model and generation
config of the first run (generate_heldout.py: sa-bo / sa-zh gemini-3.6-flash, thinking
budget 16384; sa-en gemini-3.8-flash, thinking HIGH; nothing else set), and scores the
second run against the first exactly as the student is scored:

  span-level  eval_heldout.py           (same number, same span text)
  link-level  baselines/stat_baselines  (Sanskrit word, target token) links

The second run is scored as a system output, i.e. on all rows and without the seed
gates (verbatim, <=90% tagged, >=2 spans for sa-en), like the student; the figures
restricted to rows whose second run passes the gates are reported as well.

  python heldout_rerun.py --validate            # prompts reconstruct run 1 exactly? (no API calls)
  python heldout_rerun.py --direction sa-bo --smoke 3
  python heldout_rerun.py --direction sa-bo     # resumable; --limit N for the next N rows only
  python heldout_rerun.py --score               # add --partial to score an unfinished run

Outputs: heldout/run2/<dir>/gemini_raw.ndjson, heldout/run2/scores.json,
~/data/qwen-dictionaries-heldout500/test.preds-teacher-run2.jsonl
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries"
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/baselines")
import generate_heldout as H
import eval_heldout as E
G, S, SU = H.G, H.S, H.SU

RUN2 = Path(HERE) / "heldout" / "run2"
HELD = H.DATA_OUT
PREDS = HELD / "test.preds-teacher-run2.jsonl"
DIRECTIONS = ["sa-bo", "sa-zh", "sa-en"]
STUDENT_CKPT = {"sa-bo": "abl-mitra9b-3ep", "sa-zh": "abl-mitra9b-3ep", "sa-en": "five-dir"}   # checkpoint that mined the direction (v3, 2026-10-13)
CORPUS_PAIRS = {"sa-bo": 669100, "sa-zh": 416674, "sa-en": 399832}               # pairs the student aligned
PRICE_IN, PRICE_OUT = 0.75, 3.75   # USD per 1M tokens, gemini-3.6/3.8-flash standard tier, October 2026


def held_rows():
    return [json.loads(l) for l in open(HELD / "test.jsonl", encoding="utf-8")]


def units(direction):
    """id -> unit (and cfg) for the held-out rows of a direction, rebuilt as in run 1."""
    if direction == "sa-en":
        cands = [json.loads(l) for l in open(S.CANDIDATES, encoding="utf-8")]
        us, cfg = [u for u in (S.build_unit(d) for d in cands) if u], None
    else:
        cfg, us, _ = H.units_pair(direction)
    want = {r["id"] for r in held_rows() if r["lang"] == direction}
    by = {}
    for u in us:
        if u["id"] in want:
            assert u["id"] not in by, f"duplicate unit id {u['id']}"
            by[u["id"]] = u
    assert set(by) == want, f"{direction}: {len(want - set(by))} held-out rows without a unit"
    return cfg, by


def prompt(direction, u, cfg):
    return S.fmt(u) if direction == "sa-en" else G.fmt_strict(u, cfg)


def parse(direction, u, out):
    """raw teacher output -> (grounded, faithful, keep), as in generate_heldout.py."""
    if direction == "sa-en":
        out = out.splitlines()[-1].strip() if "\n" in out.strip() else out
        grounded, faithful = S.parse_output(out, len(u["toks"]), u["target"])
        keep = faithful and len(grounded) >= 2 and len(grounded) <= 0.90 * len(u["toks"])
    else:
        grounded, faithful = G.parse_output(out, len(u["toks"]), u["target"])
        keep = faithful and bool(grounded) and len(grounded) <= 0.90 * len(u["toks"])
    return out, grounded, faithful, bool(keep)


def tagged_line(direction, u, out, grounded):
    """Tagged target in the coordinates of the student rows (Unicode Tibetan for sa-bo).
    Unlike student_row_pair, a span that cannot be re-grounded is dropped, not the row."""
    grounded = {int(n): s for n, s in grounded.items()}
    if direction == "sa-en":
        return S.positional_tagged(out, u["target"], grounded) or S.reconstruct(u["target"], grounded)
    if direction == "sa-bo":
        t_uni, spans = SU.U(u["target"]), {}
        for n, span in grounded.items():
            su = SU.U(span).strip("་")
            if su and su in t_uni:
                spans[n] = su
        return G.reconstruct(t_uni, spans)
    return G.reconstruct(u["target"], grounded)


def validate():
    """Rebuild every held-out student row from the reconstructed unit and the stored
    run-1 teacher output; equality shows that the units (hence the prompts) are those of run 1."""
    ok = True
    for direction in DIRECTIONS:
        cfg, by = units(direction)
        raw1 = {}
        for l in open(Path(HERE) / "heldout" / direction / "gemini_raw.ndjson", encoding="utf-8"):
            d = json.loads(l)
            raw1.setdefault(d["id"], d)
        n = bad_in = bad_tgt = 0
        for r in held_rows():
            if r["lang"] != direction:
                continue
            n += 1
            u, d = by[r["id"]], raw1[r["id"]]
            assert d["target"] == u["target"], r["id"]
            if direction == "sa-en":
                grounded = {int(k): v for k, v in d["grounded"].items()}
                tagged = S.positional_tagged(d["raw"], u["target"], grounded) or S.reconstruct(u["target"], grounded)
                row = H.minimal_row(S.fmt(u), tagged, direction)
            else:
                row = H.student_row_pair(direction, u, cfg, d)
            bad_in += int(row is None or row["input_text"] != r["input_text"])
            bad_tgt += int(row is None or row["target_text"] != r["target_text"])
        print(f"[validate] {direction}: {n} rows, input mismatches {bad_in}, label mismatches {bad_tgt}")
        ok = ok and not bad_in and not bad_tgt
    print("[validate]", "OK: run-1 rows reproduced exactly" if ok else "MISMATCH")
    return ok


def call(client, model, prompt_text, gcfg):
    """One teacher call -> (text, token usage); None after repeated failures."""
    for a in range(6):
        try:
            resp = client.models.generate_content(model=model, contents=prompt_text, config=gcfg)
            m = resp.usage_metadata
            usage = None if m is None else {"prompt": m.prompt_token_count or 0,
                                            "thinking": m.thoughts_token_count or 0,
                                            "answer": m.candidates_token_count or 0}
            return (resp.text or "").strip(), usage
        except Exception as e:
            err = str(e)
            time.sleep(15 * (a + 1) if ("429" in err or "RESOURCE_EXHAUSTED" in err) else 2 ** a)
    return None, None


def run(direction, smoke, limit=0):
    cfg, by = units(direction)
    rows = [r for r in held_rows() if r["lang"] == direction]
    out_dir = RUN2 / direction; out_dir.mkdir(parents=True, exist_ok=True)
    rawp = out_dir / "gemini_raw.ndjson"
    done = set()
    if rawp.exists() and not smoke:
        done = {json.loads(l)["id"] for l in open(rawp, encoding="utf-8")}
    todo = [by[r["id"]] for r in rows if r["id"] not in done]
    if smoke or limit:
        todo = todo[:smoke or limit]
    print(f"[{direction}] {len(rows)} held-out rows, {len(done)} done, {len(todo)} to run "
          f"({H.TEACHER[direction]})", file=sys.stderr)
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=H.API_KEY)
    if direction == "sa-en":
        gcfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=S.THINKING_LEVEL))
    else:
        gcfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=G.THINKING_BUDGET))
    lock = threading.Lock()
    st = {"done": 0, "errs": 0, "faithful": 0, "keep": 0, "t0": time.time()}
    fh = None if smoke else open(rawp, "a", encoding="utf-8")

    def work(u):
        out, usage = call(client, H.TEACHER[direction], prompt(direction, u, cfg), gcfg)
        return u, out, usage

    with ThreadPoolExecutor(max_workers=H.WORKERS) as ex:
        for f in as_completed([ex.submit(work, u) for u in todo]):
            u, out, usage = f.result()
            with lock:
                st["done"] += 1
                if out is None:
                    st["errs"] += 1
                    continue
                out, grounded, faithful, keep = parse(direction, u, out)
                st["faithful"] += int(faithful); st["keep"] += int(keep)
                if smoke:
                    print(f"\n--- {direction} {u['id']}\nTARGET: {u['target'][:200]}\nRAW:    {out[:200]}\n"
                          f"TAGGED: {tagged_line(direction, u, out, grounded)[:200]}\n"
                          f"tagged {len(grounded)}/{len(u['toks'])} faithful={faithful} keep={keep}")
                    continue
                fh.write(json.dumps({"id": u["id"], "target": u["target"], "raw": out, "grounded": grounded,
                                     "faithful": faithful, "keep": keep, "n_toks": len(u["toks"]),
                                     "n_tagged": len(grounded), "usage": usage,
                                     "tagged": tagged_line(direction, u, out, grounded)},
                                    ensure_ascii=False) + "\n")
                if st["done"] % 50 == 0:
                    fh.flush()
                    print(f"[{direction}] {st['done']} calls faithful={st['faithful']} keep={st['keep']} "
                          f"errs={st['errs']} ({time.time()-st['t0']:.0f}s)", file=sys.stderr)
    if fh:
        fh.close()
    print(f"[{direction}] DONE calls={st['done']} faithful={st['faithful']} keep={st['keep']} errs={st['errs']}",
          file=sys.stderr)


def link_scores(direction, preds, only=None):
    """The student branch of stat_baselines.score(), applied to the second teacher run."""
    import stat_baselines as B
    held = json.load(open(B.WORK / f"{direction}.held.json", encoding="utf-8"))["held"]
    if only is not None:
        held = [h for h in held if h["row"] in only]
    L = []
    for h in held:
        r = preds[h["row"]]
        target = B.spans_from_tagged(h["gold_tagged"])[1]
        tt = (B.tok_bo_uni if direction == "sa-bo" else B.TOK[direction])(target)
        links, plain = B.links_from_tagged(r["pred"], tt)
        if re.sub(r"\s+", " ", plain).strip() != re.sub(r"\s+", " ", target).strip():
            links = set()
            for m in B.TAG.finditer(r["pred"]):
                n, span = int(m.group(1)), m.group(2).strip()
                q = target.find(span) if span else -1
                if q >= 0:
                    links |= {(n - 1, j) for j, (_, s, e) in enumerate(tt) if s < q + len(span) and e > q}
        L.append(links)
    return B.score_system(held, L)


def score(partial=False):
    rows = held_rows()
    raw2 = {}
    for direction in DIRECTIONS:
        p = RUN2 / direction / "gemini_raw.ndjson"
        for l in (open(p, encoding="utf-8") if p.exists() else []):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:      # line cut off by an interrupted run
                continue
            raw2.setdefault((direction, d["id"]), d)
    missing = [r["id"] for r in rows if (r["lang"], r["id"]) not in raw2]
    if missing and not partial:
        sys.exit(f"[score] {len(missing)} held-out rows have no second-run output yet; "
                 f"re-run the directions, or pass --partial to score the rows that exist")
    have, kept = set(), set()
    for k, r in enumerate(rows):
        d = raw2.get((r["lang"], r["id"]))
        if d is None:
            continue
        have.add(k)
        r["pred"], r["faithful2"], r["keep2"] = d["tagged"], d["faithful"], d["keep"]
        if d["keep"]:
            kept.add(k)
    if missing:
        print(f"[score] PARTIAL: {len(have)} of {len(rows)} rows have a second run")
    else:
        with open(PREDS, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("predictions ->", PREDS)
    res = {"run2": {}, "span": {}, "link": {}}
    for direction in DIRECTIONS:
        ds = [raw2[(r["lang"], r["id"])] for k, r in enumerate(rows) if r["lang"] == direction and k in have]
        res["run2"][direction] = dict(
            n=len(ds), verbatim=100 * sum(d["faithful"] for d in ds) / len(ds),
            pass_gates=100 * sum(d["keep"] for d in ds) / len(ds),
            tagged=100 * sum(d["n_tagged"] for d in ds) / sum(d["n_toks"] for d in ds))
    res["span"]["all"] = E.report([r for k, r in enumerate(rows) if k in have], "teacher run 2 vs run 1, all rows")
    res["span"]["gated"] = E.report([r for k, r in enumerate(rows) if k in kept],
                                    "teacher run 2 vs run 1, rows where run 2 passes the seed gates")
    print("\n=== link-level (stat_baselines scoring) ===")
    print(f"{'dir':7} {'rows':>5} {'P':>6} {'R':>6} {'F1':>6}   95% CI F1       | gated rows: n, F1")
    for direction in DIRECTIONS:
        a, g = link_scores(direction, rows, have), link_scores(direction, rows, kept)
        res["link"][direction] = {"all": a, "gated": g}
        print(f"{direction:7} {a['n']:5d} {a['precision']:6.1f} {a['recall']:6.1f} {a['f1']:6.1f}   "
              f"[{a['f1_ci'][0]:5.1f}, {a['f1_ci'][1]:5.1f}]   | {g['n']:4d}, {g['f1']:5.1f}")
    # the student on exactly the same rows (checkpoint that mined each direction), and on all rows
    print("\n=== student vs. run 1 on the same rows | on all held-out rows ===")
    res["student"] = {}
    for direction in DIRECTIONS:
        P = [json.loads(l) for l in open(HELD / f"test.preds-{STUDENT_CKPT[direction]}.jsonl", encoding="utf-8")]
        sub = E.score([P[k] for k in sorted(have) if rows[k]["lang"] == direction])[direction]
        full = E.score([p for p in P if p["lang"] == direction])[direction]
        ls, lf = link_scores(direction, P, have), link_scores(direction, P)
        res["student"][direction] = {"span_same_rows": sub, "span_all": full, "link_same_rows": ls, "link_all": lf}
        print(f"{direction:7} n={sub['n']:3d}  span F1 {sub['f1']:5.1f} [{sub['ci']['f1'][0]:5.1f}, {sub['ci']['f1'][1]:5.1f}]"
              f"  link F1 {ls['f1']:5.1f} [{ls['f1_ci'][0]:5.1f}, {ls['f1_ci'][1]:5.1f}]"
              f"   | all: span {full['f1']:5.1f}  link {lf['f1']:5.1f}")
    # measured token usage of the teacher and the cost of labelling the whole corpora with it
    print("\n=== teacher token usage (calls with recorded usage) and cost of labelling the full corpora ===")
    res["cost"] = {}
    tot = 0.0
    for direction in DIRECTIONS:
        us = [d["usage"] for (dd, _), d in raw2.items() if dd == direction and d.get("usage")]
        if not us:
            continue
        m = {k: sum(u[k] for u in us) / len(us) for k in ("prompt", "thinking", "answer")}
        per_call = (m["prompt"] * PRICE_IN + (m["thinking"] + m["answer"]) * PRICE_OUT) / 1e6
        full = per_call * CORPUS_PAIRS[direction]; tot += full
        res["cost"][direction] = dict(n=len(us), mean_tokens=m, usd_per_call=per_call,
                                      corpus_pairs=CORPUS_PAIRS[direction], usd_corpus=full)
        print(f"{direction:7} n={len(us):3d}  prompt {m['prompt']:6.0f}  thinking {m['thinking']:6.0f}  answer {m['answer']:4.0f}"
              f"  -> ${per_call:.4f}/call  x {CORPUS_PAIRS[direction]:,} pairs = ${full:,.0f}")
    res["cost"]["total_usd"] = tot
    print(f"total ${tot:,.0f} at ${PRICE_IN}/M input, ${PRICE_OUT}/M output incl. thinking (batch: half)")
    print("\nsecond-run stats:", {d: {k: round(v, 1) for k, v in s.items()} for d, s in res["run2"].items()})
    outp = RUN2 / ("scores.partial.json" if missing else "scores.json")
    json.dump(res, open(outp, "w"), indent=1)
    print("scores ->", outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=DIRECTIONS)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="run at most this many further rows")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--partial", action="store_true", help="with --score: score only the rows run so far")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate() else 1)
    if a.score:
        return score(a.partial)
    if not a.direction:
        ap.error("need --direction, --validate or --score")
    run(a.direction, a.smoke, a.limit)


if __name__ == "__main__":
    main()
