#!/usr/bin/env python3
"""Sharded-parallel driver for the (unmodified-logic) v1 spans builders.

Splits each direction's final.ndjson into N shards — deduping by the same
(id, target, toks) key the scripts use, so cross-shard duplicates can't slip
through — runs N subprocesses of the direction's spans script via the
SPANS_IN/SPANS_OUT (or ENBO_*) env overrides, concatenates shard outputs into
the canonical hybrid/<pair>.spans.ndjson, and sums the per-shard counters.
"""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
BUILD = os.path.join(BASE, "build")
NSHARD = int(os.environ.get("NSHARD", "32"))
SCRIPTS = {
    "sa-zh": ("build_sazh_spans.py", "SPANS_IN", "SPANS_OUT"),
    "sa-bo": ("build_sabo_spans.py", "SPANS_IN", "SPANS_OUT"),
    "zh-bo": ("build_zhbo_spans.py", "SPANS_IN", "SPANS_OUT"),
    "en-bo": ("build_enbo_spans.py", "ENBO_FINAL", "ENBO_SPANS"),
    "en-bo-dnx": ("build_enbo_spans.py", "ENBO_FINAL", "ENBO_SPANS"),
    "en-bo-lh": ("build_enbo_spans.py", "ENBO_FINAL", "ENBO_SPANS"),
    "en-bo-monlam": ("build_enbo_spans.py", "ENBO_FINAL", "ENBO_SPANS"),
}


def run(direction):
    script, in_env, out_env = SCRIPTS[direction]
    final = f"{BASE}/{direction}/hybrid/final.ndjson"
    spans = f"{BASE}/{direction}/hybrid/{direction}.spans.ndjson"
    tmp = tempfile.mkdtemp(prefix=f"spans_{direction}_")
    shard_f = [open(os.path.join(tmp, f"in{i}.ndjson"), "w", encoding="utf-8") for i in range(NSHARD)]
    seen = set()
    n_in = n_dup = 0
    for line in open(final, encoding="utf-8"):
        r = json.loads(line)
        key = hash((r["id"], r["target"], tuple(tuple(t) for t in r["toks"])))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        shard_f[n_in % NSHARD].write(line)
        n_in += 1
    for f in shard_f:
        f.close()
    print(f"[{direction}] {n_in} rows into {NSHARD} shards ({n_dup} dups dropped)", flush=True)

    def one(i):
        env = dict(os.environ)
        env[in_env] = os.path.join(tmp, f"in{i}.ndjson")
        env[out_env] = os.path.join(tmp, f"out{i}.ndjson")
        p = subprocess.run([sys.executable, os.path.join(BUILD, script)],
                           env=env, cwd=BUILD, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{direction} shard {i}: {p.stderr[-800:]}")
        return p.stdout

    with ThreadPoolExecutor(NSHARD) as ex:
        outs = list(ex.map(one, range(NSHARD)))
    tok = phr = fail = span = 0
    for o in outs:
        for ln in o.splitlines():
            if ln.startswith("token spans:"):
                tok += int(ln.split()[2]); phr += int(ln.split()[5])
            elif ln.startswith("cursor-match failures:"):
                a, b = ln.split()[2].split("/"); fail += int(a); span += int(b)
    with open(spans, "w", encoding="utf-8") as out:
        for i in range(NSHARD):
            for line in open(os.path.join(tmp, f"out{i}.ndjson"), encoding="utf-8"):
                out.write(line)
    for i in range(NSHARD):
        os.remove(os.path.join(tmp, f"in{i}.ndjson")); os.remove(os.path.join(tmp, f"out{i}.ndjson"))
    os.rmdir(tmp)
    print(f"[{direction}] token spans: {tok}  phrase spans: {phr}  "
          f"cursor failures: {fail}/{span} ({100*fail/max(span,1):.3f}%)  -> {spans}", flush=True)


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["sa-zh", "sa-bo", "zh-bo", "en-bo"]):
        run(d)
