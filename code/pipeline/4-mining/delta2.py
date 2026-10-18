#!/usr/bin/env python3
"""v4 round 2 (2026-10-16): MITRA-parallel v3 was rebuilt (triangulation + the 106 missing spans, recon-postcorrected-fix106).
Re-prepped substrate: r2/<dir>/substrate.ndjson (prep into r2/ so the files the running miners read stay untouched).

A rebuilt record needs mining only if NO already-covered record has the same key (rebuilt prompt + segnrs, as delta.py):
covered = v3 mined rows  +  every record of each COVER substrate (v4 delta substrates already mined / being mined).

  python delta2.py sa-zh --cover sa-zh/substrate.delta.ndjson   -> sa-zh/substrate.delta2.ndjson
  python delta2.py sa-bo                                        -> sa-bo/substrate.delta2.ndjson
"""
import argparse, json, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mine_v4
from delta import key, KEYDIR, V3

ap = argparse.ArgumentParser()
ap.add_argument("direction")
ap.add_argument("--cover", action="append", default=[], help="substrate file(s) whose records are/will be mined in v4")
a = ap.parse_args()
d, dk = a.direction, KEYDIR[a.direction]
st = collections.Counter()
covered = {key(it) for it in mine_v4.gen_inputs(dk, input_file=os.path.join(V3, d, "substrate.ndjson"))}
st["v3_keys"] = len(covered)
for c in a.cover:
    for it in mine_v4.gen_inputs(dk, input_file=os.path.join(HERE, c)):
        covered.add(key(it)); st["cover_keys"] += 1
new = {it["id"]: key(it) for it in mine_v4.gen_inputs(dk, input_file=os.path.join(HERE, "r2", d, "substrate.ndjson"))}
with open(os.path.join(HERE, "r2", d, "substrate.ndjson"), encoding="utf-8") as fi, \
     open(os.path.join(HERE, d, "substrate.delta2.ndjson"), "w", encoding="utf-8") as fo:
    for line in fi:
        st["r2_records"] += 1
        k = new.get(json.loads(line)["id"])
        if k is None: st["r2_no_prompt"] += 1
        elif k in covered: st["covered"] += 1
        else: fo.write(line); st["to_mine"] += 1
json.dump(dict(st), open(os.path.join(HERE, d, "delta2.stats.json"), "w"), indent=1)
print(d, dict(st), flush=True)
