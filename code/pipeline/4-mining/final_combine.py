#!/usr/bin/env python3
"""v4 final combine (2026-10-16): one mined row per record of the REBUILT substrate (r2/<dir>/substrate.ndjson).
Row sources, looked up by key (rebuilt prompt + segnrs, see delta.py): v3 mined rows, then every v4 (substrate, rows)
pair given. Rows mined for records that the rebuild removed are left out. Writes <dir>/mined.ndjson.part0/part1
(alternating; prepare_finals.py reads both) + <dir>/combine.stats.json.

  python final_combine.py sa-zh substrate.delta.ndjson:mined_delta.ndjson substrate.delta2.ndjson:mined_delta2.ndjson
"""
import glob, json, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mine_v4
from delta import key, KEYDIR, V3

d, dk = sys.argv[1], KEYDIR[sys.argv[1]]
st = collections.Counter()
rows = {}


def add(substrate, row_files, tag):
    idkey = {it["id"]: key(it) for it in mine_v4.gen_inputs(dk, input_file=substrate)}
    for f in row_files:
        for line in open(f, encoding="utf-8"):
            k = idkey.get(json.loads(line)["id"])
            if k is not None:
                rows[k] = line; st[f"rows_{tag}"] += 1


add(os.path.join(V3, d, "substrate.ndjson"), [os.path.join(V3, d, f"mined.ndjson.part{k}") for k in (0, 1)], "v3")
for spec in sys.argv[2:]:      # later sources override v3 (identical key -> identical model input anyway)
    sub, out = spec.split(":")
    add(os.path.join(HERE, d, sub), sorted(glob.glob(os.path.join(HERE, d, out + ".part*"))), sub)
outs = [open(os.path.join(HERE, d, f"mined.ndjson.part{k}"), "w", encoding="utf-8") for k in (0, 1)]
n = 0
for it in mine_v4.gen_inputs(dk, input_file=os.path.join(HERE, "r2", d, "substrate.ndjson")):
    line = rows.get(key(it))
    if line is None:
        st["missing_row"] += 1; continue
    r = json.loads(line); r["id"] = it["id"]
    outs[n % 2].write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
for o in outs: o.close()
st["written"] = n
json.dump(dict(st), open(os.path.join(HERE, d, "combine.stats.json"), "w"), indent=1)
print(d, dict(st), flush=True)
