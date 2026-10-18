#!/usr/bin/env python3
"""v4 (2026-10-16): split the MITRA-parallel v3 substrate into records already mined in v3 and records to mine.

A v4 record is REUSED only when the v3 run mined a record with the exact same model input (prompt string, rebuilt
with the same gen_inputs code from the v3 substrate) AND the same Sanskrit and target segment numbers. The v3 mined row
is then copied with the v4 id. Everything else (new pairs, changed bead shape, changed segmentation) is mined again.
Same checkpoint as v3 (/mnt/data/abl-mitra9b-3ep-mm), greedy decoding, so a reused row equals what re-mining would give.

  python delta.py sa-bo sa-zh
    <dir>/reused.ndjson            v3 mined rows for unchanged records (id rewritten)
    <dir>/substrate.delta.ndjson   v4 substrate rows to mine
    <dir>/delta.stats.json
"""
import hashlib, json, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mine_v4
V3 = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v3"
KEYDIR = {"sa-bo": "sa-bo-v2", "sa-zh": "sa-zh-v2"}


def key(it):
    h = hashlib.sha1(it["prompt"].encode("utf-8"))
    h.update(json.dumps([it["src_segnr"], it["tgt_segnr"]]).encode("utf-8"))
    return h.digest()


def main():
    for direction in sys.argv[1:]:
        dk, d3, d4 = KEYDIR[direction], os.path.join(V3, direction), os.path.join(HERE, direction)
        st = collections.Counter()
        # v3 prompts by id (v3 substrate.ndjson = post-split, the records the final v3 rows were mined on)
        old_key = {it["id"]: key(it) for it in mine_v4.gen_inputs(dk, input_file=os.path.join(d3, "substrate.ndjson"))}
        old_row = {}
        for k in (0, 1):
            for line in open(os.path.join(d3, f"mined.ndjson.part{k}"), encoding="utf-8"):
                rid = json.loads(line)["id"]
                kk = old_key.get(rid)
                if kk is None:
                    st["v3_row_without_prompt"] += 1; continue
                old_row[kk] = line
        st["v3_rows_indexed"] = len(old_row)
        del old_key
        # v4 substrate: which records produce a prompt identical to a v3 one
        new_items = {}
        for it in mine_v4.gen_inputs(dk, input_file=os.path.join(d4, "substrate.ndjson")):
            new_items[it["id"]] = key(it)
        st["v4_prompts"] = len(new_items)
        with open(os.path.join(d4, "substrate.ndjson"), encoding="utf-8") as fi, \
             open(os.path.join(d4, "reused.ndjson"), "w", encoding="utf-8") as fr, \
             open(os.path.join(d4, "substrate.delta.ndjson"), "w", encoding="utf-8") as fd:
            for line in fi:
                st["v4_records"] += 1
                rid = json.loads(line)["id"]
                kk = new_items.get(rid)
                if kk is None:                      # gen_inputs skips it (no tokens / too long): nothing to mine either way
                    st["v4_no_prompt"] += 1; continue
                hit = old_row.get(kk)
                if hit is not None:
                    row = json.loads(hit); row["id"] = rid
                    fr.write(json.dumps(row, ensure_ascii=False) + "\n"); st["reused"] += 1
                else:
                    fd.write(line); st["to_mine"] += 1
        json.dump(dict(st), open(os.path.join(d4, "delta.stats.json"), "w"), indent=1)
        print(direction, dict(st), flush=True)


if __name__ == "__main__":
    main()
