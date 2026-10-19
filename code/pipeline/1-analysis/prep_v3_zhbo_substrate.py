#!/usr/bin/env python3
"""zh-bo (2026-10-19): MITRA-parallel v3 bo-zh pairs -> mining substrate for mine_v4.py --direction zh-bo-v3.

The Chinese source side is segmented exactly as in the zh-bo training seeds and the July zh-bo mine: the Chinese is
rendered into kundoku Japanese by the kokuyaku model (sa-zh/kokuyaku_translate_hf.py), the kana are dropped and the
kanji runs become the words (sa-zh/segment_zh_from_japanese.py). The translation uses greedy decoding but is not
exactly deterministic (a rerun on 200 strings reproduced 175 segmentations). A Chinese string that already carries a
segmentation in the July zh-bo substrate reuses it; the others are translated.

  python prep_v3_zhbo_substrate.py todo      -> zh-bo/kundoku/todo.<k>.ndjson (distinct Chinese strings to translate)
  (kokuyaku_translate_hf.py + segment_zh_from_japanese.py on the todo shards, in place)
  python prep_v3_zhbo_substrate.py write     -> zh-bo/substrate.ndjson + substrate.stats.json
"""
import glob, gzip, json, os, sys, collections

V3 = "/mnt2/code/mitra-multilingual-matching/mitra-parallel-v3/bo-zh_matches.ndjson.gz"
OLD = "/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026/zh-bo/sentence-pairs/bo-zh_matches_*.filtered.ndjson"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "zh-bo")
KDIR = os.path.join(OUT, "kundoku")
NSHARD = 4
key = lambda s: (s or "").strip()


def old_segmentation():
    seg = {}
    for f in sorted(glob.glob(OLD)):
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            if d.get("par_zh_segmented"):
                seg[key(d["par_string"])] = d["par_zh_segmented"]
    return seg


def v3_records():
    with gzip.open(V3, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def todo():
    os.makedirs(KDIR, exist_ok=True)
    seg = old_segmentation()
    need = []
    seen = set()
    for d in v3_records():
        k = key(d.get("par_string"))
        if k and k not in seg and k not in seen:
            seen.add(k); need.append(k)
    fs = [open(os.path.join(KDIR, f"todo.{i}.ndjson"), "w", encoding="utf-8") for i in range(NSHARD)]
    for i, k in enumerate(need):
        fs[i % NSHARD].write(json.dumps({"par_string": k}, ensure_ascii=False) + "\n")
    for f in fs:
        f.close()
    print(f"reusable segmentations: {len(seg)}  to translate: {len(need)} distinct Chinese strings")


def write():
    seg = old_segmentation()
    st = collections.Counter()
    for p in sorted(glob.glob(os.path.join(KDIR, "todo.*.ndjson"))):
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            if d.get("par_zh_segmented"):
                seg.setdefault(key(d["par_string"]), d["par_zh_segmented"]); st["segmentation_new"] += 1
    with open(os.path.join(OUT, "substrate.ndjson"), "w", encoding="utf-8") as o:
        for d in v3_records():
            st["records"] += 1
            s = seg.get(key(d.get("par_string")))
            if not s:
                st["skip_no_segmentation"] += 1
                continue
            d["par_zh_segmented"] = s
            o.write(json.dumps(d, ensure_ascii=False) + "\n")
            st["written"] += 1
    json.dump(dict(st), open(os.path.join(OUT, "substrate.stats.json"), "w"), indent=1)
    print(dict(st))


if __name__ == "__main__":
    {"todo": todo, "write": write}[sys.argv[1]]()
