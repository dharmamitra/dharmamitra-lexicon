#!/usr/bin/env python3
"""v3 regeneration, stage 1: MITRA-parallel v2 (sa-bo, sa-zh) -> mining substrate in the row shape that
mine_direction.py (sa-bo) and mine_sazh_v3.py (sa-zh) consume.

Sanskrit side: the ByT5 analysis (`full_analysis`: unsandhied / lemma / tag) is taken from the DharmaNexus Sanskrit
segments of the record's root_segnr, concatenated in order -- the same analyser and schema as the v1 substrate. A record
is skipped if any of its Sanskrit segments has no analysis.
Chinese side (sa-zh): CHINESE_SEGMENTED = the DharmaNexus word segmentation (`analyzed`) of the par_segnr segments.
(Tested 2026-10-07: on the 496 held-out sa-zh pairs the student scores 87.6 span F1 with this segmentation against 87.4
with the kundoku-derived segmentation of the training data.)
Tibetan side (sa-bo): Wylie par_string as released (mine_direction.py converts to Unicode at the prompt boundary).

  python prep_v2_substrate.py sa-bo sa-zh   ->  <dir>/substrate.ndjson + <dir>/substrate.stats.json
"""
import glob, gzip, json, os, re, sys, collections

V2 = "/mnt2/code/mitra-multilingual-matching/mitra-parallel-v3"
SA_SEG = os.path.expanduser("~/data/dharmanexus-sanskrit/segments")
ZH_SEG = os.path.expanduser("~/data/dharmanexus-chinese/segments")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        import ast
        return ast.literal_eval(v)
    return [v] if v else []


class SegIndex:
    """lazy per-file index of DharmaNexus segments by segmentnr."""
    def __init__(self, root, file_of):
        self.root, self.file_of, self.cache = root, file_of, {}

    def get(self, segnr):
        f = self.file_of(segnr)
        if f is None:
            return None
        if f not in self.cache:
            d = {}
            for p in glob.glob(os.path.join(self.root, f + ".json")) + glob.glob(os.path.join(self.root, f + "$*.json")):
                for s in json.load(open(p, encoding="utf-8")):
                    d[s["segmentnr"]] = s
            self.cache[f] = d
            if len(self.cache) > 400:                 # bound memory: drop the oldest file
                self.cache.pop(next(iter(self.cache)))
        return self.cache[f].get(segnr)


sa_index = SegIndex(SA_SEG, lambda s: s.split(":")[0])


def zh_file(s):
    m = re.match(r"(ZH_.+?_\d{3}):", s)
    return m.group(1) if m else s.split(":")[0]


zh_index = SegIndex(ZH_SEG, zh_file)


def main():
    for direction in sys.argv[1:]:
        src = os.path.join(V2, f"{direction}_matches.ndjson.gz")
        out_dir = os.path.join(HERE, direction)
        os.makedirs(out_dir, exist_ok=True)
        st = collections.Counter()
        with gzip.open(src, "rt", encoding="utf-8") as f, open(os.path.join(out_dir, "substrate.ndjson"), "w", encoding="utf-8") as o:
            for line in f:
                d = json.loads(line)
                st["records"] += 1
                rs, ps = lst(d.get("root_segnr")), lst(d.get("par_segnr"))
                segs = [sa_index.get(s) for s in rs]
                if not segs or any(s is None for s in segs):
                    st["skip_sanskrit_segment_missing"] += 1
                    continue
                fa = [t for s in segs for t in (s.get("full_analysis") or [])]
                if not fa or any(not s.get("full_analysis") for s in segs):
                    st["skip_sanskrit_analysis_missing"] += 1
                    continue
                row = {"id": d["id"], "root_segnr": rs, "par_segnr": ps,
                       "root_string": d.get("root_string", ""), "par_string": d.get("par_string", ""),
                       "score": d.get("score"), "full_analysis": fa}
                if direction == "sa-zh":
                    zs = [zh_index.get(s) for s in ps]
                    if not zs or any(z is None for z in zs):
                        st["skip_chinese_segment_missing"] += 1
                        continue
                    seg = " ".join(" ".join((z.get("analyzed") or z.get("original", "")).split()) for z in zs)
                    if not seg.strip():
                        st["skip_chinese_segmentation_empty"] += 1
                        continue
                    row["par_zh_segmented"] = seg
                o.write(json.dumps(row, ensure_ascii=False) + "\n")
                st["written"] += 1
                if st["records"] % 50000 == 0:
                    print(direction, dict(st), flush=True)
        json.dump(dict(st), open(os.path.join(out_dir, "substrate.stats.json"), "w"), indent=1)
        print(direction, "DONE", dict(st), flush=True)


if __name__ == "__main__":
    main()
