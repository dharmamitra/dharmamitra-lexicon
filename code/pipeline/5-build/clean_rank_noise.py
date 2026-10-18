#!/usr/bin/env python3
"""Filter rank noise from a bidict direction:
  - drop senses whose GLOSS contains a clause-boundary marker (shad / | for Tibetan,
    sentence punctuation for Chinese) -- a gloss should never span a boundary;
  - drop senses with PMI below PMIMIN (generic-word attachments that merely co-occur);
  - drop whole headwords that are themselves boundary artifacts.
Re-sorts by count, updates n/n_senses. Backup: <file>.prenoise.bak

Usage: clean_rank_noise.py <bidict.ndjson> <head_key> <sense_key> <boundary_chars>
"""
import json, math, collections, shutil, sys, os

PMIMIN = float(os.environ.get("RANKNOISE_PMI", "1.0"))
path, hkey, skey, BCHARS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
has_boundary = lambda s: any(c in s for c in BCHARS)

rows = [json.loads(l) for l in open(path, encoding="utf-8")]
F_g = collections.Counter(); F_h = {}; N = 0
for d in rows:
    F_h[d[hkey]] = d["n"]; N += d["n"]
    for s in d["senses"]:
        F_g[s[skey]] += s["count"]

kept_rows = 0; dropped_heads = 0; dropped_senses = 0; tot_senses = 0
with open(path + ".tmp", "w", encoding="utf-8") as o:
    for d in rows:
        tot_senses += len(d["senses"])
        if has_boundary(d[hkey]):           # headword itself is a boundary artifact
            dropped_heads += 1; dropped_senses += len(d["senses"]); continue
        keep = []
        for s in d["senses"]:
            g = s[skey]
            if has_boundary(g):
                continue
            pmi = math.log2(s["count"] * N / (F_h[d[hkey]] * F_g[g])) if F_g[g] else 0
            if pmi < PMIMIN:
                continue
            keep.append(s)
        dropped_senses += len(d["senses"]) - len(keep)
        if not keep:
            dropped_heads += 1; continue
        keep.sort(key=lambda s: -s["count"])
        d["senses"] = keep; d["n"] = sum(s["count"] for s in keep); d["n_senses"] = len(keep)
        o.write(json.dumps(d, ensure_ascii=False) + "\n"); kept_rows += 1

shutil.copyfile(path, path + ".prenoise.bak")
os.replace(path + ".tmp", path)
print(f"{os.path.basename(path)}: kept {kept_rows} headwords (dropped {dropped_heads}), "
      f"dropped {dropped_senses}/{tot_senses} senses ({100*dropped_senses/tot_senses:.1f}%)")
