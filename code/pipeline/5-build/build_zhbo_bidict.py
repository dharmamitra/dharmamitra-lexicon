#!/usr/bin/env python3
"""zh-bo bidict WITH offsets (into Tibetan) for highlighting. Replaces the old mined
bo_zh_bidict with the hybrid+offset version (.prehybrid.bak kept).
  bo_zh_bidict.zh2bo.ndjson : {"zh", senses:[{bo,count,ex:[{id,tgt_segnr,target,start,end}]}], n, n_senses}
  bo_zh_bidict.bo2zh.ndjson : {"bo", senses:[{zh,count,ex:[...]}], n, n_senses}
  bo_zh_spans.ndjson        : full per-span occurrence list (copied)
"""
import json, collections, shutil, os
BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
SPANS = f"{BASE}/zh-bo/hybrid/zh-bo.spans.ndjson"
OUT = os.environ.get("OUTDIR", f"{BASE}/final-output"); K = None   # no cap: every attestation is kept (as sa-bo/sa-zh v4; was 8)
zh2bo = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set()]))
bo2zh = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set()]))
def add(t, h, s, ex):
    c = t[h][s]; c[0] += 1
    if (ex["id"], ex["start"], ex["end"]) not in c[2]:  # one attestation per occurrence (as sa-bo v4)
        c[1].append(ex); c[2].add((ex["id"], ex["start"], ex["end"]))
for line in open(SPANS, encoding="utf-8"):
    e = json.loads(line); key = e.get("head", e["zh"]); bo = e["bo"]
    if not key or not bo: continue
    ex = {"id": e["sentence_id"], "tgt_segnr": e["tgt_segnr"], "target": e["target"],
          "start": e["start"], "end": e["end"], "indices": e["indices"]}   # indices: 1-based Chinese word positions
    add(zh2bo, key, bo, ex); add(bo2zh, bo, key, ex)   # both directions -> fully bidirectional
def dump(path, hk, sk, t):
    if os.path.exists(path) and not os.path.exists(path + ".prehybrid.bak"):
        shutil.copyfile(path, path + ".prehybrid.bak")
    with open(path, "w", encoding="utf-8") as o:
        for h in sorted(t, key=lambda h: -sum(v[0] for v in t[h].values())):
            senses = sorted(t[h].items(), key=lambda kv: (-kv[1][0], kv[0]))
            arr = [{sk: s, "count": v[0], "ex": sorted(v[1], key=lambda x: len(x["target"]))[:K]} for s, v in senses]
            o.write(json.dumps({hk: h, "senses": arr, "n": sum(v[0] for v in t[h].values()), "n_senses": len(arr)}, ensure_ascii=False) + "\n")
dump(f"{OUT}/bo_zh_bidict.zh2bo.ndjson", "zh", "bo", zh2bo)
dump(f"{OUT}/bo_zh_bidict.bo2zh.ndjson", "bo", "zh", bo2zh)
shutil.copyfile(SPANS, f"{OUT}/bo_zh_spans.ndjson")
print(f"zh2bo headwords: {len(zh2bo)}   bo2zh headwords: {len(bo2zh)}   (K={K})")
