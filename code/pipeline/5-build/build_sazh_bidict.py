#!/usr/bin/env python3
"""sa-bo bidict WITH offsets (into Tibetan) for frontend highlighting. Replaces the
old mined skt_zh_bidict with the hybrid+offset version (backup kept as .prehybrid.bak).
  skt_zh_bidict.sa2zh.ndjson : {"sa", senses:[{bo,count,ex:[{id,tgt_segnr,target,start,end}]}], n, n_senses}
  skt_zh_bidict.zh2sa.ndjson : {"bo", senses:[{sa,count,ex:[...]}], n, n_senses}
  skt_zh_spans.ndjson        : full per-span occurrence list (copied)
"""
import json, collections, shutil, os
BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
SPANS = f"{BASE}/sa-zh/hybrid/sa-zh.spans.ndjson"
OUT = os.environ.get("OUTDIR", f"{BASE}/final-output"); K = None   # no cap: every attestation is kept (2026-10-11; was 8)
sa2zh = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set(), 0]))
zh2sa = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set(), 0]))
def add(t, h, s, ex, relaxed=False):
    c = t[h][s]; c[0] += 1                       # count = occurrences
    c[3] += relaxed                              # occurrences from relaxed compound windows (rule M)
    if (ex["id"], ex["start"], ex["end"]) not in c[2]:  # one attestation per occurrence
        c[1].append(ex); c[2].add((ex["id"], ex["start"], ex["end"]))
for line in open(SPANS, encoding="utf-8"):
    e = json.loads(line); key = e.get("head", e["lemma"]); zh = e["zh"]
    if not key or not zh: continue
    ex = {"id": e["sentence_id"], "tgt_segnr": e["tgt_segnr"], "target": e["target"],
          "start": e["start"], "end": e["end"]}
    rx = "rule" in e
    add(sa2zh, key, zh, ex, rx); add(zh2sa, zh, key, ex, rx)   # both directions -> fully bidirectional
def dump(path, hk, sk, t):
    for old in (path, path):  # backup existing once
        if os.path.exists(path) and not os.path.exists(path + ".prehybrid.bak"):
            shutil.copyfile(path, path + ".prehybrid.bak")
    with open(path, "w", encoding="utf-8") as o:
        for h in sorted(t, key=lambda h: -sum(v[0] for v in t[h].values())):
            senses = sorted(t[h].items(), key=lambda kv: (-kv[1][0], kv[0]))
            arr = [dict({sk: s, "count": v[0], "ex": sorted(v[1], key=lambda x: len(x["target"]))[:K]},
                        **({"relaxed": v[3]} if v[3] else {})) for s, v in senses]
            o.write(json.dumps({hk: h, "senses": arr, "n": sum(v[0] for v in t[h].values()), "n_senses": len(arr)}, ensure_ascii=False) + "\n")
dump(f"{OUT}/skt_zh_bidict.sa2zh.ndjson", "sa", "zh", sa2zh)
dump(f"{OUT}/skt_zh_bidict.zh2sa.ndjson", "zh", "sa", zh2sa)
shutil.copyfile(SPANS, f"{OUT}/skt_zh_spans.ndjson")
print(f"sa2zh headwords: {len(sa2zh)}   zh2sa headwords: {len(zh2sa)}   (K={K})")
