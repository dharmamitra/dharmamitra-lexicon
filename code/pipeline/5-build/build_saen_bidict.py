#!/usr/bin/env python3
"""sa-en bidict WITH offsets (into the English) from sa-en.spans.ndjson.
  skt_en_bidict.sa2en.ndjson : {"sa", senses:[{en,count,ex:[{id,target,start,end,src}]}], n, n_senses}
  skt_en_bidict.en2sa.ndjson : {"en", senses:[{sa,count,ex:[...]}], n, n_senses}
  skt_en_spans.ndjson        : full per-span occurrence list (copied)
English sense/head keys are case-folded (King/king) with the verbatim span kept in
the example offsets. Mitrasamgraha has no segment numbers, so each example carries
the Sanskrit source sentence (`src`) and its id encodes split/license/file:line.
"""
import json, collections, shutil, os
BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
SPANS = os.environ.get("SPANS_OUT") or f"{BASE}/sa-en/hybrid/sa-en.spans.ndjson"
OUT = os.environ.get("OUTDIR", f"{BASE}/final-output"); K = None   # no cap: every attestation is kept (2026-10-11; was 8)
sa2en = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set()]))
en2sa = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, [], set()]))


def en_key(s):
    s = " ".join(s.split()).lower()
    return s[:-1] if s.endswith(",") else s


def add(t, h, s, ex):
    c = t[h][s]; c[0] += 1
    if (ex["id"], ex["start"], ex["end"]) not in c[2]:  # one attestation per occurrence
        c[1].append(ex); c[2].add((ex["id"], ex["start"], ex["end"]))


for line in open(SPANS, encoding="utf-8"):
    e = json.loads(line); key = e.get("head", e["lemma"]); en = en_key(e["en"])
    if not key or not en: continue
    ex = {"id": e["sentence_id"], "target": e["target"], "start": e["start"], "end": e["end"],
          "src": e["src_sentence"], "license": e.get("license", "")}
    if e.get("low_coverage"): ex["low_coverage"] = True
    add(sa2en, key, en, ex); add(en2sa, en, key, ex)


def dump(path, hk, sk, t):
    if os.path.exists(path) and not os.path.exists(path + ".prehybrid.bak"):
        shutil.copyfile(path, path + ".prehybrid.bak")
    with open(path, "w", encoding="utf-8") as o:
        for h in sorted(t, key=lambda h: -sum(v[0] for v in t[h].values())):
            senses = sorted(t[h].items(), key=lambda kv: (-kv[1][0], kv[0]))
            arr = [{sk: s, "count": v[0], "ex": sorted(v[1], key=lambda x: len(x["target"]))[:K]} for s, v in senses]
            o.write(json.dumps({hk: h, "senses": arr, "n": sum(v[0] for v in t[h].values()), "n_senses": len(arr)}, ensure_ascii=False) + "\n")


os.makedirs(OUT, exist_ok=True)
dump(f"{OUT}/skt_en_bidict.sa2en.ndjson", "sa", "en", sa2en)
dump(f"{OUT}/skt_en_bidict.en2sa.ndjson", "en", "sa", en2sa)
shutil.copyfile(SPANS, f"{OUT}/skt_en_spans.ndjson")
print(f"sa2en headwords: {len(sa2en)}   en2sa headwords: {len(en2sa)}   (K={K})")
