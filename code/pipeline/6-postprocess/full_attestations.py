#!/usr/bin/env python3
"""Replace the capped example lists (8 per sense) of the released dictionaries by the complete list of
attestations (2026-10-11).

Headwords, senses, counts and order are taken unchanged from SRC_DIR; only `ex` is rebuilt from the spans file of
the pair (which holds every aligned occurrence, already ṁ->ṃ normalised by postprocess_final.py). One attestation
per occurrence, i.e. per Sanskrit token (deduplicated on sentence id, Sanskrit token indices, start, end; the indices
are kept, so that two occurrences of a headword in one sentence remain distinct), ordered by length of the translation (shortest first,
as before; ties in corpus order). Attestation fields are those of the build_*_bidict.py scripts.
Checks that every sense has exactly `count` attestations and reports deviations.

Usage: full_attestations.py SRC_DIR DST_DIR
"""
import json, os, sys, shutil, unicodedata
from collections import defaultdict, Counter
from multiprocessing import Pool

SRC, DST = sys.argv[1], sys.argv[2]
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):                 # e.g. ONLY_PAIRS=skt_tib: rebuild one pair only
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]


def norm(s):                                     # = postprocess_final.norm
    return unicodedata.normalize("NFC", s).replace("ṁ", "ṃ").replace("Ṁ", "Ṃ")


def en_key(s):                                   # = build_saen_bidict.en_key
    s = " ".join(s.split()).lower()
    return s[:-1] if s.endswith(",") else s


def ex_of(e, tag):
    if tag == "en":
        x = {"id": e["sentence_id"], "target": e["target"], "start": e["start"], "end": e["end"],
             "src": e["src_sentence"], "license": e.get("license", "")}
        if e.get("low_coverage"): x["low_coverage"] = True
    else:
        x = {"id": e["sentence_id"], "tgt_segnr": e["tgt_segnr"], "target": e["target"],
             "start": e["start"], "end": e["end"]}
    if e.get("indices"): x["indices"] = e["indices"]
    return x


def process(pt):
    pfx, tag = pt
    need = set()
    for d in ("sa2" + tag, tag + "2sa"):
        for l in open(f"{SRC}/{pfx}_bidict.{d}.ndjson"):
            e = json.loads(l)
            if d.startswith("sa"):
                need.update((e["sa"], s[tag]) for s in e["senses"])
            else:
                need.update((s["sa"], e[tag]) for s in e["senses"])
    att, seen = defaultdict(list), set()
    for i, l in enumerate(open(f"{SRC}/{pfx}_spans.ndjson")):
        e = json.loads(l); h = e.get("head", e["lemma"]); g = e[tag]
        if not h or not g: continue
        h = norm(h)                              # as postprocess_final.norm (37 sa-bo heads are not NFC in the spans)
        if tag == "en": g = en_key(g)
        k = (h, g)
        if k not in need: continue
        u = (h, g, e["sentence_id"], e["start"], e["end"], tuple(e.get("indices") or ()))
        if u in seen: continue
        seen.add(u)
        x = ex_of(e, tag)
        att[k].append((len(x["target"]), i, json.dumps(x, ensure_ascii=False)))
    del seen
    for v in att.values(): v.sort()
    stats = Counter()
    for d in ("sa2" + tag, tag + "2sa"):
        with open(f"{DST}/{pfx}_bidict.{d}.ndjson", "w") as o:
            for l in open(f"{SRC}/{pfx}_bidict.{d}.ndjson"):
                e = json.loads(l); hk = "sa" if d.startswith("sa") else tag; sk = tag if hk == "sa" else "sa"
                parts = []
                for s in e["senses"]:
                    k = (e["sa"], s[tag]) if hk == "sa" else (s["sa"], e[tag])
                    a = att.get(k, [])
                    stats[d + " senses"] += 1; stats[d + " attestations"] += len(a); stats[d + " count"] += s["count"]
                    if len(a) != s["count"]: stats[d + " senses count!=len"] += 1
                    meta = {kk: vv for kk, vv in s.items() if kk != "ex"}
                    body = json.dumps(meta, ensure_ascii=False)[:-1]
                    parts.append(body + ', "ex": [' + ", ".join(x for _, _, x in a) + "]}")
                head = {kk: vv for kk, vv in e.items() if kk != "senses"}
                o.write('{"%s": %s, "senses": [%s], %s\n' % (
                    hk, json.dumps(e[hk], ensure_ascii=False), ", ".join(parts),
                    json.dumps({kk: vv for kk, vv in head.items() if kk != hk}, ensure_ascii=False)[1:]))
    return pfx, dict(stats)


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with Pool(3) as p:
        res = dict(p.map(process, PAIRS))
    for f in os.listdir(SRC):
        if "_bidict." not in f and os.path.isfile(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            shutil.copyfile(f"{SRC}/{f}", f"{DST}/{f}")
    json.dump(res, open(f"{DST}/full_attestations_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
