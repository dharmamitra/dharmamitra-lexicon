#!/usr/bin/env python3
"""Repair a lemmatizer encoding bug (2026-10-18): the analyzer writes U+FFB1 for ḷ in kḷp (kﾱp, vikﾱp, ...) and
'rﾱ' for ṛ (krﾱtrima, krﾱṣṇa). Only keys and lemmas are repaired (sa2X "sa", "forms" keys, X2sa sense "sa", spans
"head"/"lemma", flagged tsv headword/member); Sanskrit sentences stay as in the e-texts. U+FFB1 that comes from the
e-texts themselves (mantra syllables such as huﾱā, vaﾱākāra) is left alone. Entries/senses that coincide after the
repair are merged (ex lists concatenated, counts summed), files re-sorted by n.

Usage: fix_lemma_encoding.py DIR   (in place; back up first)"""
import json, sys, os, re, collections
D = sys.argv[1]
def fix(s):
    return s.replace("kﾱp", "kḷp").replace("rﾱ", "ṛ") if s and "ﾱ" in s else s
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
st = collections.Counter()

def write_sorted(path, entries):
    entries.sort(key=lambda e: -e["n"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as o:
        for e in entries: o.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def merge_senses(senses, key):
    out = {}
    for s in senses:
        k = s[key]
        if k in out:
            out[k]["ex"] += s["ex"]; out[k]["count"] += s["count"]
        else: out[k] = s
    r = list(out.values()); r.sort(key=lambda s: -s["count"]); return r

for pfx, t in PAIRS:
    # sa -> X
    p = f"{D}/{pfx}_bidict.sa2{t}.ndjson"; ents = {}; order = []
    for l in open(p, encoding="utf-8"):
        e = json.loads(l)
        if "ﾱ" in e["sa"] or "ﾱ" in "".join(e.get("forms", {})):
            e["sa"] = fix(e["sa"]); e["forms"] = {fix(k): v for k, v in e.get("forms", {}).items()}; st[f"{t}:sa2X_fixed"] += 1
        k = e["sa"]
        if k in ents:
            a = ents[k]; st[f"{t}:sa2X_merged"] += 1
            a["senses"] = merge_senses(a["senses"] + e["senses"], t)
            a["n"] += e["n"]; a["n_senses"] = len(a["senses"])
            f = collections.Counter(a.get("forms", {})); f.update(e.get("forms", {}))
            if f: a["forms"] = dict(sorted(f.items(), key=lambda kv: -kv[1]))
        else: ents[k] = e; order.append(k)
    write_sorted(p, [ents[k] for k in order])
    # X -> sa
    p = f"{D}/{pfx}_bidict.{t}2sa.ndjson"; out = []
    for l in open(p, encoding="utf-8"):
        if "ﾱ" in l:
            e = json.loads(l); ch = False
            for s in e["senses"]:
                if "ﾱ" in s["sa"]: s["sa"] = fix(s["sa"]); ch = True
            if ch:
                n0 = len(e["senses"]); e["senses"] = merge_senses(e["senses"], "sa"); e["n_senses"] = len(e["senses"])
                st[f"{t}:X2sa_fixed"] += 1; st[f"{t}:X2sa_sense_merged"] += n0 - len(e["senses"])
            out.append(e)
        else: out.append(json.loads(l))
    write_sorted(p, out)
    # spans
    p = f"{D}/{pfx}_spans.ndjson"; tmp = p + ".tmp"
    with open(p, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as o:
        for l in f:
            if "ﾱ" in l:
                d = json.loads(l); h, lm = d.get("head"), d.get("lemma")
                d["head"] = fix(h); d["lemma"] = fix(lm)
                if (h, lm) != (d["head"], d["lemma"]): st[f"{t}:spans_fixed"] += 1
                l = json.dumps(d, ensure_ascii=False) + "\n"
            o.write(l)
    os.replace(tmp, p)
    # flagged
    p = f"{D}/flagged_{pfx}.tsv"
    if os.path.exists(p):
        L = open(p, encoding="utf-8").read().split("\n")
        L = [("\t".join(fix(c) if i in (0, 2) else c for i, c in enumerate(x.split("\t")))) for x in L]
        open(p, "w", encoding="utf-8").write("\n".join(L))
print(json.dumps(st, indent=1))
