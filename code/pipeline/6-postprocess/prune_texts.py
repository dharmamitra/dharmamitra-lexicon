#!/usr/bin/env python3
"""Remove every attestation from a list of Sanskrit texts (2026-10-17: texts whose Sanskrit is wholly or partly
reconstructed from a Tibetan or Chinese translation; list in reconstructed_texts.tsv).

Per pair: sa->X and X->sa entries lose the attestations whose sentence id belongs to an excluded text; sense counts
are recomputed (relaxed counts scaled by the share of attestations kept), empty senses and entries are dropped,
senses are re-ranked by count (stable) and entries re-sorted by n; `forms` counts are scaled the same way.
Spans files lose the rows of excluded sentences. Other files are copied.

Text of an attestation: bo/zh ids "SA_T06_sthmavt:12_BO_..." -> "SA_T06_sthmavt"; en ids "train/cc-by-sa/T06sthmavt_u.tsv:5"
-> "T06sthmavt_u" (matched against the list's en column).

Usage: prune_texts.py SRC_DIR DST_DIR LIST.tsv      (ONLY_PAIRS=skt_zh,... to restrict)
Corpus files: prune_texts.py --corpus IN.ndjson OUT.ndjson LIST.tsv   (rows keyed by "id")
"""
import json, os, re, sys, shutil, collections, unicodedata
from multiprocessing import Pool

PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]
nfc = lambda s: unicodedata.normalize("NFC", s)


def load_list(path):
    sa, en = set(), set()
    for l in open(path, encoding="utf-8"):
        if l.startswith("#") or not l.strip():
            continue
        f = l.rstrip("\n").split("\t")
        if f[0] != "-":
            sa.add(nfc(f[0]))
        if len(f) > 1 and f[1] not in ("", "-"):
            en.update(nfc(x) for x in f[1].split(","))
    return sa, en


def text_of(i):
    if i.startswith("SA_"):
        return nfc(i.split(":")[0])
    return nfc(os.path.basename(i.split(":")[0]).rsplit(".tsv", 1)[0])


def excluded(i):
    t = text_of(i)
    return t in SA or t in EN


def prune_entries(src, dst, gloss_key, st, tag):
    out = []
    for l in open(src, encoding="utf-8"):
        e = json.loads(l); st[f"{tag}_in"] += 1
        n0 = e.get("n", 0); senses = []
        for s in e["senses"]:
            ex = [x for x in s["ex"] if not excluded(x["id"])]
            st[f"{tag}_att_removed"] += len(s["ex"]) - len(ex)
            if not ex:
                continue
            if "relaxed" in s and s["ex"]:
                s["relaxed"] = round(s["relaxed"] * len(ex) / len(s["ex"]))
            s["ex"] = ex; s["count"] = len(ex); senses.append(s)
        if not senses:
            st[f"{tag}_entries_removed"] += 1
            continue
        senses.sort(key=lambda s: -s["count"])
        e["senses"] = senses; e["n"] = sum(s["count"] for s in senses); e["n_senses"] = len(senses)
        if "forms" in e and n0 and e["n"] != n0:
            fs = {k: round(v * e["n"] / n0) for k, v in e["forms"].items()}
            e["forms"] = {k: v for k, v in sorted(fs.items(), key=lambda kv: -kv[1]) if v > 0}
        out.append(e)
    out.sort(key=lambda e: -e["n"])
    with open(dst, "w", encoding="utf-8") as o:
        for e in out:
            o.write(json.dumps(e, ensure_ascii=False) + "\n")
    st[f"{tag}_out"] = len(out); st[f"{tag}_att_out"] = sum(e["n"] for e in out)


def process_pair(pfx, tag):
    st = collections.Counter()
    prune_entries(f"{SRC}/{pfx}_bidict.sa2{tag}.ndjson", f"{DST}/{pfx}_bidict.sa2{tag}.ndjson", tag, st, f"sa2{tag}")
    prune_entries(f"{SRC}/{pfx}_bidict.{tag}2sa.ndjson", f"{DST}/{pfx}_bidict.{tag}2sa.ndjson", "sa", st, f"{tag}2sa")
    with open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f, open(f"{DST}/{pfx}_spans.ndjson", "w", encoding="utf-8") as o:
        for l in f:
            m = re.search(r'"sentence_id": "([^"]+)"', l)
            if m and excluded(m.group(1)):
                st["spans_removed"] += 1; continue
            o.write(l)
    return pfx, dict(st)


def prune_corpus(src, dst):
    k = r = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
        for l in f:
            m = re.search(r'"id": "([^"]+)"', l)
            if m and excluded(m.group(1)):
                r += 1; continue
            o.write(l); k += 1
    print(json.dumps({"src": src, "kept": k, "removed": r}))


if __name__ == "__main__":
    if sys.argv[1] == "--corpus":
        SA, EN = load_list(sys.argv[4]); prune_corpus(sys.argv[2], sys.argv[3]); sys.exit()
    SRC, DST, LST = sys.argv[1:4]
    SA, EN = load_list(LST)
    os.makedirs(DST, exist_ok=True)
    with Pool(len(PAIRS)) as p:
        res = dict(p.starmap(process_pair, PAIRS))
    done = {f"{pfx}_{s}" for pfx, t in PAIRS for s in ("spans.ndjson", f"bidict.sa2{t}.ndjson", f"bidict.{t}2sa.ndjson")}
    for f in os.listdir(SRC):
        if f not in done and os.path.isfile(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            shutil.copy2(f"{SRC}/{f}", f"{DST}/{f}")
    if os.path.isdir(f"{SRC}/pre-mdc-20260919") and not os.path.exists(f"{DST}/pre-mdc-20260919"):
        shutil.copytree(f"{SRC}/pre-mdc-20260919", f"{DST}/pre-mdc-20260919", symlinks=True)
    shutil.copy2(LST, f"{DST}/excluded_texts.tsv")
    json.dump(res, open(f"{DST}/prune_texts_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
