#!/usr/bin/env python3
"""Keep only full compounds as phrase entries (2026-10-13, user decision).

The span builders emit, for every stitched compound run, all of its contiguous sub-compounds as phrase spans of their
own (a-b-c also as a-b and b-c), each also under its sandhied surface form. This removes the sub-compounds:
a phrase span is dropped when, in the same sentence, its Sanskrit token indices are a proper subset of the indices of
another phrase span that is a compound (hyphen-joined lemma without spaces). Full compounds, their surface variants,
head-only compounds, non-compound collocations and all single-word (token) spans are kept.

Applied to the spans files and, through (sentence id, indices), to every attestation of the dictionaries; senses
whose attestations are all dropped are removed, counts and `n` are recomputed (count == len(ex)), senses re-ranked
by count (ties keep their order), headwords without senses removed. Nothing else changes.

Usage: drop_subcompounds.py SRC_DIR DST_DIR      (ONLY_PAIRS=skt_tib,... to restrict)
"""
import json, os, sys, shutil
from collections import defaultdict, Counter
from multiprocessing import Pool

SRC, DST = sys.argv[1], sys.argv[2]
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]


def is_compound(row):
    lem = row.get("lemma") or ""
    return "-" in lem and " " not in lem


def drop_set(pfx):
    """(sentence_id, indices) of sub-compound phrase spans."""
    by_sent = defaultdict(list)
    with open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f:
        for l in f:
            if '"phrase"' not in l:
                continue
            r = json.loads(l)
            if r.get("type") != "phrase":
                continue
            by_sent[r["sentence_id"]].append((tuple(r["indices"]), is_compound(r)))
    drop = set()
    for sid, rows in by_sent.items():
        comps = {ix for ix, c in rows if c}
        for ix, _ in rows:
            s = set(ix)
            if any(len(c) > len(ix) and s < set(c) for c in comps):
                drop.add((sid, ix))
    return drop


def process(pt):
    pfx, tag = pt
    drop = drop_set(pfx)
    st = Counter(drop_keys=len(drop))
    with open(f"{DST}/{pfx}_spans.ndjson", "w", encoding="utf-8") as o, open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f:
        for l in f:
            if '"phrase"' in l:
                r = json.loads(l)
                if r.get("type") == "phrase" and (r["sentence_id"], tuple(r["indices"])) in drop:
                    st["spans_dropped"] += 1
                    st["spans_dropped_surface" if r.get("head_kind") == "surface" else "spans_dropped_hyphen"] += 1
                    continue
                st["phrase_kept"] += 1
            else:
                st["token_kept"] += 1
            o.write(l)
    for d in ("sa2" + tag, tag + "2sa"):
        hk = "sa" if d.startswith("sa") else tag
        with open(f"{DST}/{pfx}_bidict.{d}.ndjson", "w", encoding="utf-8") as o, \
                open(f"{SRC}/{pfx}_bidict.{d}.ndjson", encoding="utf-8") as f:
            for l in f:
                e = json.loads(l)
                st[d + " headwords_in"] += 1
                senses = []
                for s in e["senses"]:
                    ex = [x for x in s["ex"] if len(x.get("indices") or ()) < 2
                          or (x["id"], tuple(x["indices"])) not in drop]
                    st[d + " att_dropped"] += len(s["ex"]) - len(ex)
                    if not ex:
                        st[d + " senses_dropped"] += 1
                        continue
                    s["ex"] = ex; s["count"] = len(ex); senses.append(s)
                if not senses:
                    st[d + " headwords_dropped"] += 1
                    continue
                senses.sort(key=lambda s: -s["count"])          # stable: ties keep their order
                e["senses"] = senses; e["n"] = sum(s["count"] for s in senses)
                if "n_senses" in e:
                    e["n_senses"] = len(senses)
                o.write(json.dumps(e, ensure_ascii=False) + "\n")
                st[d + " headwords_out"] += 1; st[d + " senses_out"] += len(senses); st[d + " att_out"] += e["n"]
    return pfx, dict(st)


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with Pool(len(PAIRS)) as p:
        res = dict(p.map(process, PAIRS))
    json.dump(res, open(f"{DST}/drop_subcompounds_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
