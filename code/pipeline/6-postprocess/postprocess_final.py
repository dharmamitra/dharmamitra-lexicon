#!/usr/bin/env python3
"""Final post-processing of the released dictionaries (2026-10-10).

1. Anusvāra normalisation: every Sanskrit key (headwords of sa2X, Sanskrit senses of X2sa,
   the `sa` field of the Hirakawa layer, `head`/`lemma` of the spans files) has ṁ -> ṃ
   (Ṁ -> Ṃ). Entries and senses that become identical are merged: counts summed, examples
   concatenated (capped at MAX_EX), senses re-ranked by count. Attested text (src, target,
   offsets) is left verbatim.
2. E-text error filter: a Sanskrit headword is dropped when one of its members (split at
   '-', ' ', '_'; members of >= 3 characters) occurs at most RARE times as lemma or form in
   the ByT5-analysed dharmanexus-sanskrit corpus AND lies one edit away from a lemma or form
   that occurs at least FREQ times there (likely typing error of the e-text), unless the member
   is a headword of Monier-Williams or Apte (ref_allowlist_mw_apte.txt, keys normalised as
   in build/eval/common.py:sa_key; written by eval_sa_en.load_reference). Sandhied
   surface keys that are unknown to the corpus and attested < 5 times cannot be checked
   this way and are kept. Dropped headwords are also removed as senses from X2sa and are
   written to flagged_<pair>.tsv with the suspected intended form.
   The spans files (raw alignments) are normalised but not filtered.

Usage: postprocess_final.py SRC_DIR DST_DIR     (counts: dnx_counts.pkl from dnx_counts.py)
"""
import json, os, pickle, re, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool

RARE, FREQ, MAX_EX = 5, 300, None   # MAX_EX None: no cap on examples (2026-10-11)
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):                 # e.g. ONLY_PAIRS=skt_tib: rebuild one pair only
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]


def norm(s):
    return unicodedata.normalize("NFC", s).replace("ṁ", "ṃ").replace("Ṁ", "Ṃ")


# ---------------------------------------------------------------- corpus counts / filter
def load_counts():
    tot, _, _ = pickle.load(open(os.path.join(HERE, "dnx_counts.pkl"), "rb"))
    L, F = defaultdict(int), defaultdict(int)
    for src, dst in ((tot["lem"], L), (tot["form"], F)):
        for w, c in src.items():
            dst[norm(w)] += c
    return L, F


L, F = load_counts()
cnt = lambda w: max(L.get(w, 0), F.get(w, 0))
FREQW = {}
for C in (L, F):
    for w, c in C.items():
        if c >= FREQ and len(w) >= 3:
            FREQW[w] = max(FREQW.get(w, 0), c)
dels = lambda w: {w} | {w[:i] + w[i + 1:] for i in range(len(w))}
IDX = defaultdict(set)
for w in FREQW:
    for d in dels(w):
        IDX[d].add(w)


_ANU_RE = re.compile(r"ṃ(?=([kgcjṭḍtdpb]))")
_ANU = [("kg", "ṅ"), ("cj", "ñ"), ("ṭḍ", "ṇ"), ("td", "n"), ("pb", "m")]


def ref_key(s):          # = build/eval/common.py:sa_key for IAST without punctuation
    s = norm(s).lower().replace("ń", "ṅ")
    return _ANU_RE.sub(lambda m: next(n for c, n in _ANU if m.group(1) in c), s)


ALLOW = set(open(os.path.join(HERE, "ref_allowlist_mw_apte.txt")).read().split("\n")) - {""}


def neighbour(w):
    c = set()
    for d in dels(w):
        c |= IDX.get(d, set())
    c.discard(w)
    return max(c, key=lambda x: (FREQW[x], x)) if c else None


def flag(hw, n):
    """(member, suspected form) if hw looks like an e-text error, else None."""
    multi = any(ch in hw for ch in " -_")
    if not multi and L.get(hw, 0) == 0 and F.get(hw, 0) == 0 and n < 5:
        return None                                   # surface key: not checkable
    for m in (x for x in re.split(r"[- _]+", hw) if len(x) >= 3):
        if cnt(m) <= RARE and ref_key(m) not in ALLOW:
            nb = neighbour(m)
            if nb:
                return m, nb
    return None


# ---------------------------------------------------------------- merging
def merge_senses(senses, key):
    out, pos = [], {}
    for s in senses:
        k = s[key]
        if k not in pos:
            pos[k] = len(out); out.append(dict(s)); continue
        t = out[pos[k]]
        t["count"] += s["count"]
        t["ex"] = (t.get("ex", []) + s.get("ex", []))[:MAX_EX]
        if "relaxed" in t and "relaxed" not in s:     # relaxed only if every merged sense was
            del t["relaxed"]
    out.sort(key=lambda s: -s["count"])               # stable: ties keep first-seen order
    return out


def finish(e):
    e["n"] = sum(s["count"] for s in e["senses"])
    e["n_senses"] = len(e["senses"])
    return e


def process_pair(pfx, tag):
    src, dst = SRC, DST
    # forward: Sanskrit -> X
    entries, order = {}, []
    for l in open(f"{src}/{pfx}_bidict.sa2{tag}.ndjson"):
        e = json.loads(l); k = norm(e["sa"])
        if k in entries:
            entries[k]["senses"] += e["senses"]
        else:
            e["sa"] = k; entries[k] = e; order.append(k)
    merged_hw = len(order)
    dropped, keep = {}, []
    for k in order:
        e = entries[k]
        e["senses"] = merge_senses(e["senses"], tag); finish(e)
        f = flag(k, e["n"])
        if f:
            dropped[k] = (e["n"], *f)
        else:
            keep.append(k)
    with open(f"{dst}/{pfx}_bidict.sa2{tag}.ndjson", "w") as o:
        for k in keep:
            o.write(json.dumps(entries[k], ensure_ascii=False) + "\n")
    with open(f"{dst}/flagged_{pfx}.tsv", "w") as o:
        o.write("headword\tattestations\tmember\tsuspected_form\n")
        for k, (n, m, nb) in sorted(dropped.items(), key=lambda x: (-x[1][0], x[0])):
            o.write(f"{k}\t{n}\t{m}\t{nb}\n")
    # reverse: X -> Sanskrit
    rev_in = rev_out = 0
    with open(f"{dst}/{pfx}_bidict.{tag}2sa.ndjson", "w") as o:
        for l in open(f"{src}/{pfx}_bidict.{tag}2sa.ndjson"):
            e = json.loads(l); rev_in += 1
            for s in e["senses"]:
                s["sa"] = norm(s["sa"])
            e["senses"] = merge_senses([s for s in e["senses"] if s["sa"] not in dropped], "sa")
            if e["senses"]:
                o.write(json.dumps(finish(e), ensure_ascii=False) + "\n"); rev_out += 1
    return pfx, dict(sa_in=sum(1 for _ in open(f"{src}/{pfx}_bidict.sa2{tag}.ndjson")),
                     sa_after_norm=merged_hw, sa_dropped=len(dropped), sa_out=len(keep),
                     dropped_att=sum(v[0] for v in dropped.values()), rev_in=rev_in, rev_out=rev_out)


def norm_span_line(l):
    if "ṁ" not in l and "Ṁ" not in l:
        return l
    x = json.loads(l)
    for k in ("head", "lemma"):
        if isinstance(x.get(k), str):
            x[k] = norm(x[k])
    return json.dumps(x, ensure_ascii=False) + "\n"


def process_spans(pfx):
    with open(f"{SRC}/{pfx}_spans.ndjson") as i, open(f"{DST}/{pfx}_spans.ndjson", "w") as o, Pool(32) as p:
        for l in p.imap(norm_span_line, i, chunksize=20000):
            o.write(l)
    return pfx


def process_hirakawa():
    seen = set()
    with open(f"{DST}/skt_zh_hirakawa_layer.ndjson", "w") as o:
        for l in open(f"{SRC}/skt_zh_hirakawa_layer.ndjson"):
            x = json.loads(l); x["sa"] = norm(x["sa"])
            if (x["zh"], x["sa"]) in seen:
                continue
            seen.add((x["zh"], x["sa"])); o.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    SRC, DST = sys.argv[1], sys.argv[2]
    os.makedirs(DST, exist_ok=True)
    stats = {}
    for pfx, tag in PAIRS:
        k, v = process_pair(pfx, tag); stats[k] = v; print(k, v, flush=True)
    if any(p == "skt_zh" for p, _ in PAIRS):
        process_hirakawa()
    for pfx, _ in PAIRS:
        print("spans", process_spans(pfx), flush=True)
    json.dump(stats, open(f"{DST}/postprocess_stats.json", "w"), indent=1)
