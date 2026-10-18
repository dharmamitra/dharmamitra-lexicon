#!/usr/bin/env python3
"""Where do the textually attested Negi equivalents that MITRA-dict lacks come from, and is the cited passage in
our corpus at all?

For every missing (headword, Negi equivalent) pair of the attested-only evaluation (NEGI_ATTESTED=1), take the
citations in the Negi entry of that Tibetan headword whose Sanskrit quotation contains the equivalent, and check
whether the quoted Sanskrit occurs (any 4-word sequence, normalised) in the Sanskrit side of our aligned corpus,
of MITRA-parallel v2, and of v1. Tallied per source siglum. The same is done for a sample of equivalents we DO
have, as a baseline.

Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/negi-sources.json and printed tables.
"""
import os, collections, glob, gzip, json, os, re, sys
os.environ["NEGI_ATTESTED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import negi
from tibetan import tib_key
import novel

FINAL = __import__("os").environ.get("CORPUS_SA_BO", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-bo/hybrid/final.ndjson")
V2 = os.path.expanduser("~/data/mitra-parallel/v2/sa-bo_matches.ndjson.gz")
V1 = os.path.expanduser("~/data/mitra-parallel/v1/mitra-parallel/tsv")
K = 4
DEVQ = re.compile(r"([ऀ-ॣ०-ॿ][ऀ-ॿ\s'ऽ।॥,?!-]{8,}?)\s*((?:[ऀ-ॿ]{1,7}\.\s?){1,4})\s?\d")


def nsa(s):
    s = (s or "").lower().replace("ṁ", "ṃ")
    return re.sub(r"[^a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ]+", " ", s).split()


def shingles(toks):
    return {hash(" ".join(toks[i:i + K])) for i in range(len(toks) - K + 1)}


def corpus_shingles():
    out = {}
    s = set()
    for line in open(FINAL, encoding="utf-8"):
        i = line.find('"src_original": "')
        if i < 0:
            continue
        j = line.find('", "', i + 17)
        s |= shingles(nsa(json.loads('"' + line[i + 17:j] + '"')))
    out["ours"] = s
    s = set()
    with gzip.open(V2, "rt") as f:
        for l in f:
            s |= shingles(nsa(json.loads(l)["root_string"]))
    out["v2"] = s
    s = set()
    for fn in os.listdir(V1):
        with open(os.path.join(V1, fn), encoding="utf-8") as f:
            next(f, None)
            for l in f:
                t = l.split("\t", 3)
                if len(t) >= 4 and re.match(r"^[KT]\d+D\d", t[2]):
                    s |= shingles(nsa(t[1]))
    out["v1"] = s
    return out


def citations(raw_lines):
    """(siglum, quoted Sanskrit in IAST) for every Devanagari quotation followed by a source siglum."""
    out = []
    for val in raw_lines:
        for m in DEVQ.finditer(val):
            q, sg = m.group(1), re.sub(r"\s", "", m.group(2))
            if negi.is_lexical_siglum(sg):
                continue
            ia = negi.dev2iast(q).lower()
            out.append((sg, ia))
    return out


def main():
    raw = collections.defaultdict(list)
    for line in open(negi.NEGI, encoding="utf-8"):
        if "\t" in line:
            hw, val = line.rstrip("\n").split("\t", 1)
            raw[tib_key(negi.norm_wylie_head(hw))].append(val)
    pairs, _ = negi.load_negi(tib_key)
    tk_to_heads = collections.defaultdict(set)      # xref: also look in the entries the headword points to
    for hw, tk, k, src in pairs:
        tk_to_heads[tk].add(tib_key(hw))
    cap = novel.run("sa-bo")
    missing, present = [], []
    for r in cap["recs"]:
        if r.get("ref_missing") is None:
            continue
        miss = set(r["ref_missing"])
        for x in r["refs_show"] if r["nrefs"] <= 12 else []:
            pass
        allrefs = cap["index"][C.sa_key(r["sa"])]
        for x in allrefs:
            (missing if x in miss else present).append((r["sa"], r["n"], x))
    print(f"attested Negi equivalents for headwords in E: missing {len(missing)}, matched {len(present)}")
    sh = corpus_shingles()
    print({k: len(v) for k, v in sh.items()})

    def evidence(sa, bo):
        stem = C.sa_key(sa)[:max(3, len(C.sa_key(sa)) - 2)]
        cites = [c for c in citations(raw.get(bo, [])) if stem in c[1].replace(" ", "")]
        return cites

    res = {}
    for label, pool in (("missing", missing), ("matched", present)):
        by_sg = collections.defaultdict(lambda: collections.Counter())
        tot = collections.Counter()
        for sa, n, bo in pool:
            cites = evidence(sa, bo)
            if not cites:
                tot["no_quoted_citation_found"] += 1
                continue
            tot["with_citation"] += 1
            found = {k: False for k in sh}
            for sg, ia in cites:
                q = shingles(nsa(ia))
                c = by_sg[sg]
                c["cites"] += 1
                for k, s in sh.items():
                    hit = bool(q & s)
                    c[k] += hit
                    found[k] |= hit
            for k, v in found.items():
                tot[f"any_cited_passage_in_{k}"] += v
            tot["in_none"] += not any(found.values())
        res[label] = dict(totals=dict(tot), by_siglum={k: dict(v) for k, v in by_sg.items()})
        n = tot["with_citation"]
        print(f"\n== {label}: {len(pool)} pairs; with a quoted citation containing the word: {n} "
              f"(none found: {tot['no_quoted_citation_found']})")
        for k in ("ours", "v2", "v1"):
            print(f"   cited passage found in {k:4}: {tot[f'any_cited_passage_in_{k}']:6d} ({100*tot[f'any_cited_passage_in_{k}']/max(n,1):5.1f}%)")
        print(f"   in none of them           : {tot['in_none']:6d} ({100*tot['in_none']/max(n,1):5.1f}%)")
        if label == "missing":
            print("   by siglum (citations | % found in ours | v2 | v1):")
            for sg, c in sorted(by_sg.items(), key=lambda kv: -kv[1]["cites"])[:40]:
                print(f"     {sg:12} {c['cites']:6d} | {100*c['ours']/c['cites']:5.1f} | {100*c['v2']/c['cites']:5.1f} | {100*c['v1']/c['cites']:5.1f}")
    json.dump(res, open("/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/negi-sources.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
