#!/usr/bin/env python3
"""The other side of the comparison with the reference dictionaries: what the mined
dictionaries contain that the references do not.

Reuses the reference parsing, normalisation and matching of eval_sa_{bo,zh,en}.py unchanged
(the modules are run with their per-sense match decisions recorded) and reports, per pair:

  headwords  mined single-word Sanskrit headwords with / without an entry in the reference,
             by number of attestations of the headword;
  senses     for headwords the reference has (E): mined senses confirmed / not confirmed by
             the reference, by number of attestations of the sense. "Unconfirmed" = no match
             even under the most lenient criterion (relaxed containment for Tibetan and
             Chinese; MW u Apte prose criterion for English, untranslated loans set aside).

Outputs: /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/novel.json, /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/novel-samples.md (CPU only, ~2 minutes).
"""
import collections, importlib, json, os, random, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018"
LENIENT = {"sa-bo": "relaxed", "sa-zh": "relaxed", "sa-en": "mw_apte"}
STRICT = {"sa-bo": "strict", "sa-zh": "strict", "sa-en": "mw_apte"}
MODULE = {"sa-bo": "eval_sa_bo", "sa-zh": "eval_sa_zh", "sa-en": "eval_sa_en"}
FIELD = {"sa-bo": "bo", "sa-zh": "zh", "sa-en": "en"}
# The paper describes the Sanskrit-English dictionary built from Mitrasamgraha alone (584,696 headwords);
# final-output/skt_en_bidict.sa2en.ndjson was rebuilt with further data on 2026-09-19.
MINED_OVERRIDE = {} if os.environ.get("EVAL_MINED_DIR") else {"sa-en": "pre-mdc-20260919/skt_en_bidict.sa2en.ndjson"}
DCS = os.path.expanduser("~/data/sanskrit/sanskrit-hellwig/dcs/data/conllu")
POS_CLASS = {"VERB": "verb", "NOUN": "noun", "ADJ": "adjective"}
BANDS = [b[0] for b in C.BANDS]


def _count_file(path):
    c = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line[:1] == "#":
                continue
            t = line.split("\t")
            if len(t) >= 10 and "-" not in t[0]:
                c[(t[2], t[3])] += 1
    return c


def dcs_lemmas():
    """DCS lemma inventory: {sa_key(lemma): (corpus frequency, POS class)}; frequency 0 = in the
    DCS lexicon but not attested in the annotated corpus."""
    import glob
    from multiprocessing import Pool
    files = glob.glob(os.path.join(DCS, "files", "**", "*.conllu"), recursive=True)
    tot = collections.Counter()
    with Pool(C.NPROC) as pool:
        for c in pool.imap_unordered(_count_file, files, chunksize=200):
            tot.update(c)
    freq, pos = collections.Counter(), collections.defaultdict(collections.Counter)
    for (lemma, upos), n in tot.items():
        k = C.sa_key(lemma)
        if k and C.is_single_word(lemma):
            freq[k] += n; pos[k][upos] += n
    out = {k: (n, POS_CLASS.get(pos[k].most_common(1)[0][0], "other")) for k, n in freq.items()}
    with open(os.path.join(DCS, "lookup", "dictionary.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) < 3 or not C.is_single_word(t[1]):
                continue
            k = C.sa_key(t[1])
            if k and k not in out:
                g = t[2]
                out[k] = (0, "verb" if (g[:1].isdigit() or g.startswith("Denom")) else
                          "adjective" if g == "adj" else "noun" if g in ("m", "f", "n", "mn", "mf", "fn", "mfn") else "other")
    return out, sum(tot.values()), len(files)


def run(direction):
    """-> (light, recs, index); recs carry rec['senses'] = [(gloss, count, strict, lenient, loan)]."""
    if direction in MINED_OVERRIDE:
        os.environ["EVAL_MINED"] = MINED_OVERRIDE[direction]
    else:
        os.environ.pop("EVAL_MINED", None)
    mod = importlib.import_module(MODULE[direction])
    orig_m, orig_eval, orig_run = mod.metrics_from_matchsets, mod.evaluate, C.run_eval

    def metrics(counts, matchsets, nref):
        r = orig_m(counts, matchsets, nref)
        r["_hits"] = [bool(m) for m in matchsets]
        r["_recalled"] = set().union(*matchsets) if matchsets else set()
        return r

    def evaluate(sa, key, n, senses, refs, ex0):
        rec = orig_eval(sa, key, n, senses, refs, ex0)
        s, l = rec["m"].get(STRICT[direction]), rec["m"].get(LENIENT[direction])
        if s is None or l is None:
            rec["senses"] = None
        else:
            loan = [mod.is_loan(g, sa) for g, _ in senses] if direction == "sa-en" else [False] * len(senses)
            rec["senses"] = [(g, c, hs, hl, lo) for (g, c), hs, hl, lo in zip(senses, s["_hits"], l["_hits"], loan)]
        rec["ex"] = C.highlight(ex0) if ex0 else ""
        # the reverse side: reference equivalents that no mined sense matches (lists of equivalents only)
        rec["ref_missing"] = None
        if direction != "sa-en" and l is not None:
            R = list(refs)
            rec["ref_missing"] = [R[i] for i in range(len(R)) if i not in l["_recalled"]]
            rec["ref_total"] = len(R)
        return rec

    captured = {}

    def run_eval(mined_file):
        light, recs = orig_run(mined_file)
        if "recs" not in captured:           # the second call is the shuffled-reference control
            captured.update(light=light, recs=recs, index=C.CFG["index"], mined=mined_file)
        return light, recs

    out_dir, C.OUT_DIR = C.OUT_DIR, tempfile.mkdtemp(prefix="novel-eval-")   # keep the published outputs untouched
    mod.metrics_from_matchsets, mod.evaluate, C.run_eval = metrics, evaluate, run_eval
    try:
        stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            mod.main()
        finally:
            sys.stdout = stdout
    finally:
        mod.metrics_from_matchsets, mod.evaluate, C.run_eval, C.OUT_DIR = orig_m, orig_eval, orig_run, out_dir
    return captured


def outside_reference(direction, cap, min_n=5):
    """single-word headwords without a reference entry: (sa, n, top senses)."""
    out = []
    with open(os.path.join(C.MINED_DIR, cap["mined"]), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["n"] < min_n or not C.is_single_word(d["sa"]) or C.sa_key(d["sa"]) in cap["index"]:
                continue
            out.append((d["sa"], d["n"], [(s[FIELD[direction]], s["count"]) for s in d["senses"][:3]]))
    return out


def analyse(direction, dcs):
    cap = run(direction)
    light, recs = cap["light"], [r for r in cap["recs"] if r.get("senses") is not None]
    index = cap["index"]
    res = {"headwords": {}, "senses": {}, "dcs_coverage": {}, "mined_vs_dcs": {}, "mined_file": cap["mined"]}
    # (a) how much of the DCS lemma inventory does the reference cover at all?
    cov = collections.defaultdict(collections.Counter)
    for k, (f, pc) in dcs.items():
        e = k in index
        for key in ("all", "band:" + (C.band_of(f) if f else "0"), "pos:" + pc):
            a = cov[key]
            a["lemmas"] += 1; a["in_reference"] += e; a["tokens"] += f; a["tokens_in_reference"] += f * e
        if f >= 20:
            a = cov["pos>=20:" + pc]
            a["lemmas"] += 1; a["in_reference"] += e
    res["dcs_coverage"] = {k: dict(v) for k, v in sorted(cov.items())}
    # (b) mined headwords against both inventories
    for b in BANDS:
        a = collections.Counter()
        for k, n, e in light:
            if C.band_of(n) != b:
                continue
            d = k in dcs
            a["mined"] += 1; a["in_dcs"] += d; a["in_reference"] += e
            a["dcs_and_reference"] += (d and e); a["dcs_not_reference"] += (d and not e)
            a["reference_not_dcs"] += (e and not d); a["neither"] += (not d and not e)
        res["mined_vs_dcs"][b] = dict(a)
    for b in BANDS:
        ns = [(n, e) for k, n, e in light if C.band_of(n) == b]
        res["headwords"][b] = dict(mined=len(ns), in_reference=sum(e for n, e in ns),
                                   not_in_reference=sum(not e for n, e in ns),
                                   attestations_not_in_reference=sum(n for n, e in ns if not e))
    sb = collections.defaultdict(lambda: collections.Counter())
    unconf = []
    for r in recs:
        for rank, (g, c, hs, hl, lo) in enumerate(r["senses"], 1):
            a = sb[C.band_of(c)]
            a["senses"] += 1; a["strict"] += hs; a["lenient"] += hl; a["loan"] += (lo and not hl)
            a["unconfirmed"] += (not hl and not lo); a["att_unconfirmed"] += c * (not hl and not lo)
            a["att"] += c
            if not hl and not lo and c >= 5:
                unconf.append((c, r["sa"], r["n"], g, rank, r["refs_show"][:8], r["ex"] if rank == 1 else ""))
    res["senses"] = {b: dict(sb[b]) for b in BANDS if b in sb}
    res["E_headwords"] = len(recs)
    missing = [(r["sa"], r["n"], x, r["senses"][:4], r["ref_total"]) for r in recs if r.get("ref_missing")
               for x in r["ref_missing"] if r["n"] >= 20]
    res["reference_equivalents_missing_n>=20"] = len(missing)
    res["E_headwords_with_unconfirmed_sense_ge20"] = len({u[1] for u in unconf if u[0] >= 20})
    outside = [o + (C.sa_key(o[0]) in dcs,) for o in outside_reference(direction, cap)]
    return res, unconf, outside, missing


def main():
    rnd = random.Random(C.SEED)
    allres, L = {}, ["# What the mined dictionaries have that the reference dictionaries do not", "",
                     "Generated by build/eval/novel.py. Samples are random (seed 42) unless marked 'top'.", ""]
    dcs, dcs_tokens, dcs_files = dcs_lemmas()
    print(f"DCS: {dcs_files} files, {dcs_tokens:,} tokens, {len(dcs):,} single-word lemmas "
          f"({sum(1 for f, _ in dcs.values() if f):,} attested in the corpus)")
    allres["_dcs"] = dict(files=dcs_files, tokens=dcs_tokens, lemmas=len(dcs))
    for direction in ("sa-bo", "sa-zh", "sa-en"):
        res, unconf, outside, missing = analyse(direction, dcs)
        allres[direction] = res
        print(f"\n===== {direction}   ({res['mined_file']})")
        print("DCS lemmas with an entry in the reference, by DCS corpus frequency:   lemmas | in reference (share)")
        for k, a in res["dcs_coverage"].items():
            tok = f"   tokens covered {100*a['tokens_in_reference']/a['tokens']:5.1f}%" if a.get("tokens") else ""
            print(f"  {k:22} {a['lemmas']:7d} {a['in_reference']:7d}  ({100*a['in_reference']/a['lemmas']:5.1f}%){tok}")
        print("mined headwords by attestations:   mined | DCS lemma | in ref | DCS & ref | DCS, not ref | ref, not DCS | neither")
        for b in BANDS:
            a = collections.Counter(res["mined_vs_dcs"][b])
            print(f"  {b:8} {a['mined']:7d} {a['in_dcs']:7d} {a['in_reference']:7d} {a['dcs_and_reference']:7d} "
                  f"{a['dcs_not_reference']:7d} {a['reference_not_dcs']:7d} {a['neither']:7d}   "
                  f"| of DCS-known mined headwords in ref: {100*a['dcs_and_reference']/max(a['in_dcs'],1):5.1f}%")
        print("headwords by attestations of the headword:   mined | in reference | not in reference (share)")
        for b in BANDS:
            h = res["headwords"][b]
            print(f"  {b:8} {h['mined']:7d} {h['in_reference']:7d} {h['not_in_reference']:7d}  "
                  f"({100*h['not_in_reference']/max(h['mined'],1):5.1f}%)")
        print("senses of E headwords by attestations of the sense:   senses | strict % | lenient % | unconfirmed | loans")
        for b, a in res["senses"].items():
            print(f"  {b:8} {a['senses']:7d} {100*a['strict']/a['senses']:6.1f} {100*a['lenient']/a['senses']:6.1f} "
                  f"{a['unconfirmed']:7d} {a['loan']:6d}")
        print("E headwords with at least one unconfirmed sense of >=20 attestations:",
              res["E_headwords_with_unconfirmed_sense_ge20"], "of", res["E_headwords"])
        L += [f"## {direction}", ""]
        unconf.sort(key=lambda u: -u[0])
        for title, pool in (("unconfirmed senses, top 40 by attestations", unconf[:40]),
                            ("unconfirmed senses with >=100 attestations, 40 random", rnd.sample([u for u in unconf if u[0] >= 100], min(40, sum(u[0] >= 100 for u in unconf)))),
                            ("unconfirmed senses with 20-99 attestations, 40 random", rnd.sample([u for u in unconf if 20 <= u[0] < 100], min(40, sum(20 <= u[0] < 100 for u in unconf))))):
            L += [f"### {title}", ""]
            for c, sa, n, g, rank, refs, ex in pool:
                L.append(f"- **{sa}** (n={n}) : `{g}` ({c}, rank {rank}) | ref: " + "; ".join(f"`{x}`" for x in refs)
                         + (f" | ex: {ex[:160]}" if ex else ""))
            L.append("")
        for title, lo, hi in (("reference equivalents that no mined sense matches, headword n>=1000, 40 random", 1000, 10**9),
                              ("reference equivalents that no mined sense matches, headword 100<=n<1000, 40 random", 100, 1000),
                              ("reference equivalents that no mined sense matches, headword 20<=n<100, 40 random", 20, 100)):
            pool = [m for m in missing if lo <= m[1] < hi]
            if not pool:
                continue
            L += [f"### {title} (of {len(pool)})", ""]
            for sa, n, x, top, nref in rnd.sample(pool, min(40, len(pool))):
                L.append(f"- **{sa}** (n={n}, {nref} reference equivalents) : reference `{x}` | mined: "
                         + "; ".join(f"`{g}` ({c})" for g, c, *_ in top))
            L.append("")
        outside.sort(key=lambda o: -o[1])
        for title, pool in (("headwords without a reference entry, top 40 by attestations", outside[:40]),
                            ("headwords without a reference entry, n>=100, 40 random", rnd.sample([o for o in outside if o[1] >= 100], min(40, sum(o[1] >= 100 for o in outside)))),
                            ("headwords without a reference entry, 20<=n<100, 40 random", rnd.sample([o for o in outside if 20 <= o[1] < 100], min(40, sum(20 <= o[1] < 100 for o in outside))))):
            L += [f"### {title}", ""]
            for sa, n, top, in_dcs in pool:
                L.append(f"- **{sa}** (n={n}, {'DCS lemma' if in_dcs else 'not in DCS'}) : " + "; ".join(f"`{g}` ({c})" for g, c in top))
            L.append("")
    json.dump(allres, open(os.path.join(OUT, "novel.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(OUT, "novel-samples.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n->", os.path.join(OUT, "novel.json"), os.path.join(OUT, "novel-samples.md"))


if __name__ == "__main__":
    main()
