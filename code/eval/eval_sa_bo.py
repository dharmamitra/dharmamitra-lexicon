#!/usr/bin/env python3
"""sa->bo: mined skt_tib_bidict.sa2bo.ndjson vs. Negi's Tibetan-Sanskrit Dictionary (inverted).

Run without arguments. Writes  notes/eval/parts/sa-bo.json  and  notes/eval/mismatches-sa-bo.md
CPU only.
"""
import os, sys, json, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key, stem_candidates, metrics_from_matchsets
from tibetan import tib_key, pad, has_content
import negi

MINED = os.environ.get("EVAL_MINED", "skt_tib_bidict.sa2bo.ndjson")
OUT_TAG = os.environ.get("EVAL_TAG", "")
VARIANTS = ["strict", "relaxed", "relaxed_mined_contains_ref", "strict_merged"]


def evaluate(sa, key, n, senses, refs, ex0):
    """refs: tuple of normalised Tibetan reference keys."""
    R = list(refs)
    Rpad = [pad(r) for r in R]
    Rcontent = [has_content(r) for r in R]
    ridx = {r: i for i, r in enumerate(R)}
    sk = [tib_key(s) for s, _ in senses]
    counts = [c for _, c in senses]
    cache = {}

    def msets(k):
        if k in cache:
            return cache[k]
        strict, sup, sub = set(), set(), set()
        if k:
            if k in ridx:
                strict.add(ridx[k])
            kp = pad(k)
            kc = has_content(k)
            for i, rp in enumerate(Rpad):
                if i in strict:
                    continue
                if Rcontent[i] and rp in kp:          # mined contains reference
                    sup.add(i)
                elif kc and kp in rp:                 # mined is contained in reference
                    sub.add(i)
        cache[k] = (strict, sup, sub)
        return cache[k]

    ms = [msets(k) for k in sk]
    m = {
        "strict": metrics_from_matchsets(counts, [a for a, b, c in ms], len(R)),
        "relaxed": metrics_from_matchsets(counts, [a | b | c for a, b, c in ms], len(R)),
        "relaxed_mined_contains_ref": metrics_from_matchsets(counts, [a | b for a, b, c in ms], len(R)),
    }
    # merged: collapse senses with identical normalised key, re-rank by summed count (stable)
    agg = collections.OrderedDict()
    for k, c in zip(sk, counts):
        agg[k] = agg.get(k, 0) + c
    merged = sorted(agg.items(), key=lambda kv: -kv[1])
    m["strict_merged"] = metrics_from_matchsets([c for _, c in merged], [msets(k)[0] for k, _ in merged], len(R))

    rec = {"sa": sa, "n": n, "m": m, "top": [(s, c) for s, c in senses[:3]], "refs_show": sorted(R)[:12],
           "nrefs": len(R)}
    if n >= 20 and not m["strict"]["p1"]:
        a, b, c = ms[0]
        cats = []
        if b:
            cats.append("relaxed match: mined top-1 CONTAINS a reference equivalent")
        if c:
            cats.append("relaxed match: mined top-1 is CONTAINED IN a reference equivalent")
        k0 = sk[0]
        strip = lambda x: " ".join(x.split()[:-1]) if len(x.split()) > 1 and x.split()[-1] in ("pa", "ba", "po", "bo", "ma", "mo") else x
        if k0 and any(strip(k0) == strip(r) for r in R):
            cats.append("differs only by final nominaliser pa/ba/po/bo/ma/mo")
        fh = m["strict"]["first_hit_rank"]
        cats.append(f"first strict match at rank {fh}" if fh else "no strict match at any rank")
        rec["sample"] = {"ex": C.highlight(ex0), "cats": cats, "relaxed_p1": m["relaxed"]["p1"]}
    return rec


def main():
    t0 = time.time()
    st = collections.Counter()
    pairs, st = negi.load_negi(tib_key, st)
    index = collections.defaultdict(set)
    if os.environ.get("NEGI_ATTESTED"):     # keep only equivalents with a textual citation in Negi
        keep = ("own", "xref")      # equivalents with a textual citation, directly or via the cross-referenced entry
        pairs = [p for p in pairs if p[3] in keep]
    for hw, tk, k, src in pairs:
        if " " in k or not tk:
            continue
        for cand, rule in stem_candidates(k):
            index[cand].add(tk)
    index = {k: tuple(sorted(v)) for k, v in index.items()}
    # known-pair verification of the inverted index
    checks, sizes = {}, {}
    for sa, bo in [("karuṇā", "snying rje"), ("dharma", "chos"), ("śūnyatā", "stong pa nyid"),
                   ("bodhisattva", "byang chub sems dpa'")]:
        checks[f"{sa} -> {bo}"] = tib_key(bo) in index.get(sa_key(sa), ())
        sizes[sa] = len(index.get(sa_key(sa), ()))
        print("CHECK", sa, "->", bo, checks[f"{sa} -> {bo}"], "| #Tibetan equivalents in inverted Negi:", sizes[sa])
    assert all(checks.values()) or os.environ.get("NEGI_ATTESTED")

    C.CFG.update(sense_field="bo", index=index, evaluate=evaluate)
    light, recs = C.run_eval(MINED)
    print(f"scan done in {time.time()-t0:.0f}s; mined single-word headwords={len(light)}; in E={len(recs)}")

    mined_keys = {k for k, n, e in light}
    # reference Sanskrit items (single-word, normalised citation forms) and how many are reached
    ref_forms = {k for hw, tk, k, src in pairs if " " not in k and tk}
    rules = collections.Counter()
    covered = set()
    for f in sorted(ref_forms):
        for cand, rule in stem_candidates(f):
            if cand in mined_keys:
                rules[rule] += 1
                covered.add(f)
    res = C.aggregate(recs, VARIANTS)
    # chance-agreement control: reference equivalents shuffled among E headwords of the same band
    C.CFG.update(index=C.shuffled_index(index, light))
    _, recs_sh = C.run_eval(MINED)
    res_sh = C.aggregate(recs_sh, VARIANTS)
    C.CFG.update(index=index)
    out = {
        "direction": "sa-bo",
        "mined_file": os.path.join(C.MINED_DIR, MINED),
        "reference": {"name": "Negi, Tibetan-Sanskrit Dictionary (negi_wylie.txt), inverted", "file": negi.NEGI,
                      "parse_stats": dict(st),
                      "single_word_sanskrit_citation_forms": len(ref_forms),
                      "single_word_sanskrit_citation_forms_with_a_mined_headword": len(covered),
                      "stem_rule_usage_(form->mined headword links)": dict(sorted(rules.items())),
                      "known_pair_checks": checks, "known_pair_number_of_tibetan_equivalents": sizes},
        "mined_file_lines": C.STATS["mined_file_lines"],
        "mined_single_word_headwords": C.band_counts([n for k, n, e in light]),
        "E_headwords": C.band_counts([n for k, n, e in light if e]),
        "E_fraction_of_mined_single_word_headwords": len(recs) / len(light),
        "E_fraction_of_mined_attestations": sum(n for k, n, e in light if e) / sum(n for k, n, e in light),
        "E_fraction_of_reference_sanskrit_forms": len(covered) / len(ref_forms),
        "variants": res,
        "shuffled_reference_control": res_sh,
    }
    os.makedirs(os.path.join(C.OUT_DIR, "parts"), exist_ok=True)
    C.write_json(os.path.join(C.OUT_DIR, "parts", f"sa-bo{OUT_TAG}.json"), out)

    fmt = lambda r: (r["sa"], r["n"], r["top"][0], "| ref:", r["refs_show"][:6])
    C.sanity_print(recs, "strict", fmt=fmt)
    C.sanity_print(recs, "relaxed", fmt=fmt)
    for v in VARIANTS:
        print(C.md_table(res[v], f"sa-bo / {v}"))
    for v in ("strict", "relaxed"):
        print(C.md_table(res_sh[v], f"sa-bo / {v} / SHUFFLED-REFERENCE CONTROL"))

    # mismatch samples
    pool = [r for r in recs if "sample" in r]
    L = ["# sa→bo: top-1 mismatches against Negi (n ≥ 20)", "",
         f"Pool A = headwords in E with n≥20 whose top-1 sense has NO STRICT match: {len(pool)}. "
         f"Pool B = subset with no RELAXED match either: {sum(1 for r in pool if not r['sample']['relaxed_p1'])}. "
         "40 random headwords from each (random.Random(42).sample over the pool sorted by headword). "
         "`⟦…⟧` marks the aligned span of the top-1 sense in one example sentence. "
         "Categories are mechanical only.", ""]
    for title, pl in (("Sample A — no strict match", pool),
                      ("Sample B — no relaxed match either", [r for r in pool if not r["sample"]["relaxed_p1"]])):
        L += [f"## {title}", ""]
        for i, r in enumerate(C.sample_pool(pl), 1):
            L.append(f"{i}. **{r['sa']}** (n={r['n']})  ")
            L.append("   mined top-3: " + "; ".join(f"`{s}` ({c})" for s, c in r["top"]) + "  ")
            more = f" … (+{r['nrefs']-12} more)" if r["nrefs"] > 12 else ""
            L.append("   Negi: " + "; ".join(f"`{x}`" for x in r["refs_show"]) + more + "  ")
            L.append("   example: " + r["sample"]["ex"].replace("\n", " ") + "  ")
            L.append("   mechanical: " + "; ".join(r["sample"]["cats"]))
        L.append("")
    open(os.path.join(C.OUT_DIR, f"mismatches-sa-bo{OUT_TAG}.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
