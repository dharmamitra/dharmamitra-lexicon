#!/usr/bin/env python3
"""Feasibility test: can the Negi equivalents that the mined Sanskrit->Tibetan dictionary lacks
be recovered from the corpus by string matching?

For every (headword, Negi equivalent) pair that no mined sense matches even under the relaxed
criterion (build/eval/novel.py), count the sentence pairs of the aligned corpus in which the
Sanskrit lemma occurs AND the Tibetan side contains the equivalent (contiguous syllables, final
particle / clitic normalised as in tibetan.tib_key). Each co-occurrence is classified by what the
student did with the headword in that sentence:

  free      the headword token is unaligned and no span aligned to another word contains the string
  taken     the string lies inside (or contains) a span that the student aligned to ANOTHER word
  elsewhere the student aligned the headword to a span that does not overlap the string
  overlap   the student's span for the headword overlaps the string (boundary variant)

Also counts each string over the whole corpus, for an association score against chance.
Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/negi-recover.json, /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/negi-recover-samples.md
"""
import os, ast, collections, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import novel
from tibetan import tib_key, _CLEAN, SUFFIX, FUSED

CORPUS = __import__("os").environ.get("CORPUS_SA_BO", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-bo/hybrid/final.ndjson")
OUT = "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018"
WANT = {}        # tibetan key -> set of headword keys that lack it
STRINGS = set()  # all wanted tibetan keys
MAXLEN = 1


def norm_last(s, multi):
    for suf in SUFFIX:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[:-len(suf)]
            break
    return FUSED.get(s, s) if multi else s


def sylls(text):
    return _CLEAN.sub(" ", (text or "").replace("’", "'").replace("ga[-]y", "g.y")).split()


def occurrences(sy):
    """wanted keys occurring in a syllable list -> set of keys"""
    found = set()
    n = len(sy)
    for i in range(n):
        for L in range(1, min(MAXLEN, n - i) + 1):
            k = " ".join(sy[i:i + L - 1] + [norm_last(sy[i + L - 1], L > 1)])
            if k in STRINGS:
                found.add(k)
    return found


def worker(chunk):
    str_count = collections.Counter()      # sentences containing the string
    lem_count = collections.Counter()      # sentences containing the lemma (only lemmas of interest)
    cooc = collections.defaultdict(collections.Counter)   # (lemma, string) -> category counts
    examples = {}
    nrows = 0
    for line in C.iter_lines(*chunk):
        d = json.loads(line)
        toks = d.get("toks")
        if isinstance(toks, str):
            toks = ast.literal_eval(toks)
        if not toks or not d.get("target"):
            continue
        nrows += 1
        found = occurrences(sylls(d["target"]))
        for k in found:
            str_count[k] += 1
        lemmas = collections.defaultdict(list)
        for i, t in enumerate(toks, 1):
            lemmas[C.sa_key(t[1] or t[0] or "")].append(i)
        for lem in lemmas:
            if lem in LEMMAS:
                lem_count[lem] += 1
        if not found:
            continue
        aligns = {int(n): " " + tib_key(v) + " " for n, v in (d.get("aligns") or {}).items()}
        for k in found:
            kp = " " + k + " "
            for lem in WANT[k]:
                idx = lemmas.get(lem)
                if not idx:
                    continue
                own = [aligns[i] for i in idx if i in aligns]
                other = [v for n, v in aligns.items() if n not in idx]
                if any(kp in o or o in kp for o in other):
                    cat = "taken"
                elif not own:
                    cat = "free"
                elif any(set(kp.split()) & set(o.split()) for o in own):
                    cat = "overlap"
                else:
                    cat = "elsewhere"
                cooc[(lem, k)][cat] += 1
                if cat == "free" and (lem, k) not in examples:
                    examples[(lem, k)] = (d.get("src_original", "")[:200], d["target"][:200])
    return str_count, lem_count, cooc, examples, nrows


def main():
    global MAXLEN, LEMMAS
    cap = novel.run("sa-bo")
    # The mined file can hold several headwords with the same normalised key (spelling variants such as
    # liṅga / liṁga). An equivalent is missing only if NO headword of that key has a matching sense.
    by_key = collections.defaultdict(list)
    for r in cap["recs"]:
        if r.get("ref_missing") is not None:
            by_key[C.sa_key(r["sa"])].append(r)
    pairs, naive = {}, 0
    for key, rs in by_key.items():
        naive += sum(len(r["ref_missing"]) for r in rs)
        main = max(rs, key=lambda r: r["n"])
        for x in set.intersection(*[set(r["ref_missing"]) for r in rs]):
            pairs[(key, x)] = (main["sa"], sum(r["n"] for r in rs), main["senses"][:3])
            WANT.setdefault(x, set()).add(key)
    print(f"{len(by_key)} headword keys from {len(cap['recs'])} mined headwords in E "
          f"({sum(1 for rs in by_key.values() if len(rs) > 1)} keys with several spellings); "
          f"missing pairs counted per mined headword: {naive}, per key: {len(pairs)}")
    STRINGS.update(WANT)
    MAXLEN = max(len(k.split()) for k in STRINGS)
    LEMMAS = {lem for lem, _ in pairs}
    print(f"{len(pairs)} missing (headword, Negi equivalent) pairs; {len(STRINGS)} distinct strings, "
          f"up to {MAXLEN} syllables; {len(LEMMAS)} headwords")
    parts = C.parallel_scan(CORPUS, worker)
    str_count, lem_count, cooc, examples = collections.Counter(), collections.Counter(), {}, {}
    N = 0
    for sc, lc, co, ex, n in parts:
        str_count.update(sc); lem_count.update(lc); N += n
        for k, v in co.items():
            cooc.setdefault(k, collections.Counter()).update(v)
        for k, v in ex.items():
            examples.setdefault(k, v)
    rows = []
    for (lem, k), (sa, n, top) in pairs.items():
        c = cooc.get((lem, k), {})
        tot = sum(c.values())
        # pointwise mutual information of string and lemma over sentence pairs
        pmi = math.log2(tot * N / (lem_count[lem] * str_count[k])) if tot and lem_count[lem] and str_count[k] else None
        rows.append(dict(sa=sa, n=n, bo=k, cooc=tot, free=c.get("free", 0), taken=c.get("taken", 0),
                         elsewhere=c.get("elsewhere", 0), overlap=c.get("overlap", 0),
                         lemma_sents=lem_count[lem], string_sents=str_count[k], pmi=pmi,
                         top=[(g, cnt) for g, cnt, *_ in top]))
    P = len(rows)
    def share(f):
        m = sum(1 for r in rows if f(r)); return f"{m:6d} ({100*m/P:4.1f}%)"
    summary = {
        "pairs": P, "corpus_sentence_pairs": N,
        "string_occurs_anywhere_in_corpus": sum(1 for r in rows if r["string_sents"]),
        "cooccur>=1": sum(1 for r in rows if r["cooc"] >= 1),
        "cooccur>=3": sum(1 for r in rows if r["cooc"] >= 3),
        "free>=1": sum(1 for r in rows if r["free"] >= 1),
        "free>=3": sum(1 for r in rows if r["free"] >= 3),
        "free>=1_and_pmi>=3": sum(1 for r in rows if r["free"] >= 1 and (r["pmi"] or 0) >= 3),
        "free>=3_and_pmi>=3": sum(1 for r in rows if r["free"] >= 3 and (r["pmi"] or 0) >= 3),
        "free>=3_and_pmi>=3_and_multisyllable": sum(1 for r in rows if r["free"] >= 3 and (r["pmi"] or 0) >= 3 and " " in r["bo"]),
        "attestations_free_in_free>=3_pmi>=3": sum(r["free"] for r in rows if r["free"] >= 3 and (r["pmi"] or 0) >= 3),
    }
    print(f"corpus: {N:,} sentence pairs")
    print("string occurs anywhere in the corpus     ", share(lambda r: r["string_sents"] > 0))
    print("co-occurs with the lemma >= 1            ", share(lambda r: r["cooc"] >= 1))
    print("co-occurs with the lemma >= 3            ", share(lambda r: r["cooc"] >= 3))
    print("  ... of the co-occurrences: ", {c: sum(r[c] for r in rows) for c in ("free", "taken", "elsewhere", "overlap")})
    print("free co-occurrence >= 1                  ", share(lambda r: r["free"] >= 1))
    print("free co-occurrence >= 3                  ", share(lambda r: r["free"] >= 3))
    print("free >= 1 and PMI >= 3                   ", share(lambda r: r["free"] >= 1 and (r["pmi"] or 0) >= 3))
    print("free >= 3 and PMI >= 3                   ", share(lambda r: r["free"] >= 3 and (r["pmi"] or 0) >= 3))
    print("free >= 3 and PMI >= 3, >1 syllable      ", share(lambda r: r["free"] >= 3 and (r["pmi"] or 0) >= 3 and " " in r["bo"]))
    print("free attestations in (free>=3, PMI>=3):", summary["attestations_free_in_free>=3_pmi>=3"])
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(OUT, "negi-recover.json"), "w"), ensure_ascii=False)
    rnd = random.Random(C.SEED)
    L = ["# Recovering Negi equivalents by string matching: samples", ""]
    for title, f in (("free >= 3 and PMI >= 3 (the candidates), 50 random", lambda r: r["free"] >= 3 and (r["pmi"] or 0) >= 3),
                     ("free >= 1 but PMI < 3 or free < 3, 30 random", lambda r: r["free"] >= 1 and not (r["free"] >= 3 and (r["pmi"] or 0) >= 3)),
                     ("co-occurring but never free, 30 random", lambda r: r["cooc"] >= 1 and r["free"] == 0),
                     ("string in the corpus but never with the lemma, 20 random", lambda r: r["string_sents"] > 0 and r["cooc"] == 0),
                     ("string not in the corpus at all, 20 random", lambda r: r["string_sents"] == 0)):
        pool = [r for r in rows if f(r)]
        L += [f"## {title} (of {len(pool)})", ""]
        for r in rnd.sample(pool, min(50 if "candidates" in title else 30 if "30" in title else 20, len(pool))):
            ex = examples.get((C.sa_key(r["sa"]), r["bo"]))
            L.append(f"- **{r['sa']}** (n={r['n']}) : Negi `{r['bo']}` | cooc {r['cooc']} (free {r['free']}, taken {r['taken']}, "
                     f"elsewhere {r['elsewhere']}, overlap {r['overlap']}); lemma in {r['lemma_sents']} sents, string in {r['string_sents']}; "
                     f"PMI {r['pmi'] if r['pmi'] is None else round(r['pmi'], 1)} | mined: " + "; ".join(f"`{g}` ({c})" for g, c in r["top"])
                     + (f"\n  - ex: {ex[0]} || {ex[1]}" if ex else ""))
        L.append("")
    open(os.path.join(OUT, "negi-recover-samples.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
