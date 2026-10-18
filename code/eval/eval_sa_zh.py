#!/usr/bin/env python3
"""sa->zh: mined skt_zh_bidict.sa2zh.ndjson vs. Hirakawa, Buddhist Chinese-Sanskrit Dictionary (inverted).

Reference file: mitra-dictionaries-2026/sa-zh/hirakawa_corrected.tsv
   columns: chinese <TAB> sanskrit (raw OCR) <TAB> sanskrit_corrected (LLM-corrected diacritics)
Primary reference = column 3 (sanskrit_corrected), as in the v1 scripts. A sensitivity run adds the
raw OCR column (with mechanical repairs 1->l, ņ->ṇ, ş->ṣ) as "union".

Run without arguments. Writes notes/eval/parts/sa-zh.json and notes/eval/mismatches-sa-zh.md. CPU only.
"""
import os, sys, re, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key, stem_candidates, metrics_from_matchsets

HIRA = os.environ.get("HIRA", "/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026/sa-zh/hirakawa_corrected.tsv")
MINED = os.environ.get("EVAL_MINED", "skt_zh_bidict.sa2zh.ndjson")
OUT_TAG = os.environ.get("EVAL_TAG", "")
VARIANTS = ["strict", "strict_plain", "relaxed", "relaxed_mined_contains_ref"]

try:
    import opencc
    _JP2T, _T2S = opencc.OpenCC("jp2t"), opencc.OpenCC("t2s")
    HAVE_OPENCC = True
except Exception:                      # plain equality if OpenCC is not installed
    HAVE_OPENCC = False

_CJK = re.compile(r"[^㐀-䶿一-鿿豈-﫿\U00020000-\U0002ffff]")
_cache = {}


def zh_plain(s):
    """keep CJK ideographs only (drops punctuation, whitespace, brackets, latin, kana)."""
    return _CJK.sub("", s or "")


def zh_key(s):
    """zh_plain + variant unification: OpenCC jp2t (shinjitai->traditional) then t2s (folds traditional
    variants onto one simplified code point). Applied identically to both sides."""
    p = zh_plain(s)
    if not HAVE_OPENCC or not p:
        return p
    k = _cache.get(p)
    if k is None:
        k = _T2S.convert(_JP2T.convert(p))
        _cache[p] = k
    return k


SA_ITEM = re.compile(r"^[a-zāīūṛṝḷṅñṭḍṇśṣṃḥ']+$")
REFUSAL = re.compile(r"sorry|cannot|Sanskrit text|provide|correct", re.I)


def load_hirakawa(use_ocr_column):
    st = collections.Counter()
    pairs = set()                                  # (sa_form_key, zh_plain)
    allzh = set()
    prev_zh = None
    for ln, line in enumerate(open(HIRA, encoding="utf-8")):
        if ln == 0:
            continue
        st["rows"] += 1
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            st["rows_without_3_columns"] += 1
            continue
        zh = zh_plain(p[0])
        if not zh:
            st["rows_empty_chinese"] += 1
            continue
        allzh.add(zh)
        if zh == prev_zh:
            st["rows_continuation_of_same_chinese_headword"] += 1
        prev_zh = zh
        cols = [p[2]] + ([p[1]] if use_ocr_column else [])
        got = False
        for ci, txt in enumerate(cols):
            if REFUSAL.search(txt):
                st["cells_llm_refusal_text"] += 1
                continue
            if ci == 1:                            # mechanical OCR repairs on the raw column
                txt = re.sub(r"(?<=[A-Za-zāīū])1|1(?=[A-Za-zāīū])", "l", txt).replace("ņ", "ṇ").replace("ş", "ṣ")
            for item in re.split(r"[;,]", txt):
                item = item.strip().strip('"').replace("√", "").replace("-", "").strip().lower()
                if not item:
                    continue
                if " " in item:
                    st["items_multiword_skipped"] += 1
                    continue
                if len(item) < 2 or not SA_ITEM.match(item):
                    st["items_rejected_non_IAST_or_numeric"] += 1
                    continue
                k = sa_key(item)
                if k:
                    pairs.add((k, zh)); got = True
        if got:
            st["rows_with_usable_pair"] += 1
    st["distinct_pairs_sa_zh"] = len(pairs)
    st["chinese_headwords_distinct"] = len(allzh)
    st["chinese_headwords_distinct_len_1"] = sum(1 for z in allzh if len(z) == 1)
    per = collections.Counter(z for _, z in pairs if len(z) == 1)
    st["median_sanskrit_items_per_len_1_headword_with_pair"] = sorted(per.values())[len(per) // 2] if per else 0
    bylen = collections.Counter(min(len(zh), 4) for zh in {z for _, z in pairs})
    for L, c in sorted(bylen.items()):
        st[f"chinese_headwords_with_usable_pair_len_{L if L < 4 else '4plus'}"] = c
    return pairs, st


def build_index(pairs):
    index = collections.defaultdict(set)
    for k, zh in pairs:
        for cand, rule in stem_candidates(k):
            index[cand].add(zh)
    return {k: tuple(sorted(v)) for k, v in index.items()}


def evaluate(sa, key, n, senses, refs, ex0):
    R = list(refs)
    Rk = [zh_key(r) for r in R]
    counts = [c for _, c in senses]
    plain = [zh_plain(s) for s, _ in senses]
    keys = [zh_key(s) for s in plain]
    rplain = {r: i for i, r in enumerate(R)}
    rkeys = collections.defaultdict(set)
    for i, k in enumerate(Rk):
        rkeys[k].add(i)
    cache = {}

    def msets(k):
        if k not in cache:
            strict = set(rkeys.get(k, ())) if k else set()
            sup = {i for i, r in enumerate(Rk) if k and r and i not in strict and r in k}
            sub = {i for i, r in enumerate(Rk) if k and r and i not in strict and i not in sup and k in r}
            cache[k] = (strict, sup, sub)
        return cache[k]

    ms = [msets(k) for k in keys]
    m = {"strict": metrics_from_matchsets(counts, [a for a, b, c in ms], len(R)),
         "strict_plain": metrics_from_matchsets(counts, [({rplain[p]} if p in rplain else set()) for p in plain], len(R)),
         "relaxed": metrics_from_matchsets(counts, [a | b | c for a, b, c in ms], len(R)),
         "relaxed_mined_contains_ref": metrics_from_matchsets(counts, [a | b for a, b, c in ms], len(R))}
    rec = {"sa": sa, "n": n, "m": m, "top": senses[:3], "refs_show": sorted(R, key=lambda x: (len(x), x))[:15],
           "nrefs": len(R)}
    if n >= 20 and not m["strict"]["p1"]:
        a, b, c = ms[0]
        cats = []
        if b:
            cats.append("relaxed match: mined top-1 CONTAINS a reference equivalent")
        if c:
            cats.append("relaxed match: mined top-1 is CONTAINED IN a reference equivalent")
        fh = m["strict"]["first_hit_rank"]
        cats.append(f"first strict match at rank {fh}" if fh else "no strict match at any rank")
        rec["sample"] = {"ex": C.highlight(ex0), "cats": cats, "relaxed_p1": m["relaxed"]["p1"]}
    return rec


def main():
    t0 = time.time()
    out = {"direction": "sa-zh", "mined_file": os.path.join(C.MINED_DIR, MINED),
           "reference": {"name": "Hirakawa, Buddhist Chinese-Sanskrit Dictionary (OCR-derived TSV), inverted",
                         "file": HIRA, "opencc_variant_unification": HAVE_OPENCC}}
    keep = None
    for mode, use_ocr in (("corrected_column", False), ("union_corrected_and_ocr_columns", True)):
        pairs, st = load_hirakawa(use_ocr)
        index = build_index(pairs)
        C.CFG.update(sense_field="zh", index=index, evaluate=evaluate)
        light, recs = C.run_eval(MINED)
        mined_keys = {k for k, n, e in light}
        ref_forms = {k for k, zh in pairs}
        covered = {f for f in ref_forms if any(c in mined_keys for c, r in stem_candidates(f))}
        rules = collections.Counter(r for f in ref_forms for c, r in stem_candidates(f) if c in mined_keys)
        res = C.aggregate(recs, VARIANTS)
        C.CFG.update(index=C.shuffled_index(index, light))      # chance-agreement control (same band)
        _, recs_sh = C.run_eval(MINED)
        res_sh = C.aggregate(recs_sh, VARIANTS)
        C.CFG.update(index=index)
        block = {
            "parse_stats": dict(st),
            "single_word_sanskrit_forms": len(ref_forms),
            "single_word_sanskrit_forms_with_a_mined_headword": len(covered),
            "stem_rule_usage_(form->mined headword links)": dict(sorted(rules.items())),
            "mined_file_lines": C.STATS["mined_file_lines"],
            "mined_single_word_headwords": C.band_counts([n for k, n, e in light]),
            "E_headwords": C.band_counts([n for k, n, e in light if e]),
            "E_fraction_of_mined_single_word_headwords": len(recs) / len(light),
            "E_fraction_of_mined_attestations": sum(n for k, n, e in light if e) / sum(n for k, n, e in light),
            "E_fraction_of_reference_sanskrit_forms": len(covered) / len(ref_forms),
            "variants": res, "shuffled_reference_control": res_sh}
        print(f"== reference mode: {mode}: pairs={len(pairs)} |E|={len(recs)}  ({time.time()-t0:.0f}s)")
        for k, v in sorted(st.items()):
            print(f"   {k:48s} {v}")
        if keep is None:
            keep = (recs, res)
            out.update(block)
        else:
            out["sensitivity_" + mode] = block
        for v in ("strict", "relaxed"):
            print(C.md_table(res[v], f"sa-zh [{mode}] / {v}"))
            print(C.md_table(res_sh[v], f"sa-zh [{mode}] / {v} / SHUFFLED-REFERENCE CONTROL"))
    recs, res = keep
    for sa, zh in [("dharma", "法"), ("karuṇā", "悲"), ("śūnyatā", "空"), ("bodhisattva", "菩薩")]:
        r = [x for x in recs if x["sa"] == sa]
        print("CHECK", sa, zh, "headword in E:", bool(r), "| top:", r[0]["top"][:2] if r else None)
    os.makedirs(os.path.join(C.OUT_DIR, "parts"), exist_ok=True)
    C.write_json(os.path.join(C.OUT_DIR, "parts", f"sa-zh{OUT_TAG}.json"), out)
    fmt = lambda r: (r["sa"], r["n"], r["top"][0], "| ref:", r["refs_show"][:8])
    C.sanity_print(recs, "strict", fmt=fmt)
    C.sanity_print(recs, "relaxed", fmt=fmt)
    for v in VARIANTS:
        print(C.md_table(res[v], f"sa-zh / {v}"))

    pool = [r for r in recs if "sample" in r]
    L = ["# sa→zh: top-1 mismatches against Hirakawa (n ≥ 20)", "",
         f"Pool A = headwords in E with n≥20 whose top-1 sense has NO STRICT match: {len(pool)}. "
         f"Pool B = subset with no RELAXED match either: {sum(1 for r in pool if not r['sample']['relaxed_p1'])}. "
         "40 random headwords from each (random.Random(42).sample over the pool sorted by headword). "
         "`⟦…⟧` marks the aligned span of the top-1 sense in one example sentence. Categories are mechanical only.", ""]
    for title, pl in (("Sample A — no strict match", pool),
                      ("Sample B — no relaxed match either", [r for r in pool if not r["sample"]["relaxed_p1"]])):
        L += [f"## {title}", ""]
        for i, r in enumerate(C.sample_pool(pl), 1):
            L.append(f"{i}. **{r['sa']}** (n={r['n']})  ")
            L.append("   mined top-3: " + "; ".join(f"`{s}` ({c})" for s, c in r["top"]) + "  ")
            more = f" … (+{r['nrefs']-15} more)" if r["nrefs"] > 15 else ""
            L.append("   Hirakawa: " + "; ".join(f"`{x}`" for x in r["refs_show"]) + more + "  ")
            L.append("   example: " + r["sample"]["ex"].replace("\n", " ") + "  ")
            L.append("   mechanical: " + "; ".join(r["sample"]["cats"]))
        L.append("")
    open(os.path.join(C.OUT_DIR, f"mismatches-sa-zh{OUT_TAG}.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
