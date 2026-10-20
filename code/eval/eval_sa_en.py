#!/usr/bin/env python3
"""sa->en: mined skt_en_bidict.sa2en.ndjson vs. Monier-Williams (1899 edition, mw99_records(); 2026-10-20) and Apte (1890),
Apte read from alignment-annotator/app/dict.sqlite (zlib-compressed entry HTML per headword).

The reference is definitional prose, so a mined gloss MATCHES iff all its content-word stems (English stopwords
ignored; if the gloss consists of stopwords only, all its stems) occur in the stem set of the reference text for
exactly that headword (homonyms unioned). See english.py for tokenisation / stemming.

Reference text restricted to the headword itself:
  MW72: records nest compounds/derivatives as run-on sub-entries introduced by an italic em dash ("<i>—Śūnya-tā,</i> f. ...").
        head section  = record text before the first sub-entry  -> judged under the record keyword
        sub-entries   = each run-on segment, keyed by its own (de-hyphenated, lower-cased) headword
        Italic (= Sanskrit) material and <span class="sanskrit"> are removed before tokenising.
  AP90: text before "<b>Comp.</b>"; Sanskrit spans and literature citations removed. Keys are nominatives with
        anusvara spelling ("saṃghaṭṭanaṃ --nā", "kauṃkaḥ") -> split, partial "--x" variants dropped, sa_key(),
        then common.stem_candidates() (nominative -> stem).

Variants: mw | mw_noloan | mw_apte | mw_apte_noloan | mw_mainonly   ("noloan": mined senses that merely repeat
the Sanskrit headword, e.g. karuṇā -> "karuṇā", dharma -> "Dharma", are removed before computing the metrics).
Reference recall is NOT computed for this direction (prose has no enumerable list of equivalents).

Run without arguments. Writes notes/eval/parts/sa-en.json and notes/eval/mismatches-sa-en.md. CPU only.
"""
import os, sys, re, html as htmlmod, sqlite3, zlib, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key, stem_candidates, metrics_from_matchsets
import english as E

DB = "/mnt2/code/mitra-dictionaries/alignment-annotator/app/dict.sqlite"
MINED = os.environ.get("EVAL_MINED", "skt_en_bidict.sa2en.ndjson")
OUT_TAG = os.environ.get("EVAL_TAG", "")
VARIANTS = ["mw", "mw_noloan", "mw_apte", "mw_apte_noloan", "mw_mainonly"]
IAST = re.compile(r"^[a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ]+$")

TAG = re.compile(r"<[^>]+>")
LB_IT = re.compile(r'-\s*</i>\s*<div class="div-lb"></div>\s*<i>')
LB = re.compile(r'-\s*<div class="div-lb"></div>\s*')
ITALIC = re.compile(r"<i>.*?</i>", re.S)
SKT_SPAN = re.compile(r'<span class="sanskrit">.*?</span>', re.S)
CITE = re.compile(r"<cite[^>]*>.*?</cite>", re.S)
PAGE = re.compile(r"\[Page[^\]]*\]")


def plain(h, drop_italic):
    h = SKT_SPAN.sub(" ", h)
    h = CITE.sub(" ", h)
    if drop_italic:
        h = ITALIC.sub(" ", h)
    h = TAG.sub(" ", h)
    return re.sub(r"\s+", " ", htmlmod.unescape(h)).strip()


def mw_sections(h):
    """-> head_text, [(sub_headword_key, sub_text), ...]"""
    h = PAGE.sub(" ", h)
    h = LB_IT.sub("", h)
    h = LB.sub("", h)
    parts = h.split("<i>—")
    head = plain(parts[0], True)
    subs = []
    fem = [v for v in ("ā", "ī") if re.search(r"\(<i>%s</i>\),\s*f\." % v, parts[0])]
    for sec in parts[1:]:
        hws = []
        m = re.match(r"\s*([^<,(]+)", sec)
        if m:
            hws.append(m.group(1))
        m2 = re.match(r"[^<]*</i>\s*(?:[mfn]\.|ind\.)?\s*or\s*<i>\s*([^<,(]+)", sec)
        if m2:
            hws.append(m2.group(1))
        txt = plain("<i>" + sec, True)
        for hw in hws:
            k = sa_key(re.sub(r"[-\s°.]", "", hw))
            if k and IAST.match(k):
                subs.append((k, txt))
    return head, subs, fem

MW99 = os.environ.get("MW99", "/mnt2/code/mitra-dictionaries/refdicts/mw99-20261020/mw.txt")
_CARD = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
_ORD = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
        "sixteenth seventeenth eighteenth nineteenth twentieth").split()
NUMW = {**{str(i + 1): w for i, w in enumerate(_CARD)}, **{"30": "thirty", "40": "forty", "50": "fifty", "60": "sixty",
        "70": "seventy", "80": "eighty", "90": "ninety", "100": "hundred", "1000": "thousand"},
        **{f"{i + 1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'}": w for i, w in enumerate(_ORD)},
        "100th": "hundredth", "1000th": "thousandth"}
NUM = re.compile(r"\b\d+(?:st|nd|rd|th)?\b")       # the 1899 edition writes numerals as digits ("of 4 kinds", "10th")
MW99_DROP = re.compile(r"<(s|ls|gk|etym|lang|hom|old|arab|pb)\b[^>]*>.*?</\1>", re.S)


def mw99_records():
    """Monier-Williams 1899 from the Cologne source (sanskrit-lexicon/csl-orig v02/mw/mw.txt): one record per
    headword, derivative, compound or sense paragraph (<e> 1/2 = headword/derivative, 3/4 = compound, A/B/... =
    further sense paragraphs). -> (IAST key, e, English text). The text is the part after the headword marker "¦";
    Sanskrit (<s>), literature references (<ls>), Greek/etymology/language labels, homonym numbers and superseded
    readings (<old>) are removed, every other tag is unwrapped (abbreviations, <s1> names, <bot> plant names kept).
    The 1899 edition abbreviates words repeated within an article ("gr˚ man", "th˚-eyed"): such a word is expanded to
    the most frequent word with that prefix in the article (records since the last <e>1 record), else left as the
    prefix. Numerals up to 20 and round numbers are written out ("4" -> "four", "10th" -> "tenth"),
    as in the 1872 edition. Records without English content of their own that only refer to another headword ("= tūṇa", "id.") take
    the text of that headword (one level; "id." = the preceding headword)."""
    from indic_transliteration import sanscript
    slp = lambda x: sanscript.transliterate(x, sanscript.SLP1, sanscript.IAST)
    NOCONTENT = {"mfn", "ind", "see", "col", "sch", "comm", "and", "the", "also", "with", "for", "acc", "nom", "gen",
                 "loc", "abl", "instr", "dat", "voc", "partic", "esp", "prob", "perhaps", "cf"}
    recs = []
    k1 = e = None; body = []
    for line in open(MW99, encoding="utf-8"):
        if line.startswith("<L>"):
            k1 = re.search(r"<k1>([^<]+)", line).group(1); e = re.search(r"<e>([^<\s]+)", line).group(1); body = []
        elif line.startswith("<LEND>"):
            t = " ".join(body)
            head, raw = t.split("¦", 1) if "¦" in t else ("", t)
            hs = re.search(r"<s>([^<]*)</s>", head)
            pre = hs.group(1).split("˚")[0] if hs and "˚" in hs.group(1) else ""
            refs = [r if not r.startswith("˚") else pre + r[1:] for r in re.findall(r"=\s*<s>([^<]+)</s>", raw)]
            t = MW99_DROP.sub(" ", raw)
            t = re.sub(r"<[^>]+>", " ", t)
            t = NUM.sub(lambda m: NUMW.get(m.group(0).lower(), m.group(0)), t)
            recs.append([slp(k1), e, re.sub(r"\s+", " ", t).strip(), refs, "<ab>id.</ab>" in raw])
        else:
            body.append(line.strip())
    # expand abbreviated words per article
    art, start = 0, 0
    def expand(lo, hi):
        words = collections.Counter(w.lower() for r in recs[lo:hi] for w in re.findall(r"[A-Za-z]{3,}", r[2])
                                    if w.lower() not in E.STOPWORDS)
        for r in recs[lo:hi]:
            if "˚" in r[2]:
                def rep(m):
                    p = m.group(1).lower()
                    c = [(n, w) for w, n in words.items() if w.startswith(p) and len(w) > len(p)]
                    return max(c)[1] if c else m.group(1)
                r[2] = re.sub(r"([A-Za-z]+)˚", rep, r[2]).replace("˚", " ")
    for i, r in enumerate(recs):
        if r[1] == "1" and i > start:
            expand(start, i); start = i
    expand(start, len(recs))
    own = collections.defaultdict(list)
    stub = lambda r: not any(w.lower() not in NOCONTENT for w in re.findall(r"[A-Za-z]{3,}", r[2]))
    for r in recs:
        if not stub(r):
            own[r[0]].append(r[2])
    cur = prev = None                                  # prev = the preceding headword (for "id.")
    for r in recs:
        k, e, t, refs, idem = r
        if k != cur:
            prev, cur = cur, k
        yield k, e, t
        if stub(r):
            targets = [slp(x.replace("-", "").replace("—", "")) for x in refs]
            if idem and prev:
                targets.append(prev)
            for x in targets:
                for t2 in own.get(x, []):
                    yield k, e + "ref", t2


def apte_keys(keyword):
    kw = re.sub(r"\([^)]*\)", "", keyword)
    out = []
    for part in re.split(r"[,;]", kw):
        for tok in part.split():
            if tok.startswith(("--", "°", "-")):
                continue
            k = sa_key(tok)
            if k and IAST.match(k):
                out.append(k)
    return out


def load_reference():
    st = collections.Counter()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ref = collections.defaultdict(lambda: {"mw": set(), "mw_main": set(), "ap": set(), "mw_txt": [], "ap_txt": []})
    for k1, e, txt in mw99_records():
        if e.endswith("ref"):
            st["mw99_crossref_texts"] += 1
        else:
            st["mw72_records"] += 1                  # stat names kept from the 1872 loader (make_report.py); = MW 1899
        k = sa_key(k1)
        if not k or not IAST.match(k):
            st["mw72_records_key_rejected"] += 1
            continue
        ss = E.stem_set(txt)
        r = ref[k]
        r["mw"] |= ss; r["mw_txt"].append(txt)
        if e[:1] in ("1", "2"):                       # headwords and derivatives; 3/4 = compounds
            r["mw_main"] |= ss
        else:
            st["mw72_subentries"] += 1
    mw_keys = {k for k, r in ref.items() if r["mw"]}
    mw_main_keys = {k for k, r in ref.items() if r["mw_main"]}
    ap_forms = set()
    for keyword, blob in con.execute("select keyword, html from entries where dict='ap90' order by ord"):
        st["ap90_records"] += 1
        h = zlib.decompress(blob).decode("utf-8")
        h = h.split("<b>Comp.</b>")[0]
        txt = plain(h, False)
        ss = E.stem_set(txt)
        keys = apte_keys(keyword)
        if not keys:
            st["ap90_records_key_rejected"] += 1
        for k in keys:
            ap_forms.add(k)
            for cand, rule in stem_candidates(k):
                r = ref[cand]
                r["ap"] |= ss
                if rule == "exact" or len(r["ap_txt"]) == 0:
                    r["ap_txt"].append(txt)
    index = {}
    for k, r in ref.items():
        if not (r["mw"] or r["ap"]):
            st["reference_keys_with_empty_text_dropped"] += 1      # e.g. pure cross-reference stubs
            continue
        index[k] = {"mw": frozenset(r["mw"]) or None, "mw_main": frozenset(r["mw_main"]) or None,
                    "ap": frozenset(r["ap"]) or None,
                    "mw_txt": " ‖ ".join(r["mw_txt"])[:300], "ap_txt": " ‖ ".join(r["ap_txt"])[:300]}
    st["mw72_distinct_keys_main"] = len(mw_main_keys)
    st["mw72_distinct_keys_main_plus_sub"] = len(mw_keys)
    st["ap90_distinct_citation_forms"] = len(ap_forms)
    return index, st, mw_keys, mw_main_keys, ap_forms


def is_loan(gloss, sa):
    g = re.sub(r"[^a-z]", "", E.fold(gloss))
    h = re.sub(r"[^a-z]", "", E.fold(sa))
    if not g or not h:
        return False
    forms = {h, h + "s"}
    if h.endswith("n"):
        forms |= {h[:-1], h[:-1] + "s"}          # karman -> karma, ātman -> atma
    if g.startswith("the") and g[3:] in forms:
        return True
    return g in forms


def evaluate(sa, key, n, senses, ref, ex0):
    counts = [c for _, c in senses]
    gl = []
    for s, _ in senses:
        content, allst = E.gloss_stems(s)
        gl.append(content if content else allst)
    loan = [is_loan(s, sa) for s, _ in senses]

    def ms(refset):
        return [({0} if (g and all(t in refset for t in g)) else set()) for g in gl]

    m = {}
    sets = {}
    if ref["mw"]:
        sets["mw"] = ref["mw"]
    if ref["mw_main"]:
        sets["mw_mainonly"] = ref["mw_main"]
    if ref["mw"] or ref["ap"]:
        sets["mw_apte"] = (ref["mw"] or frozenset()) | (ref["ap"] or frozenset())
    keep = [i for i, l in enumerate(loan) if not l]
    for v, rs in sets.items():
        full = ms(rs)
        m[v] = metrics_from_matchsets(counts, full, 0)
        if v in ("mw", "mw_apte") and keep:
            m[v + "_noloan"] = metrics_from_matchsets([counts[i] for i in keep], [full[i] for i in keep], 0)
    rec = {"sa": sa, "n": n, "m": m, "top": senses[:3], "loan_top1": loan[0] if loan else False,
           "all_loan": not keep, "has_mw": bool(ref["mw"]), "has_ap": bool(ref["ap"]),
           "refs_show": [(ref["mw_txt"] or ref["ap_txt"])[:80]]}
    prim = "mw" if "mw" in m else "mw_apte"
    if n >= 20 and ((("mw" in m) and not m["mw"]["p1"]) or not m["mw_apte"]["p1"]):
        rs = sets["mw_apte"]
        missing = [t for t in gl[0] if t not in rs]
        cats = []
        if loan and loan[0]:
            cats.append("top-1 gloss repeats the Sanskrit headword (untranslated loanword)")
        if senses and not E.gloss_stems(senses[0][0])[0]:
            cats.append("top-1 gloss consists of stopwords only")
        if len(gl[0]) > 1 and len(missing) == 1:
            cats.append("multi-word gloss, exactly one content stem missing")
        for v in ("mw", "mw_apte"):
            if v in m:
                fh = m[v]["first_hit_rank"]
                cats.append(f"[{v}] " + (f"first match at rank {fh}" if fh else "no match at any rank"))
        rec["sample"] = {"ex": C.highlight(ex0), "cats": cats, "missing": missing,
                         "mw_txt": ref["mw_txt"], "ap_txt": ref["ap_txt"],
                         "mw_p1": m["mw"]["p1"] if "mw" in m else None, "mwap_p1": m["mw_apte"]["p1"]}
    return rec


def main():
    t0 = time.time()
    index, st, mw_keys, mw_main_keys, ap_forms = load_reference()
    print(f"reference loaded in {time.time()-t0:.0f}s:", dict(st))
    for k in ("karuṇā", "dharma", "śūnyatā", "bodhisattva", "kṛ"):
        r = index.get(sa_key(k))
        print("CHECK", k, "mw:", bool(r and r["mw"]), "apte:", bool(r and r["ap"]), "|", (r or {}).get("mw_txt", "")[:120])
    C.CFG.update(sense_field="en", index=index, evaluate=evaluate)
    light, recs = C.run_eval(MINED)
    print(f"scan done {time.time()-t0:.0f}s; mined single-word headwords={len(light)}; in E(mw∪apte)={len(recs)}")
    res = C.aggregate(recs, VARIANTS)

    # chance-agreement control: reference entries shuffled among E headwords of the same band / same availability
    strat = lambda k: (bool(index[k]["mw"]), bool(index[k]["ap"]))
    C.CFG.update(index=C.shuffled_index(index, light, strat))
    _, recs_sh = C.run_eval(MINED)
    res_sh = C.aggregate(recs_sh, VARIANTS)
    C.CFG.update(index=index)

    mined_keys = {k for k, n, e in light}
    nlight = len(light)
    tot_att = sum(n for k, n, e in light)
    E_mw = [(k, n) for k, n, e in light if e and index[k]["mw"]]
    E_mwmain = [(k, n) for k, n, e in light if e and index[k]["mw_main"]]
    E_all = [(k, n) for k, n, e in light if e]
    ap_cov = {f for f in ap_forms if any(c in mined_keys for c, r in stem_candidates(f))}
    out = {
        "direction": "sa-en", "mined_file": os.path.join(C.MINED_DIR, MINED),
        "reference": {"name": "Monier-Williams 1899 (Cologne csl-orig mw.txt) and Apte 1890 (ap90) from dict.sqlite", "file": MW99 + " ; " + DB,
                      "parse_stats": dict(st)},
        "mined_file_lines": C.STATS["mined_file_lines"],
        "mined_single_word_headwords": C.band_counts([n for k, n, e in light]),
        "E_headwords": {"mw": C.band_counts([n for k, n in E_mw]), "mw_mainonly": C.band_counts([n for k, n in E_mwmain]),
                        "mw_apte": C.band_counts([n for k, n in E_all])},
        "E_fraction_of_mined_single_word_headwords": {"mw": len(E_mw) / nlight, "mw_mainonly": len(E_mwmain) / nlight,
                                                      "mw_apte": len(E_all) / nlight},
        "E_fraction_of_mined_attestations": {"mw": sum(n for k, n in E_mw) / tot_att,
                                             "mw_apte": sum(n for k, n in E_all) / tot_att},
        "E_fraction_of_reference_sanskrit_items": {
            "mw_main_plus_sub_keys": len(mw_keys & mined_keys) / len(mw_keys),
            "mw_main_keys": len(mw_main_keys & mined_keys) / len(mw_main_keys),
            "apte_citation_forms": len(ap_cov) / len(ap_forms),
            "counts": {"mw_keys": len(mw_keys), "mw_keys_mined": len(mw_keys & mined_keys),
                       "mw_main_keys": len(mw_main_keys), "mw_main_keys_mined": len(mw_main_keys & mined_keys),
                       "apte_forms": len(ap_forms), "apte_forms_mined": len(ap_cov)}},
        "headwords_whose_senses_are_all_loanwords": {"mw": sum(1 for r in recs if r["all_loan"] and r["has_mw"]),
                                                     "mw_apte": sum(1 for r in recs if r["all_loan"])},
        "headwords_with_loanword_top1": {"mw": sum(1 for r in recs if r["loan_top1"] and r["has_mw"]),
                                         "mw_apte": sum(1 for r in recs if r["loan_top1"])},
        "variants": res,
        "shuffled_reference_control": res_sh,
    }
    os.makedirs(os.path.join(C.OUT_DIR, "parts"), exist_ok=True)
    C.write_json(os.path.join(C.OUT_DIR, "parts", f"sa-en{OUT_TAG}.json"), out)

    fmt = lambda r: (r["sa"], r["n"], r["top"][0], "| ref:", r["refs_show"][0])
    C.sanity_print(recs, "mw", fmt=fmt)
    C.sanity_print(recs, "mw_apte", fmt=fmt)
    for v in VARIANTS:
        print(C.md_table(res[v], f"sa-en / {v}"))
    for v in ("mw", "mw_apte"):
        print(C.md_table(res_sh[v], f"sa-en / {v} / SHUFFLED-REFERENCE CONTROL"))

    pool = [r for r in recs if "sample" in r]
    poolA = [r for r in pool if r["sample"]["mw_p1"] == 0]
    poolB = [r for r in pool if not r["sample"]["mwap_p1"]]
    L = ["# sa→en: top-1 mismatches against MW72 / Apte (n ≥ 20)", "",
         f"Pool A = headwords in E(MW72) with n≥20 whose top-1 gloss does NOT match the MW72 text: {len(poolA)}. "
         f"Pool B = headwords in E(MW72∪Apte) with n≥20 whose top-1 gloss matches NEITHER: {len(poolB)}. "
         "40 random headwords from each (random.Random(42).sample over the pool sorted by headword). "
         "`⟦…⟧` marks the aligned span of the top-1 gloss. `missing` = content stems of the top-1 gloss absent "
         "from MW72∪Apte. Reference texts truncated to 300 characters. Categories are mechanical only.", ""]
    for title, pl in (("Sample A — top-1 not matched by MW72", poolA), ("Sample B — top-1 matched by neither MW72 nor Apte", poolB)):
        L += [f"## {title}", ""]
        for i, r in enumerate(C.sample_pool(pl), 1):
            s = r["sample"]
            L.append(f"{i}. **{r['sa']}** (n={r['n']})  ")
            L.append("   mined top-3: " + "; ".join(f"`{g}` ({c})" for g, c in r["top"]) + "  ")
            L.append("   MW72: " + (s["mw_txt"] or "—") + "  ")
            L.append("   Apte: " + (s["ap_txt"] or "—") + "  ")
            L.append("   example: " + s["ex"].replace("\n", " ") + "  ")
            L.append(f"   missing stems: {s['missing']}; mechanical: " + "; ".join(s["cats"]))
        L.append("")
    open(os.path.join(C.OUT_DIR, f"mismatches-sa-en{OUT_TAG}.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
