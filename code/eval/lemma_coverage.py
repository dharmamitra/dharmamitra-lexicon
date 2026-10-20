#!/usr/bin/env python3
"""Lemma coverage of the reference dictionaries.

For each reference (Monier-Williams 1899, Apte 1890, Edgerton BHS 1953, Böhtlingk-Roth PW, Negi inverted, Hirakawa
inverted) the set of Sanskrit keys it contains, normalised to DCS-style stems with the same rules as the evaluation
(common.stem_candidates), and the share of
  (a) DCS lemmas by corpus frequency, and
  (b) the single-word headwords of each MITRA-dict pair with at least 20 attestations
that it contains. Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/lemma-coverage.json and a printed table.
"""
import os, collections, json, os, re, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key, stem_candidates
import negi, tibetan, novel
import eval_sa_en, eval_sa_zh

DB = eval_sa_en.DB
MINED = {"sa-bo": "skt_tib_bidict.sa2bo.ndjson", "sa-zh": "skt_zh_bidict.sa2zh.ndjson",
         "sa-en": "pre-mdc-20260919/skt_en_bidict.sa2en.ndjson"}
VERB_FORM = re.compile(r"(ti|te|nti|nte|yati|yate)$")
# Edgerton (and partly PW) list verbs under the 3rd singular present; DCS lemmatizes to the root or causative stem.
# These headwords are counted but NOT mapped to stems (suffix stripping guesses wrong roots; removed 2026-10-12).
CORPUS = {"sa-bo": __import__("os").environ.get("CORPUS_SA_BO", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-bo/hybrid/final.ndjson"),
          "sa-zh": __import__("os").environ.get("CORPUS_SA_ZH", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-zh/hybrid/final.ndjson")}


def keys_from_forms(forms):
    out = set()
    for f in forms:
        for cand, rule in stem_candidates(f):
            out.add(cand)
    return out


def plain_keys(dict_code):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    forms, verbs = set(), 0
    for (kw,) in con.execute("select keyword from entries where dict=?", (dict_code,)):
        for part in re.split(r"[,;]\s*", kw):
            k = sa_key(re.sub(r"[-/\\^]", "", part))
            if k:
                forms.add(k)
                if VERB_FORM.search(k):
                    verbs += 1
    return keys_from_forms(forms), len(forms), verbs


def main():
    refs, info = {}, {}
    index, st, mw_keys, mw_main_keys, ap_forms = eval_sa_en.load_reference()
    refs["Monier-Williams 1899"] = {k for k, r in index.items() if r["mw"]}
    refs["Apte 1890"] = {k for k, r in index.items() if r["ap"]}
    refs["MW ∪ Apte"] = refs["Monier-Williams 1899"] | refs["Apte 1890"]
    refs["Edgerton BHS"], n, v = plain_keys("bhs")
    info["Edgerton BHS"] = dict(headwords=n, finite_verb_form_headwords=v)
    refs["MW ∪ Apte ∪ BHS"] = refs["MW ∪ Apte"] | refs["Edgerton BHS"]
    refs["Böhtlingk-Roth PW"], n, v = plain_keys("pwg")
    pairs, _ = negi.load_negi(tibetan.tib_key)
    refs["Negi (inverted)"] = keys_from_forms({k for hw, tk, k, src in pairs if " " not in k and tk})
    hpairs, _ = eval_sa_zh.load_hirakawa(True)
    refs["Hirakawa (inverted)"] = set(eval_sa_zh.build_index(hpairs))
    dcs, _, _ = novel.dcs_lemmas()
    # lemmas the analyser produced in each corpus: mined headwords are counted only if they are such lemmas
    # (excludes the sandhied surface entries of compounds, e.g. abdhātuḥ)
    lem_re = re.compile(r"""\[\s*(?:'[^']*'|\\?"[^"\\]*\\?"),\s*(?:'([^']*)'|\\?"([^"\\]*)\\?")\s*\]""")
    analyser = {d: set() for d in MINED}
    for d, p in CORPUS.items():
        for line in open(p, encoding="utf-8"):
            i = line.find('"toks": ')
            analyser[d].update(sa_key(x or y) for x, y in lem_re.findall(line[i:i + 20000]))
    for line in open("/mnt2/code/mitra-dictionaries/qwen-dictionaries/final-output/pre-mdc-20260919/skt_en_spans.ndjson", encoding="utf-8"):
        i = line.find('"lemma": "')
        if i >= 0 and '"type": "token"' in line[:40]:
            analyser["sa-en"].add(sa_key(line[i + 10:line.find('"', i + 10)]))
    mined = {}
    for d, f in MINED.items():
        hs = {}
        for line in open(os.path.join(C.MINED_DIR, f), encoding="utf-8"):
            r = json.loads(line)
            if C.is_single_word(r["sa"]) and r["n"] >= 20:
                k = sa_key(r["sa"])
                if k in analyser[d]:
                    hs[k] = hs.get(k, 0) + r["n"]
        mined[d] = set(hs)
    cols = [("DCS ≥1000", lambda: {k for k, (f, _) in dcs.items() if f >= 1000}),
            ("DCS 100–999", lambda: {k for k, (f, _) in dcs.items() if 100 <= f < 1000}),
            ("DCS 20–99", lambda: {k for k, (f, _) in dcs.items() if 20 <= f < 100}),
            ("DCS 20–999", lambda: {k for k, (f, _) in dcs.items() if 20 <= f < 1000}),
            ("DCS verbs ≥20", lambda: {k for k, (f, p) in dcs.items() if f >= 20 and p == "verb"}),
            ("sa-en n≥20", lambda: mined["sa-en"]), ("sa-bo n≥20", lambda: mined["sa-bo"]),
            ("sa-zh n≥20", lambda: mined["sa-zh"])]
    sets = {c: f() for c, f in cols}
    res = {}
    print(f"{'reference':22} {'keys':>8} " + " ".join(f"{c:>13}" for c, _ in cols))
    print(f"{'(size)':22} {'':>8} " + " ".join(f"{len(sets[c]):>13,d}" for c, _ in cols))
    for name, ks in refs.items():
        row = {c: 100 * len(sets[c] & ks) / len(sets[c]) for c, _ in cols}
        res[name] = dict(keys=len(ks), **row)
        print(f"{name:22} {len(ks):8,d} " + " ".join(f"{row[c]:12.1f}%" for c, _ in cols))
    # what the Buddhist headwords missing from MW ∪ Apte look like, and how many BHS adds
    for d in ("sa-bo", "sa-zh"):
        miss = sorted(mined[d] - refs["MW ∪ Apte"])
        add = [k for k in miss if k in refs["Edgerton BHS"]]
        res[f"{d}_missing_from_MW_Apte"] = len(miss)
        res[f"{d}_of_those_in_BHS"] = len(add)
        print(f"{d}: {len(miss)} headwords (n≥20) not in MW ∪ Apte, {len(add)} of them in BHS; e.g. {miss[:25]}")
    res["_info"] = info
    json.dump(res, open("/home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/lemma-coverage.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
