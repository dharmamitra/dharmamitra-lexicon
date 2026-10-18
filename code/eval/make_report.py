#!/usr/bin/env python3
"""Merge notes/eval/parts/sa-*.json into refdict-agreement.json and write refdict-agreement.md.
Run after eval_sa_bo.py, eval_sa_zh.py, eval_sa_en.py (or use run_all.sh). No arguments.
All result numbers are read from the JSON parts. The only hand-entered figures are three one-off exploration counts
that the scripts do not recompute, each marked as such in the text: the Porter/nltk verification (verify_porter.py
output), the grep count of "ga[-]y" senses, and the number of mixed-case headwords."""
import os, sys, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

P = os.path.join(C.OUT_DIR, "parts")
bo = json.load(open(os.path.join(P, "sa-bo.json"), encoding="utf-8"))
zh = json.load(open(os.path.join(P, "sa-zh.json"), encoding="utf-8"))
en = json.load(open(os.path.join(P, "sa-en.json"), encoding="utf-8"))

merged = {
    "_meta": {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "scripts": "/home/sebastian/papers/mitra-dict-2026/build/eval/{common,negi,tibetan,english,eval_sa_bo,eval_sa_zh,eval_sa_en,make_report}.py",
        "scope": "single-word Sanskrit headwords (no space, hyphen or underscore), Sanskrit->X direction; E = mined headwords that also occur as a Sanskrit item in the reference",
        "bands": "headword frequency n (total attestations of the headword in the mined file)",
        "metric_definitions": {
            "P@k": "share of E headwords with >=1 reference match among the k top-ranked mined senses (file order = count desc)",
            "attw_precision_micro": "sum(count of matching senses) / sum(count of all senses), pooled over E headwords",
            "attw_precision_macro": "mean over E headwords of the per-headword attestation-weighted precision",
            "recall_all_macro": "mean over E headwords of (# reference equivalents matched by ANY mined sense / # reference equivalents)",
            "recall_top10_macro": "same, top-10 mined senses only",
            "shuffled_reference_control": "identical computation after giving every E headword the reference entry of another E headword of the same band (seed 42): chance-agreement floor"},
    },
    "sa-bo": bo, "sa-zh": zh, "sa-en": en,
}
C.write_json(os.path.join(C.OUT_DIR, "refdict-agreement.json"), merged)

pct = C.fmt_pct
BN = [b[0] for b in C.BANDS]


def headline_row(label, E, m, sh=None):
    a = m["all"]
    r = f"| {label} | {E} | {pct(a['P@1'])} | {pct(a['P@3'])} | {pct(a['P@5'])} | {pct(a['attw_precision_micro'])} | {pct(a['attw_precision_macro'])} | {pct(a['recall_all_macro'])} | {pct(a['recall_top10_macro'])} |"
    if sh:
        r += f" {pct(sh['all']['P@1'])} / {pct(sh['all']['attw_precision_micro'])} |"
    else:
        r += " – |"
    return r


def band_row(label, m):
    return f"| {label} | " + " | ".join(f"{pct(m[b]['P@1'])} ({m[b]['headwords']})" if m[b].get("headwords") else "–" for b in BN) + " |"


L = []
A = L.append
A("# Reference-dictionary agreement of the mined Sanskrit→X dictionaries (annotator-free, lexicon level)")
A("")
A(f"Generated {merged['_meta']['generated']} by `build/eval/make_report.py`; all numbers are copied from `refdict-agreement.json` "
  "(nested: direction → `variants` → matching variant → band → metrics + raw counts). CPU only; the three evaluation scripts each run in < 10 s on this machine (48 processes).")
A("")
A("**Reading guide.** Reference dictionaries are incomplete w.r.t. real translation usage, so strict figures are *lower bounds* on precision. "
  "The relaxed figures are *not* bounds: they additionally accept partial overlaps (see rules) and have a small but non-zero chance-agreement floor, which is reported (shuffled-reference control).")
A("")
A("## 1. Headline (all bands pooled)")
A("")
A("| direction / matching variant | \\|E\\| | P@1 | P@3 | P@5 | att-w prec. micro | att-w prec. macro | ref. recall@all (macro) | ref. recall@top10 (macro) | shuffled control P@1 / att-w micro |")
A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
bv, bs = bo["variants"], bo["shuffled_reference_control"]
A(headline_row("sa→bo vs Negi — strict", bo["E_headwords"]["all"], bv["strict"], bs["strict"]))
A(headline_row("sa→bo vs Negi — relaxed (containment either way)", bo["E_headwords"]["all"], bv["relaxed"], bs["relaxed"]))
A(headline_row("sa→bo vs Negi — relaxed, mined ⊇ reference only", bo["E_headwords"]["all"], bv["relaxed_mined_contains_ref"], bs["relaxed_mined_contains_ref"]))
A(headline_row("sa→bo vs Negi — strict, mined senses merged by normalised key", bo["E_headwords"]["all"], bv["strict_merged"], bs["strict_merged"]))
zv, zs = zh["variants"], zh["shuffled_reference_control"]
A(headline_row("sa→zh vs Hirakawa — strict (OpenCC-unified)", zh["E_headwords"]["all"], zv["strict"], zs["strict"]))
A(headline_row("sa→zh vs Hirakawa — strict, plain code-point equality", zh["E_headwords"]["all"], zv["strict_plain"], zs["strict_plain"]))
A(headline_row("sa→zh vs Hirakawa — relaxed (substring either way)", zh["E_headwords"]["all"], zv["relaxed"], zs["relaxed"]))
A(headline_row("sa→zh vs Hirakawa — relaxed, mined ⊇ reference only", zh["E_headwords"]["all"], zv["relaxed_mined_contains_ref"], zs["relaxed_mined_contains_ref"]))
ev, es = en["variants"], en["shuffled_reference_control"]
A(headline_row("sa→en vs MW72 — all senses", ev["mw"]["all"]["headwords"], ev["mw"], es["mw"]))
A(headline_row("sa→en vs MW72 — loanword senses skipped", ev["mw_noloan"]["all"]["headwords"], ev["mw_noloan"], es["mw_noloan"]))
A(headline_row("sa→en vs MW72 ∪ Apte — all senses", ev["mw_apte"]["all"]["headwords"], ev["mw_apte"], es["mw_apte"]))
A(headline_row("sa→en vs MW72 ∪ Apte — loanword senses skipped", ev["mw_apte_noloan"]["all"]["headwords"], ev["mw_apte_noloan"], es["mw_apte_noloan"]))
A(headline_row("sa→en vs MW72 record keywords only (no run-on sub-entries)", ev["mw_mainonly"]["all"]["headwords"], ev["mw_mainonly"], es["mw_mainonly"]))
A("")
A("Reference recall is not defined for sa→en (the reference is prose, not a list of equivalents) and is therefore not reported (n/a).")
A("")
A("### P@1 by headword-frequency band (number of E headwords in parentheses)")
A("")
A("| direction / variant | " + " | ".join(f"n {b}" for b in BN) + " |")
A("|---|" + "---:|" * len(BN))
A(band_row("sa→bo strict", bv["strict"])); A(band_row("sa→bo relaxed", bv["relaxed"]))
A(band_row("sa→zh strict", zv["strict"])); A(band_row("sa→zh relaxed", zv["relaxed"]))
A(band_row("sa→en MW72", ev["mw"])); A(band_row("sa→en MW72, no loanwords", ev["mw_noloan"]))
A(band_row("sa→en MW72∪Apte", ev["mw_apte"])); A(band_row("sa→en MW72∪Apte, no loanwords", ev["mw_apte_noloan"]))
A("")
A("## 2. Data sources and sizes")
A("")
A("| | mined file (`qwen-dictionaries/final-output/`) | lines = headwords | single-word headwords | reference | reference size | \\|E\\| | E / mined single-word headwords | E share of single-word attestations (Σn) | reference Sanskrit items reached |")
A("|---|---|---:|---:|---|---|---:|---:|---:|---:|")
bst = bo["reference"]["parse_stats"]
A(f"| sa→bo | `skt_tib_bidict.sa2bo.ndjson` | {bo['mined_file_lines']:,} | {bo['mined_single_word_headwords']['all']:,} | Negi, `references/negi_wylie.txt` | {bst['lines']:,} lines; {bst['entries_with_sanskrit_equivalent']:,} entries yield ≥1 Sanskrit equivalent; {bst['pairs_tib_skt']:,} (bo, sa) pairs; {bo['reference']['single_word_sanskrit_citation_forms']:,} distinct single-word Sanskrit citation forms | {bo['E_headwords']['all']:,} | {pct(bo['E_fraction_of_mined_single_word_headwords'])}% | {pct(bo['E_fraction_of_mined_attestations'])}% | {bo['reference']['single_word_sanskrit_citation_forms_with_a_mined_headword']:,} = {pct(bo['E_fraction_of_reference_sanskrit_forms'])}% |")
zst = zh["parse_stats"]
A(f"| sa→zh | `skt_zh_bidict.sa2zh.ndjson` | {zh['mined_file_lines']:,} | {zh['mined_single_word_headwords']['all']:,} | Hirakawa, `mitra-dictionaries-2026/sa-zh/hirakawa_corrected.tsv` (column `sanskrit_corrected`) | {zst.get('rows', 0):,} rows; {zst.get('rows_with_usable_pair', 0):,} rows yield a usable pair; {zst.get('distinct_pairs_sa_zh', 0):,} distinct (sa, zh) pairs; {zh['single_word_sanskrit_forms']:,} distinct single-word Sanskrit forms | {zh['E_headwords']['all']:,} | {pct(zh['E_fraction_of_mined_single_word_headwords'])}% | {pct(zh['E_fraction_of_mined_attestations'])}% | {zh['single_word_sanskrit_forms_with_a_mined_headword']:,} = {pct(zh['E_fraction_of_reference_sanskrit_forms'])}% |")
est = en["reference"]["parse_stats"]; ec = en["E_fraction_of_reference_sanskrit_items"]["counts"]
A(f"| sa→en | `skt_en_bidict.sa2en.ndjson` | {en['mined_file_lines']:,} | {en['mined_single_word_headwords']['all']:,} | MW **1872** (`mw72`) + Apte 1890 (`ap90`) in `alignment-annotator/app/dict.sqlite` | MW72: {est['mw72_records']:,} records → {est['mw72_distinct_keys_main']:,} record keys + {est['mw72_subentries']:,} run-on sub-entries + {est['mw72_feminine_forms_keyed']:,} nested feminines = {est['mw72_distinct_keys_main_plus_sub']:,} keys; Apte: {est['ap90_records']:,} records, {est['ap90_distinct_citation_forms']:,} citation forms | MW72: {en['E_headwords']['mw']['all']:,}; MW72∪Apte: {en['E_headwords']['mw_apte']['all']:,} | {pct(en['E_fraction_of_mined_single_word_headwords']['mw'])}% / {pct(en['E_fraction_of_mined_single_word_headwords']['mw_apte'])}% | {pct(en['E_fraction_of_mined_attestations']['mw'])}% / {pct(en['E_fraction_of_mined_attestations']['mw_apte'])}% | MW72 keys {ec['mw_keys_mined']:,}/{ec['mw_keys']:,} = {pct(en['E_fraction_of_reference_sanskrit_items']['mw_main_plus_sub_keys'])}%; Apte forms {ec['apte_forms_mined']:,}/{ec['apte_forms']:,} = {pct(en['E_fraction_of_reference_sanskrit_items']['apte_citation_forms'])}% |")
A("")
A("Number of mined single-word headwords / E headwords per band:")
A("")
A("| | " + " | ".join(BN) + " | all |")
A("|---|" + "---:|" * (len(BN) + 1))
for lab, d in (("sa→bo mined", bo["mined_single_word_headwords"]), ("sa→bo E", bo["E_headwords"]),
               ("sa→zh mined", zh["mined_single_word_headwords"]), ("sa→zh E", zh["E_headwords"]),
               ("sa→en mined", en["mined_single_word_headwords"]), ("sa→en E (MW72)", en["E_headwords"]["mw"]),
               ("sa→en E (MW72∪Apte)", en["E_headwords"]["mw_apte"])):
    A(f"| {lab} | " + " | ".join(f"{d[b]:,}" for b in BN) + f" | {d['all']:,} |")
A("")
A("Other resources inspected and **not used**: `stardict/` and `mac-dict/` (`mitra-skt-tib`, `mitra-skt-chn`, … ) are StarDict / macOS-Dictionary *exports of the project's own mined (v1, 2026) dictionaries* (bookname “MITRA Sanskrit-Tibetan (2026)”, author Dharmamitra), i.e. not independent references. "
  "No stand-alone Mahāvyutpatti file was found there; Mahāvyutpatti is only present indirectly as a source cited inside Negi (siglum म.व्यु.). `references/wogihara.txt` (Wogihara, Sanskrit–Japanese) and the `bhs`/`pwg` tables and the short-gloss table `gloss` (from `dictionary.csv`, provenance not documented) in `dict.sqlite` exist but were not part of the task and were not used. "
  "Circularity check: a case-insensitive grep for hirakawa / negi / monier / mahavy / dict.sqlite over `qwen-dictionaries/*.py` and `qwen-dictionaries/build/*.py` finds nothing, the README describes Gemini-mined seeds + a fine-tuned span model only, and no sense in `skt_tib`/`skt_zh` sa2x files carries a `source` field (in sa2en the 70 hits are the English gloss “source” itself); what the underlying LLMs saw in pre-training cannot be checked. (The v1 script `merge_hirakawa.py` did inject Hirakawa pairs into the *v1* sa-zh dictionary; the v2 file evaluated here contains no such senses.)")
A("")
A("## 3. Rules actually implemented")
A("")
A("**Scope.** Mined entries whose headword contains no space, hyphen or underscore. `n` and sense order are taken from the file as published (senses are *not* merged or re-ranked, except in the explicitly labelled `strict_merged` variant for Tibetan).")
A("")
A("**Sanskrit key (all directions, both sides), `common.sa_key`.** NFC, lowercase, `ṁ→ṃ`, surrounding punctuation/digits stripped, anusvāra before a stop → homorganic nasal (`saṃgha`=`saṅgha`; needed for Apte's `aṃga`-style spelling). Nothing else is changed on the mined side.")
A("")
A("**Citation form → stem (reference side only), `common.stem_candidates`.** Negi, Apte and partly Hirakawa cite nominatives. Each reference form is linked to the mined headword equal to the form itself or to one of: strip `-ḥ` (dharmaḥ→dharma); `-aḥ/-iḥ/-uḥ → -as/-is/-us` (manaḥ→manas); `-āḥ→-a`; strip final `m/ṃ` (phalam→phala); `-ā→-an` (rājā→rājan); `-ā→-ṛ` (kartā→kartṛ); `-ī→-in` (yogī→yogin); `-a→-an` (karma→karman); `-vān/-mān→-vat/-mat`. A candidate has an effect only if it *is* a mined headword; all candidates that are mined headwords receive the reference equivalents (union). "
  f"Links used for Negi: {bo['reference']['stem_rule_usage_(form->mined headword links)']}; for Hirakawa: {zh['stem_rule_usage_(form->mined headword links)']}. Finite verb forms, present participles and other inflected citation forms are not reduced.")
A("")
A("**Negi parsing (`negi.py`).** Line = `Wylie headword TAB body`. Headword: text before the first `/` (Negi writes `brlags/ o nas` for “brlags, also brlags nas”), `* _` removed. Body: backslashes and parenthesised/bracketed material removed; split into groups at `;`, `•`, sense numbers (`1.`, `I.`, `i.`); in each group only the part *before the first em dash* is read (what follows is citation); source sigla with folio numbers, multi-part abbreviations and a whitelist of grammatical abbreviations (वि. पा. क्रि. ना. सं. कृ. अव्य. …) are deleted, everything after द्र./तुल. (“see/cf.”) is dropped; groups whose pre-dash part contains Wylie are skipped unless they start with `=` (synonym cross-reference, then the Devanagari after the Wylie is read); groups containing a daṇḍa are treated as Sanskrit quotations and skipped; items are split at `,` `/`, transliterated Devanagari→IAST (own table), and must consist of IAST letters only; items of ≥5 words are dropped. "
  "Noise filters: (a) *hierarchy rule* – if the header of an enclosing level (entry start < `•` < `I.` < `1.` < `i.`) already supplied an equivalent, Devanagari in its numbered children is Negi's Sanskrit *explanation* of the sub-sense (e.g. `व्यूहराजः 1. समाधिविशेषः … 2. ना. बोधिसत्त्वः`) and is not taken; (b) after ना. (proper name) or पा. (technical term) only the first item of a group is taken (the rest are class descriptors such as बोधिसत्त्वः, प्रज्ञापारमितामुखम्); (c) non-initial items ending in -viśeṣa/-bheda (“a kind of …”) are dropped. "
  "Entries consisting only of a cross-reference (`= X/`) inherit the equivalents of X (one level). "
  f"Counts: {json.dumps(bst, ensure_ascii=False)}. Known-pair checks (all must pass, asserted in the script): {bo['reference']['known_pair_checks']}.")
A("")
A("**Tibetan key (`tibetan.tib_key`, both sides).** Copied/extended from `qwen-dictionaries/build/tib_norm.py` (not imported). (0) repair of a conversion artefact in the mined file: `ga[-]y`→`g.y`; (1) `/ _ * |` → space; (2) iteratively drop trailing particle syllables `gi kyi gyi yi 'i gis kyis gyis yis la na du tu su ru las nas dang ni kyang yang 'ang 'am 'o` while >1 syllable remains (a final `las` is kept after a genitive or after `(')phrin`, where it is the noun “action”), and strip the fused clitics `'i 'o 'am 'ang 'is` from the last syllable; (3) fused case on nominalisers in multi-syllable expressions: `par→pa, bar→ba, pas→pa, bas→ba, por→po, pos→po, bor→bo, bos→bo`. Not done: s-instrumental on other syllables (`rjes`), verb-stem normalisation, removal of `pa/ba/po`. Wylie case is preserved.")
A("- *strict*: normalised strings equal. *relaxed*: strict, or one normalised string is a contiguous **syllable-level** sub-sequence of the other, provided the shorter one contains at least one syllable outside the function-syllable list `FUNC` of `kill_stopword_glosses.py`. *relaxed_mined_contains_ref*: only the direction mined ⊇ reference (`snying rje chen po` ⊇ `snying rje`). *strict_merged*: mined senses with the same normalised key are merged (counts summed) and re-ranked before computing P@k.")
A("")
A("**Hirakawa (`eval_sa_zh.py`).** TSV columns `chinese`, `sanskrit` (raw OCR), `sanskrit_corrected` (LLM-corrected diacritics; this column is the primary reference, as in the v1 scripts). Chinese key: CJK ideographs only. Sanskrit cell split at `, ;`; `√` and hyphens removed; items with spaces (multi-word) skipped; items must be IAST letters only (≥2 letters); cells containing LLM refusal text are skipped. "
  f"Counts: {json.dumps(zst, ensure_ascii=False)}. *strict*: equality after variant unification with OpenCC (`jp2t` then `t2s`, applied to both sides; OpenCC available = {zh['reference']['opencc_variant_unification']}); *strict_plain*: equality of the raw ideograph strings; *relaxed*: strict or substring either way; *relaxed_mined_contains_ref*: only mined ⊇ reference. "
  f"Sensitivity run with the union of both Sanskrit columns (raw OCR column repaired mechanically `1→l`, `ņ→ṇ`, `ş→ṣ`): |E| = {zh['sensitivity_union_corrected_and_ocr_columns']['E_headwords']['all']:,}, strict P@1 = {pct(zh['sensitivity_union_corrected_and_ocr_columns']['variants']['strict']['all']['P@1'])}, relaxed P@1 = {pct(zh['sensitivity_union_corrected_and_ocr_columns']['variants']['relaxed']['all']['P@1'])} (full tables in the JSON under `sensitivity_union_corrected_and_ocr_columns`).")
A("")
A("**MW72 / Apte (`eval_sa_en.py`, `english.py`).** Entry HTML from `dict.sqlite` (zlib). MW72: page markers removed, line-break hyphenation undone (`com- <div class=\"div-lb\">passion` → `compassion`), record split at italic em dashes into a head section (judged under the record keyword; homonymous records unioned) and run-on sub-entries keyed by their own de-hyphenated headword (`—Śūnya-tā, f. or śūnya-tva` → `śūnyatā`, `śūnyatva`); if the head section of an a-stem contains `(ā), f.` / `(ī), f.` the head text is also keyed under the feminine stem (karuṇa→karuṇā). Italic (= Sanskrit) material and `span.sanskrit` are deleted before tokenising, so an untranslated headword cannot match itself through the entry's own lemma line. Apte: text before `<b>Comp.</b>`; `span.sanskrit` and `<cite>` deleted; keys such as `saṃghaṭṭanaṃ --nā`, `vārddhuṣiḥ, vārddhuṣikaḥ` split, partial `--x` forms dropped, then `sa_key` + `stem_candidates`. "
  "English normalisation: lowercase → strip diacritics → tokens `[a-z]+` → British→American folding (`-our→-or`, `-ise→-ize`) → small irregular-form map (said→say, went→go, men→man …, listed in `english.py`) → Porter (1980) stemmer implemented in `english.py` (nltk is not installed in the interpreter used; the implementation was checked against `nltk.PorterStemmer(mode='ORIGINAL_ALGORITHM')` from another conda env on the full vocabulary of mined glosses + MW72/Apte texts, 216,596 tokens: 16 disagreements, all on tokens of ≤2 letters, see `verify_porter.py`). "
  "A mined gloss *matches* iff all its content-word stems (own stopword list in `english.py`; if the gloss has only stopwords, all its stems) are in the stem set of the reference text. "
  f"*noloan* variants: senses whose letters (diacritics folded, optional leading “the”, optional plural -s, optional loss of stem-final -n) equal the headword are removed before computing the metrics; headwords left without any sense are dropped from that variant ({en['headwords_whose_senses_are_all_loanwords']['mw']:,} of the MW72 E set, {en['headwords_whose_senses_are_all_loanwords']['mw_apte']:,} of MW72∪Apte; top-1 is such a loanword for {en['headwords_with_loanword_top1']['mw']:,} / {en['headwords_with_loanword_top1']['mw_apte']:,} headwords).")
A("")
A("## 4. Result tables")
A("")
for title, res in (("sa→bo vs Negi — strict", bv["strict"]), ("sa→bo vs Negi — relaxed (containment either way)", bv["relaxed"]),
                   ("sa→bo vs Negi — relaxed, mined ⊇ reference only", bv["relaxed_mined_contains_ref"]),
                   ("sa→bo vs Negi — strict, senses merged by normalised key", bv["strict_merged"]),
                   ("sa→zh vs Hirakawa — strict (OpenCC-unified)", zv["strict"]), ("sa→zh vs Hirakawa — strict, plain equality", zv["strict_plain"]),
                   ("sa→zh vs Hirakawa — relaxed (substring either way)", zv["relaxed"]),
                   ("sa→zh vs Hirakawa — relaxed, mined ⊇ reference only", zv["relaxed_mined_contains_ref"]),
                   ("sa→en vs MW72 — all senses", ev["mw"]), ("sa→en vs MW72 — loanword senses skipped", ev["mw_noloan"]),
                   ("sa→en vs MW72 ∪ Apte — all senses", ev["mw_apte"]), ("sa→en vs MW72 ∪ Apte — loanword senses skipped", ev["mw_apte_noloan"]),
                   ("sa→en vs MW72 record keywords only", ev["mw_mainonly"])):
    A(C.md_table(res, title))
A("### Chance-agreement floor (shuffled-reference control)")
A("")
A("Same computation after assigning to every E headword the reference entry of a *different* E headword of the same frequency band (for sa→en additionally the same MW/Apte availability); seed 42. This is a single permutation: the ≥1000 band has only a few hundred headwords, so its control value is noisy (for sa→en MW72 P@1 it was 4.8% in a development run with a slightly different reference index; the final value is in the table below).")
A("")
for title, res in (("sa→bo strict — shuffled", bs["strict"]), ("sa→bo relaxed — shuffled", bs["relaxed"]),
                   ("sa→zh strict — shuffled", zs["strict"]), ("sa→zh relaxed — shuffled", zs["relaxed"]),
                   ("sa→en MW72 — shuffled", es["mw"]), ("sa→en MW72∪Apte — shuffled", es["mw_apte"])):
    A(C.md_table(res, title))
A("## 5. Mismatch samples (for manual inspection)")
A("")
A("`mismatches-sa-bo.md`, `mismatches-sa-zh.md`, `mismatches-sa-en.md` (same directory). Each holds two seed-42 samples of 40 headwords with n ≥ 20: "
  "A = top-1 not matched under the strict rule (sa→en: not matched by MW72); B = top-1 not matched under the most lenient rule either (relaxed; sa→en: MW72∪Apte). "
  "Per headword: top-3 mined senses with counts, reference equivalents (MW/Apte text truncated to 300 characters), one example sentence with the aligned span in ⟦…⟧, and purely mechanical flags "
  "(relaxed containment direction, “differs only by final pa/ba/po…”, rank of the first matching sense, loanword top-1, missing stems). No semantic classification was attempted.")
A("")
A("## 6. Caveats, bugs and limitations noticed")
A("")
A("1. **MW is the 1872 edition, not the 1899 one.** `dict.sqlite` contains `mw72` (Monier-Williams 1872), Apte 1890, Edgerton BHS and PW; there is no MW 1899. MW72 nests compounds/derivatives inside the record of the base word; they were recovered by splitting at run-on markers "
  f"({est['mw72_subentries']:,} sub-entries, {est['mw72_feminine_forms_keyed']:,} nested feminines). Without that step |E(MW72)| would be {en['E_headwords']['mw_mainonly']['all']:,} instead of {en['E_headwords']['mw']['all']:,} (variant `mw_mainonly`). Second headwords introduced by “or” inside a sub-entry are only caught in the simplest pattern; participles/derivatives listed in the body of root records without an em dash are not keyed.")
A("2. **sa→en matching is bag-of-stems against prose**, not equivalence: synonyms not used by MW/Apte fail (arrogance vs. “pride, haughtiness”; renunciation vs. “renouncing” – Porter does not conflate these), while any gloss whose words all occur somewhere in a long entry succeeds. The shuffled control quantifies the latter: negligible overall but "
  f"{pct(es['mw']['>=1000']['P@1'])}% P@1 / {pct(es['mw']['>=1000']['attw_precision_micro'])}% att-w micro in the ≥1000 band for MW72 ({pct(es['mw_apte']['>=1000']['P@1'])}% / {pct(es['mw_apte']['>=1000']['attw_precision_micro'])}% for MW72∪Apte), where entries are longest. Hyphenation differences (`anthill` vs. `ant-hill`), archaic spellings and irregular forms outside the small map still cause false negatives. Reference recall cannot be computed for this direction.")
A(f"3. **The Hirakawa file is a noisy, incomplete OCR/LLM derivative.** Of {zst.get('rows', 0):,} rows only {zst.get('rows_with_usable_pair', 0):,} yield a usable pair: {zst.get('rows_without_3_columns', 0):,} rows lack columns, {zst.get('cells_llm_refusal_text', 0):,} `sanskrit_corrected` cells contain LLM refusal text (“I am sorry, but ‘1966’ is not Sanskrit…”, i.e. page/entry numbers were fed to the corrector), {zst['items_rejected_non_IAST_or_numeric']:,} items are numeric/non-IAST. "
  "The LLM ‘correction’ also altered some valid forms (e.g. samantaka→syamantaka, avidyānirodha→avidyānirodhāt, Japanese しばる→“śibāru”). Multi-line dictionary entries were split into rows at line ends, producing word fragments as items (`dha r` / `maiśvarya`, `prati` / `saṃyukta`; "
  f"{zst.get('rows_continuation_of_same_chinese_headword', 0):,} continuation rows). The long main entries of single characters are affected most: of {zst.get('chinese_headwords_distinct_len_1', 0):,} distinct one-character Chinese headwords in the file only {zst.get('chinese_headwords_with_usable_pair_len_1', 0):,} have any usable Sanskrit item (median {zst.get('median_sanskrit_items_per_len_1_headword_with_pair', 0)} items), and in the cases inspected the main entry is reduced to a number (縛: rows contain `2863`, `baddhatva`, `vah` but not bandha; 少: one usable row), so frequent one-character renderings (bandha→縛, alpaka→少) count as strict errors although the printed Hirakawa has them. This depresses strict P@1 and explains a large part of the strict/relaxed gap in sa→zh (relaxed accepts 縛 ⊂ 結縛).")
A("4. **Relaxed matching is lenient in the direction mined ⊂ reference** (a one-syllable / one-character mined sense inside a longer reference equivalent: `yul` in `yul ma yin`, 一 in 一處). The one-directional variant `relaxed_mined_contains_ref` is the safer figure; both are reported. Chance floors for relaxed matching are given in the shuffled control "
  f"(sa→bo ≥1000 band: P@1 {pct(bs['relaxed']['>=1000']['P@1'])}%, att-w micro {pct(bs['relaxed']['>=1000']['attw_precision_micro'])}%; sa→zh ≥1000 band: P@1 {pct(zs['relaxed']['>=1000']['P@1'])}%, att-w micro {pct(zs['relaxed']['>=1000']['attw_precision_micro'])}%).")
A(f"5. **Negi parsing is heuristic.** All {bst['lines']:,} lines split into headword/body ({bst.get('lines_without_tab', 0)} without tab), but {bst['entries_without_sanskrit_equivalent']:,} entries yield no Sanskrit equivalent ({bst['entries_pure_xref_unresolved']:,} of them unresolved cross-references; the rest are entries that only contain citations/sub-entries), {bst['groups_skipped_wylie_subentry']:,} groups with embedded Tibetan sub-entries (e.g. `ka chen महास्तम्भः`) were skipped rather than mined for additional pairs, and {bst['items_rejected_nonIAST']:,} items were rejected after transliteration. "
  "Residual noise remains in the inverted index: Sanskrit explanations that are not equivalents (e.g. `sattāmātre` under chos), synonyms reached through `=` cross-references, and class words after proper names that escaped the filters; e.g. bodhisattva still has " + str(bo['reference']['known_pair_number_of_tibetan_equivalents']['bodhisattva']) + " Tibetan “equivalents”, many of them epithets. Such noise inflates the recall denominator (recall is therefore also a lower bound) and has almost no effect on strict precision (shuffled strict P@1 ≈ 0). Parenthesised variant forms in Negi are ignored. Negi's verbs are cited as finite forms and are not reduced to roots, so verbal lemmas are under-represented in E.")
A("6. **Tibetan normalisation is light.** Inflected verb stems (byed/byas/bya), honorific/plain pairs, optional `pa/ba/po`, abbreviated compounds (`stong nyid` vs `stong pa nyid`) and s-instrumentals on other syllables are *not* unified under the strict rule; many strict ‘errors’ in the samples are of this kind (see flags). Stripping trailing `la/na/dang/ni/las…` is applied symmetrically to both sides, but can conflate lexical finals (e.g. a noun ending in `… la`).")
A("7. **Data-quality finding in the mined sa→bo file:** `g.y-` is written in two ways, `g.yas` and `ga[-]yas` (2,772 sense strings with `ga[-]yo/ya/ye/yu` in the whole sa2bo file according to a one-off grep over the `bo` fields, not recomputed by the scripts, plus rarer `[+]`, `[&]`, `[~]` artefacts of the Unicode→Wylie conversion). The evaluation repairs `ga[-]y`→`g.y`; the published dictionary still contains both spellings as separate senses.")
A(f"8. **E is a small, frequency-biased subset of the headword list** ({pct(bo['E_fraction_of_mined_single_word_headwords'])}% / {pct(zh['E_fraction_of_mined_single_word_headwords'])}% / {pct(en['E_fraction_of_mined_single_word_headwords']['mw'])}% of single-word headwords for bo / zh / en-MW72) but covers most attestations ({pct(bo['E_fraction_of_mined_attestations'])}% / {pct(zh['E_fraction_of_mined_attestations'])}% / {pct(en['E_fraction_of_mined_attestations']['mw'])}% of Σn). A large part of the single-word headwords outside E are unlemmatised surface forms and unsegmented compounds with n = 1 (e.g. `skandhāsaṃgrahādduḥkhasamudayasatyatvaṃ`), which no dictionary can judge. Inflected mined headwords enter E only when the reference happens to cite the identical form (e.g. `svasaṃvedanam`). Mixed-case duplicates (`Buddha` vs `buddha`; 152 / 35 / 20 single-word headwords containing capitals in bo / zh / en, one-off exploration count) are evaluated as separate headwords under the same key.")
A("9. **Low bands are structurally capped:** with n = 1 there is a single sense, so P@1 = P@3 = P@5 = att-w precision, and recall cannot exceed 1/|refs|. The pooled att-w micro precision is dominated by a few hundred very frequent headwords.")
A("10. **Independence.** Negi and Hirakawa excerpt the same canonical translations (Kangyur/Tengyur, Taishō) that form the mined corpus; agreement therefore measures whether the miner recovers the equivalences lexicographers recorded from these texts, not translation quality on unseen material. MW72/Apte are independent of the English translations mined.")
A("11. Sense order in the published files (count-descending) is used as is; ties are left in file order. `P@k` for Tibetan is computed on unmerged surface senses (`snying rje`, `snying rje'i` … occupy separate ranks), which makes P@3/P@5 conservative; `strict_merged` shows the effect of merging.")
A("")
open(os.path.join(C.OUT_DIR, "refdict-agreement.md"), "w", encoding="utf-8").write("\n".join(L))
print("wrote", os.path.join(C.OUT_DIR, "refdict-agreement.json"), "and .md")
