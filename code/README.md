# Code

The scripts that built the released lexica and produced the evaluation in the paper, copied as they were run.

**Paths.** Paths are absolute and refer to the original build machine (`/mnt2/code/mitra-dictionaries/...`,
`/home/sebastian/...`, `/mnt/data/...`). To rerun, change the module-level constants and the environment
variables below. Most scripts accept environment overrides:

| Variable / constant | Meaning |
|---|---|
| `SAEN_DATA`, `SAEN_OUT` (`enrich_saen.py`) | Mitrasaṃgraha TSV directory (`{train/<license>,val,test}/*.tsv`), output directory |
| `V2`, `SA_SEG`, `ZH_SEG` (`prep_v3_substrate.py`) | MITRA-parallel v3, DharmaNexus Sanskrit and Chinese segment files |
| `GEMINI_API_KEY` | teacher model key (all teacher calls) |
| `DICTSPAN_CKPT`, `GPU_UTIL`, `MAX_NUM_SEQS` | student checkpoint (vLLM layout) and inference settings for mining |
| `SRC`, `OUT` (`build_sft_dataset_saen.py`) | seed directory, SFT dataset output |
| `BASE`, `OUTDIR`, `SPANS_IN`, `SPANS_OUT` (`5-build/`) | mined rows in, lexicon files out |
| `ONLY_PAIRS`, `REC_skt_*` (`6-postprocess/`) | restrict to one pair; corpus file with the token analyses of each pair |
| `EVAL_MINED_DIR`, `CORPUS_SA_BO`, `CORPUS_SA_ZH`, `HIRAKAWA_FINAL`, `HIRA`, `LAYER` (`eval/`) | lexicon set, sentence-pair corpora, Hirakawa file and layer |
| `OUT_DIR` (`eval/common.py`), `OUT` (`eval/novel.py`), `N` (`eval/run_set.sh`) | evaluation output directory |
| `BASELINE_WORK`, `CORPUS_*` (`baselines/stat_baselines.py`) | working directory and corpora of the statistical aligners |

**External dependencies (not included).**
- ByT5-Sanskrit analyzer: `analyze_worker.py` from the DharmaNexus Sanskrit processing pipeline
  (`~/code/dharmanexus-sanskrit-processing`, imported by `enrich_saen.py`), with its CTranslate2 model.
- Training framework `lme` (imported by `3-student/run_dictmining.py` and `dictionary_mining.py`; data processors,
  model mixins and the SFT pipeline). The experiment definitions used for the paper are in `dictionary_mining.py`.
- Python packages: `vllm`, `transformers`, `google-genai`, `pyewts`, `rapidfuzz`, `pyahocorasick`, `bitsandbytes`,
  `indic_transliteration` (evaluation).
- Statistical aligners for the baselines: fast_align, MGIZA++, eflomal.

**Data not included.** Corpora and reference dictionaries are not redistributed here:
- Mitrasaṃgraha (`github.com/dharmamitra/mitrasamgraha-dataset`), MITRA-parallel v3, DharmaNexus segment files.
- Reference dictionaries for the evaluation: Monier-Williams 1899 and Apte 1890 (Cologne Digital Sanskrit
  Dictionaries, `sanskrit-lexicon.uni-koeln.de`). Monier-Williams is read from the Cologne source file `mw.txt`
  (`github.com/sanskrit-lexicon/csl-orig`, `v02/mw/mw.txt`; constant or environment variable `MW99` in
  `eval/eval_sa_en.py`), Apte from a SQLite file `dict.sqlite` (constant `DB`), Negi's Tibetan–Sanskrit dictionary (Steinert's collection,
  `github.com/christiansteinert/tibetan-dictionary`; constant `NEGI` in `eval/negi.py`), Hirakawa's Buddhist
  Chinese–Sanskrit Dictionary (`HIRA`; not freely available). The Hirakawa layer of the web interface
  (`5-build/build_hirakawa_layer.py`) is built from the same file and is not part of the release.
- `6-postprocess/postprocess_final.py` needs `dnx_counts.pkl` (lemma/form counts over DharmaNexus Sanskrit,
  produced by `dnx_counts.py`) and `ref_allowlist_mw_apte.txt` (the headwords of Monier-Williams 1872 and
  Apte, one per line, derived from the Cologne data; this build filter used the 1872 edition). `5-build/tib_norm.py` expects `tib_verb_map.json` under
  `refresh2026/` relative to the script.
- The DCS corpus (`novel.py`, constant `DCS`) for the lemma-coverage evaluation.

## Build chain

Steps in the order they were run. Paper sections in brackets.

**1. Analysis of the Sanskrit** (Sanskrit analysis)
1. `1-analysis/enrich_saen.py --shard i --num-shards n`: Mitrasaṃgraha TSVs → `sa-en/enriched/*.ndjson`.
   Sanskrit is analyzed with ByT5-Sanskrit (`full_analysis`: unsandhied words, lemmas, tags) in chunks of at
   most 150 characters. Row ids are `<split>/<license>/<file>.tsv:<line>`.
2. `1-analysis/prep_v3_substrate.py sa-bo` / `sa-zh`: MITRA-parallel v3 → `<dir>/substrate.ndjson`. The Sanskrit
   analysis is taken from the DharmaNexus Sanskrit segments, the Chinese word segmentation from the DharmaNexus
   Chinese segments, the Tibetan as released.

**2. Teacher labeling** (Teacher labeling; held-out evaluation of the student)
1. `2-teacher-labels/generate_seeds.py` (sa-bo, sa-zh, zh-bo, en-bo) and `generate_seeds_saen.py` (sa-en): teacher
   calls with the labeling protocol; outputs kept only if verbatim and grounded →
   `<dir>/seeds.ndjson` + `gemini_raw.ndjson`. Shared prompt, parsing and reconstruction code: `mine_spans.py`.
2. `2-teacher-labels/seeds_to_unicode.py`: Tibetan seeds from Wylie to Unicode → `seeds_unicode.ndjson`.
3. `2-teacher-labels/generate_heldout.py`: about 500 further teacher-labeled pairs per Sanskrit direction,
   disjoint from the seeds (the held-out set). `heldout_rerun.py`: second teacher run on a subset
   (teacher self-agreement). `eval_heldout.py`: span and link scores of a student checkpoint on the held-out set.
   `heldout_disagreements.py`: classification of student–teacher disagreements.

**3. Student training** (Student)
1. `3-student/build_sft_dataset_saen.py`: seeds of five directions (sa-en duplicated; 50 validation and 20 test
   examples held out per direction) → `train/val/test.jsonl` (9,366 training examples).
   (`build_sft_dataset.py`: the four-direction predecessor with the same minimal prompt.)
2. `3-student/run_qwen_dictionaries_saen.sh` → `run_dictmining.py` with `SpanProjectionQwenDictSaenExperiment`
   (MITRA Qwen3.5-9B stage 2, full fine-tuning, 8-bit AdamW, learning rate 2e-5, effective batch 64, one epoch).
   `run_9b_3ep.sh`: the same for three epochs (`SpanProjectionQwenDictSaen3epExperiment`).
   `run_ablation.sh`, `run_2b_3ep.sh`: the other students of the baseline table.
3. `3-student/convert_to_mm.py`: checkpoint → the multimodal layout that vLLM loads.

**4. Mining** (Inference)
1. Sanskrit–English: `4-mining/mine_direction.py --direction sa-en` with `DICTSPAN_CKPT` = the one-epoch 9B student
   → `sa-en/mined.ndjson` (one row per sentence pair, with output, `conf`, `ung_frac`, `coverage`, `verbatim`,
   `numbering_bad`, `aligns`).
2. Sanskrit–Tibetan and Sanskrit–Chinese: `4-mining/run_v4.sh`, `run_v4b.sh` with the three-epoch 9B student.
   `delta.py`/`delta2.py` reuse rows of an earlier run whose model input and segment numbers are identical
   (greedy decoding, same checkpoint), `mine_v4.py` mines the rest, `final_combine.py` writes one row per pair of
   the substrate. `run_sazh.sh`: the same build for Sanskrit–Chinese. Mining the whole substrate with
   `mine_v4.py` gives the same rows.

**5. Filters and lexicon building** (Filters; From Alignments to Lexicon Entries)
1. Sanskrit–English: `5-build/run_saen_build.sh` → `build_saen_spans.py` (gate: verbatim, numbering, `conf` ≥ −0.45,
   `ung_frac` ≤ 0.25; compound phrase spans) → `build_saen_bidict.py` → `clean_rank_noise.py` (clause-boundary and
   PMI ≥ 1.0 filters) → `kill_src_stopwords.py`.
2. Sanskrit–Tibetan / Sanskrit–Chinese: `5-build/prepare_finals.py sa-bo sa-zh` (mined rows → `hybrid/final.ndjson`,
   all verbatim rows) → `run_direction.sh sa-bo|sa-zh` (`run_spans_parallel.py` → `build_sa{bo,zh}_spans.py` →
   `build_sa{bo,zh}_bidict.py` → `clean_rank_noise.py` → `kill_stopword_glosses.py` (Tibetan function syllables)
   → `kill_src_stopwords.py`) → `dedup_ex_all.py`.

**6. Postprocessing** (Lexicon entries; compound keys; reconstructed texts)
1. `6-postprocess/postprocess_final.py SRC DST`: normalizes anusvāra in keys; drops headwords that look like e-text
   errors (rare, one edit from a frequent lemma, not in Monier-Williams/Apte) → `flagged_<pair>.tsv`.
2. `full_attestations.py SRC DST`: rebuilds every sense's attestation list from the spans (one per occurrence).
3. `drop_subcompounds.py SRC DST`: keeps only full compounds.
4. `extend_compounds.py SRC DST`: two formation rules (formatives -tva/-tā/-mātra; Chinese 、 enumerations).
5. `rekey_compounds.py SRC DST`: compound entries keyed on hyphenated members with the final member as stem;
   case forms under `forms` (uses `dcs_verbs.txt`).
6. `prune_texts.py SRC DST reconstructed_texts.tsv`: removes all attestations from Sanskrit texts that are wholly or
   partly reconstructed from Tibetan or Chinese; `--corpus` prunes the sentence-pair corpora the same way.
7. `fix_lemma_encoding.py DIR`: normalizes lemma encoding in keys and lemmas.

**7. Assembly**: `7-assembly/derive_v4zh.sh`, `derive_v4bo.sh` run steps 5–6 per pair and assemble the set.
`7-assembly/release_build_assets.sh` and `release_derive.py` produce the release files from the set: the CC BY-SA
subset of Sanskrit–English (attestations with `license: cc-by-sa`; counts and `forms` recomputed), the alignment
files, and the zstd-compressed assets with `SHA256SUMS.txt`.

**Chinese–Tibetan (release v1.1, not in the paper).** `4-mining/run_zhbo.sh` runs all stages:
1. `1-analysis/prep_v3_zhbo_substrate.py todo`: the distinct Chinese strings of MITRA-parallel v3 bo-zh without a
   segmentation from the earlier zh-bo substrate. `kokuyaku_translate_hf.py` renders them into kundoku Japanese
   with the kokuyaku model (Qwen3.5-0.8B, checkpoint path in the script, not released) and
   `segment_zh_from_japanese.py` keeps the runs of Chinese characters as words. `prep_v3_zhbo_substrate.py write`
   → `zh-bo/substrate.ndjson`.
2. `4-mining/mine_v4.py --direction zh-bo-v3` with the three-epoch 9B student.
3. `5-build/prepare_finals.py zh-bo` → `run_direction.sh zh-bo` (`build_zhbo_spans.py`, `build_zhbo_bidict.py`,
   then the filters of step 5.2). No `dedup_ex_all.py` (every attestation is kept) and none of the Sanskrit postprocessing of step 6.
4. `7-assembly/release_v1.1_build_assets.sh`: the v1.0 assets plus the four Chinese–Tibetan files.

## Baselines

`baselines/stat_baselines.py`: fast_align, MGIZA++ (grow-diag-final-and) and eflomal on the same sentence pairs;
link F1 against the held-out teacher labels, lexica built from their links. Runners: `run_norecon.sh` (paper),
`run_v4bo.sh`, `run_v4zh.sh`; tables: `summarize_baselines.py`.

## Evaluation

`eval/run_set.sh` runs everything on the released set (CPU): `run_all.sh` (`eval_sa_en.py` vs Monier-Williams 1899/Apte,
`eval_sa_bo.py` vs Negi, `eval_sa_zh.py` vs Hirakawa, `make_report.py`), the Negi variant restricted to attested
equivalents, `novel.py`, `lemma_coverage.py`, `attestation_composition.py`, `hirakawa_compare.py`,
`hirakawa_gap/overview.py`, `paper_numbers.py`, `negi_recover.py`. Further: `hapax_sample.py` (review sample of
single-attestation entries), `saen_genre.py`, `negi_sources.py`, `negi_bands.py`, `negi_domains.py`,
`hirakawa_texts.py`, `hirakawa_gap/layer_ctl.py` (controls), `baseline_numbers.py`. Gold-alignment scoring:
see `gold_eval.py` and `../data/gold/`.

## License

The code is released under the MIT License.
