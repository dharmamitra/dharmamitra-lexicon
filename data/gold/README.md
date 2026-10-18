# Sanskrit–English expert gold alignments

Three Sanskritists, A, B and C, independently aligned the same 100 Sanskrit–English sentence pairs of
Mitrasaṃgraha. These files are the data of the gold evaluation and the inter-annotator agreement study of the
paper (Table gold and the expert-gold column of Table baselines), together with the scripts that reproduce
those numbers (`code/eval/`).

## Files

| File | Content |
|---|---|
| `sa-en_gold_alignments.jsonl` | 300 annotations (100 sentences × annotators A, B, C), one JSON object per sentence and annotator |
| `items.json` | the 100 sentences as shown to the annotators: Sanskrit analysis, English word tokens, and the span `pred` from which the lexicon was built |
| `baseline_links.jsonl` | links of fast_align, MGIZA++ and eflomal on the same 100 sentences (from the runs of Table baselines) |
| `genre.json` | genre and period of each sentence (Mitrasaṃgraha catalog) |
| `students/*.ndjson` | outputs of the six students of Table baselines on the 100 sentences |

### `sa-en_gold_alignments.jsonl`

- `id`: Mitrasaṃgraha identifier, `<split>[/<license group>]/<file>:<line>`
- `annotator`: `A`, `B` or `C`
- `sanskrit`, `english`: the sentence pair
- `tokens`: the Sanskrit analysis (ByT5-Sanskrit), `i` (1-based), `unsandhied`, `lemma`
- `units`: one unit per analysed Sanskrit word (`kind` `token`) and per maximal compound (`kind` `stack`), with
  - `indices`: the Sanskrit tokens of the unit
  - `pred`: the span of the lexicon, as an English word range `[start_word, end_word]` (indices into `en_tokens` of `items.json`), or `null`
  - `gold`: the annotator's decision, `status` `aligned` or `none`; if aligned, `start_word`/`end_word` (word range), `start`/`end` (character offsets in `english`) and `text`
  - `change`: how the decision relates to `pred` (`unchanged`, `unchanged_none`, `added`, `removed`, `moved`)
- `comment`: free-text remark of the annotator (in 24 annotations)

The annotators corrected a preselected span (`pred`) rather than aligning from scratch. The guidelines leave function
words and pronouns unaligned.

### `students/`

Each file is the output of `mine_direction.py --direction sa-en` for the 100 sentences, with `output` (the English
sentence with numbered tags) and the quality signals of the alignment output. Names follow Table baselines:

| File | Student |
|---|---|
| `qwen35-2b-mitra.ndjson` | MITRA Qwen3.5-2B |
| `qwen35-2b-mitra-3ep.ndjson` | MITRA Qwen3.5-2B, three epochs |
| `qwen35-9b-vanilla.ndjson` | Qwen3.5-9B without Buddhist pretraining |
| `gemma2-9b-mitra.ndjson` | Gemma-2-MITRA 9B |
| `qwen35-9b-mitra.ndjson` | MITRA Qwen3.5-9B, one epoch (the checkpoint that aligned the Sanskrit–English corpus) |
| `qwen35-9b-mitra-3ep.ndjson` | MITRA Qwen3.5-9B, three epochs |

## Reproducing the numbers

```
python3 code/eval/gold_eval.py gold-eval.json    # agreement, alpha, model P/R/F1 per annotator, consensus, majority, links
python3 code/eval/score_students.py              # expert-gold link F1 of the students (mean over A, B, C)
```

`gold_eval.py` reproduces every value of the evaluation file used for the paper (853 values, identical up to the
annotator letters). `score_students.py` prints, per student, the link F1 against each annotator and the mean, which
is the last column of Table baselines:

| Student | A | B | C | Mean |
|---|---|---|---|---|
| MITRA Qwen3.5-2B | 76.6 | 70.8 | 77.2 | 74.9 |
| MITRA Qwen3.5-2B, three epochs | 77.1 | 70.7 | 78.6 | 75.5 |
| Qwen3.5-9B without Buddhist pretraining | 81.9 | 74.6 | 83.2 | 79.9 |
| Gemma-2-MITRA 9B | 80.5 | 75.3 | 82.5 | 79.4 |
| MITRA Qwen3.5-9B, one epoch | 86.0 | 79.2 | 88.7 | 84.6 |
| MITRA Qwen3.5-9B, three epochs | 85.6 | 78.9 | 87.2 | 83.9 |

Outputs that do not reproduce the English verbatim (0–7 of 100 per student) are located span by span through text
search, as in the held-out link F1 of Table baselines; `--no-fallback` scores them as empty instead.

## Sources of the 100 sentences

| Text | Translator | Mitrasaṃgraha split | License group | Sentences |
|---|---|---|---|---|
| Abhidharmakośakārikā | Pruden | train | cc-by-sa | 1 |
| Arthaviniścayasūtra | Ānandajoti | train | cc-by-sa | 8 |
| Arthaśāstra | Shama Sastry | train | cc-by-sa | 9 |
| Bhāmatī | Sastri + Raja | train | cc-by-sa | 1 |
| Buddhacarita | E. B. Cowell | test | none stated | 2 |
| Buddhacarita | E. B. Cowell | val | none stated | 1 |
| Chāndogya-Upaniṣad (Śaṅkarabhāṣya) | Jha | train | cc-by-sa | 6 |
| Garga Saṃhita | Danavir Goswami | train | cc-by-sa | 2 |
| Gautama-Dharmasūtra | Bühler | train | cc-by-sa | 2 |
| Harivaṃśa | Dutt | train | cc-by-sa | 5 |
| Itihāsa extracts (Rāmāyaṇa, Mahābhārata) | M. N. Dutt; K. M. Ganguli | test | none stated | 3 |
| Itihāsa extracts (Rāmāyaṇa, Mahābhārata) | M. N. Dutt; K. M. Ganguli | val | none stated | 1 |
| Jātakamālā | Speyer | train | cc-by-sa | 7 |
| Jīvanmuktiviveka | H. Ramachandra Swamy | train | cc-by-sa | 4 |
| Kathāsaritsāgara | C. H. Tawney | test | none stated | 2 |
| Kathāsaritsāgara | C. H. Tawney | val | none stated | 2 |
| Kāvyādarśa | S K Belvalkar | train | cc-by-sa | 1 |
| Kṛṣṇa-Yajurveda | Keith | train | cc-by-sa | 2 |
| Madhyāntavibhāgabhāṣya | Anacker | train | cc-by-sa | 1 |
| Madhyāntavibhāgaṭīkā | Stanley | train | cc-by-sa | 1 |
| Madhyāntavibhāgaṭīkā 1 | Shcherbatskoy | train | cc-by-sa | 1 |
| Manusmṛti | Bühler | test | none stated | 1 |
| Manusmṛti | Bühler | train | cc-by-sa | 1 |
| Mīmāṃsāsūtra Tantravarttika | Jha | train | cc-by-sa | 2 |
| Nyāyamañjari | Bhattacharyya | train | cc-by-sa | 4 |
| Sarva-darśana-saṃgraha | Cowell | train | cc-by-sa | 1 |
| Skanda-Purāṇa | Tagare | val | none stated | 1 |
| Sāṃkhyakārikā | Horace Hayman Wilson | train | cc-by-sa | 1 |
| Tantrāloka | Singh | train | cc-by-nc-nd | 6 |
| Triṃśikāvijñaptibhāṣya | Jacobi | train | cc-by-sa | 2 |
| Vimalakīrtinirdeśa | Robert A. F. Thurman | val | none stated | 3 |
| Viṣṇu-Purāṇa | Horace Hayman Wilson | train | cc-by-sa | 5 |
| Viṣṇusmṛti | Jolly | train | cc-by-sa | 2 |
| Yogavasiṣṭha | Mitra | test | none stated | 1 |
| Yogavasiṣṭha | Mitra | val | none stated | 2 |
| Śatapatha-Brāhmaṇa | Eggeling | test | none stated | 1 |
| Śatapatha-Brāhmaṇa | Eggeling | train | cc-by-sa | 3 |
| Śukla-Yajurveda | Griffiths | train | cc-by-sa | 2 |

## License

The annotations reproduce the sentence pairs, so they are released under CC BY-NC-ND 4.0, like the complete
Sanskrit–English lexicon: six sentences come from a translation under CC BY-NC-ND, and the val and test splits of
Mitrasaṃgraha carry no license tag and include a translation still under copyright (Thurman's Vimalakīrtinirdeśa).
