# Teacher labels, training data and held-out evaluation

These are the files behind Sections "Teacher Labeling" and "Student Model" of the paper and the link-F1 columns of its Table "baselines".

## seeds/ and teacher-raw/

| Direction | Teacher calls (`teacher-raw/`) | Seeds kept (`seeds/`) | Teacher |
|---|---:|---:|---|
| sa-en | 1,200 | 1,000 | gemini-3.8-flash, highest reasoning level (September 2026) |
| sa-bo | 3,542 | 2,000 | gemini-3.6-flash, reasoning budget 16,384 (July 2026) |
| sa-zh | 2,518 | 2,000 | gemini-3.6-flash, reasoning budget 16,384 (July 2026) |
| zh-bo (auxiliary) | 2,522 | 2,000 | gemini-3.6-flash |
| en-bo (auxiliary) | 2,469 | 1,786 | gemini-3.6-flash |

`seeds/<dir>.ndjson`: one kept seed per line, `{input_text, target_text, lang, ...}`. `input_text` is the full
teacher prompt (labeling protocol, analyzed source with numbered tags, target sentence), `target_text` the teacher
output: the target sentence reproduced verbatim with numbered tags around the span that renders each source word.
sa-en seeds also carry `id`, `file` and `license` of the Mitrasaṃgraha line.

`teacher-raw/<dir>.ndjson`: every teacher call, including those that were discarded, with the raw output, the
grounded spans, and whether the output reproduced the target verbatim (`faithful`).
Together, the five directions record 12,251 teacher calls.

## sft/

The training mixture of the student: `train.jsonl` (9,366 examples), `val.jsonl` (250, i.e. 50 per direction) and
`test.jsonl` (100, i.e. 20 per direction). The fields are `{input_text, target_text, lang}`. `input_text` is the
one-line instruction followed by the data block. The sa-en seeds appear twice in `train.jsonl`.
Built by `code/pipeline/build_sft_dataset_saen.py` from `seeds/`.

## heldout/

- `test.jsonl`: 1,489 further sentence pairs labeled by the teacher under the same protocol (sa-bo 493, sa-zh 496,
  sa-en 500). They are disjoint from the training seeds. This is the reference for the link F1 "vs. teacher labels"
  in Table "baselines".
- `teacher-raw/`: the raw teacher calls that produced them (900 / 600 / 600).
- `teacher-rerun/`: a second teacher run with identical settings on 155 of these pairs (sa-bo 47, sa-zh 65,
  sa-en 43). This is the self-agreement ceiling of Table "ablation".
- `student-predictions/`: the outputs of each student of Table "baselines" on `test.jsonl`:

| File | Student |
|---|---|
| `qwen35-2b-mitra.jsonl` | Qwen3.5-2B, MITRA pretraining |
| `qwen35-2b-mitra-3ep.jsonl` | same, 3 epochs |
| `qwen35-9b-vanilla.jsonl` | Qwen3.5-9B, no Buddhist pretraining |
| `gemma2-9b-mitra.jsonl` | Gemma-2-MITRA 9B |
| `qwen35-9b-mitra.jsonl` | MITRA Qwen3.5-9B-Base stage 2, 1 epoch (aligned the Sanskrit–English corpus) |
| `qwen35-9b-mitra-3ep.jsonl` | same, 3 epochs (aligned the Sanskrit–Tibetan and Sanskrit–Chinese corpora; published as `buddhist-nlp/qwen35-mitra-lexicon-annotation`) |

Scoring: `code/pipeline/eval_heldout.py --score-only <predictions>` (span level) and `code/baselines/` (link level).

## License

CC BY-NC-ND 4.0. The sa-en seeds and held-out pairs include lines from Mitrasaṃgraha texts under CC BY-NC-ND, and
the en-bo seeds come from 84000 translations, which are published under CC BY-NC-ND.
