# MITRA-lexicon

Lexica of attested translations from Sanskrit into English, Tibetan and Chinese, with the data, models and code
needed to reproduce them and their evaluation.

Every sense in the lexica is a count over a parallel corpus, and every count can be traced to its sentence pairs:
each attestation carries the identifier of the sentence, the translation, and the character offsets of the
equivalent within it.

- Web interface: https://lexicon.dharmamitra.org
- Word alignment model (student): https://huggingface.co/buddhist-nlp/qwen35-mitra-lexicon-annotation
- Paper: *MITRA-lexicon: Lexica of attested translations from Sanskrit into English, Tibetan and Chinese* (ISCLS 2027, under review)

## Lexica

| Pair | Direction | Headwords | Senses | Attestations | Source corpus |
|---|---|---:|---:|---:|---|
| Sanskrit–English | sa→en | 418,927 | 950,845 | 3,293,993 | Mitrasaṃgraha |
| | en→sa | 396,289 | | | |
| Sanskrit–Tibetan | sa→bo | 529,599 | 895,229 | 4,665,998 | MITRA-parallel v3 |
| | bo→sa | 445,176 | | | |
| Sanskrit–Chinese | sa→zh | 223,081 | 422,074 | 2,447,898 | MITRA-parallel v3 |
| | zh→sa | 171,780 | | | |
| **Total** | | **2,184,852** | **2,268,148** | **10,407,889** | |

Senses and attestations are properties of a pair and identical in both of its directions.
Sanskrit texts that are wholly or partly reconstructed from a Tibetan or Chinese translation are excluded
(`data/excluded_texts.tsv`).

## Download

The large files are attached to the [GitHub release v1.0](../../releases/tag/v1.0) as zstd-compressed
line-delimited JSON. They decompress with `zstd -d --long=27 FILE.zst`, and `SHA256SUMS.txt` lists their checksums.

| File | Content | Lines | License |
|---|---|---:|---|
| `mitra-lexicon-sa-en.sa2en.ndjson.zst`, `….en2sa…` | Sanskrit–English lexicon, complete (the version described in the paper) | 418,927 / 396,289 | CC BY-NC-ND 4.0 |
| `mitra-lexicon-sa-en-by-sa.sa2en.ndjson.zst`, `….en2sa…` | Sanskrit–English lexicon, attestations from CC BY-SA texts only | 384,073 / 364,517 | CC BY-SA 4.0 |
| `mitra-lexicon-sa-bo.sa2bo.ndjson.zst`, `….bo2sa…` | Sanskrit–Tibetan lexicon | 529,599 / 445,176 | CC BY-SA 4.0 |
| `mitra-lexicon-sa-zh.sa2zh.ndjson.zst`, `….zh2sa…` | Sanskrit–Chinese lexicon | 223,081 / 171,780 | CC BY-SA 4.0 |
| `mitra-lexicon-<pair>.spans.ndjson.zst` | span files from which the lexica were built: one row per aligned word or compound | sa-en 3,315,925; sa-en-by-sa 3,059,007; sa-bo 4,724,699; sa-zh 2,475,228 | as the lexicon of the pair |
| `mitra-lexicon-<pair>.alignments.ndjson.zst` | aligner output for every sentence pair, with its quality signals | sa-en 395,528; sa-en-by-sa 365,854; sa-bo 566,068; sa-zh 340,691 | as the lexicon of the pair |

### Lexicon entries

```json
{"sa": "kṛpā",
 "senses": [
   {"en": "mercy", "count": 151,
    "ex": [{"id": "train/cc-by-sa/GE07hv_conpu.tsv:1716",
            "target": "Going out a-hunting the king Shāntanu, out of mercy, took them up.",
            "start": 46, "end": 51,
            "src": "kṛpayā tac ca jagrāha śaṃtanur mṛgayāṃ gataḥ /",
            "license": "cc-by-sa", "indices": [1]}, ...]},
   {"en": "compassion", "count": 110, "ex": [...]},
   {"en": "pity", "count": 86, "ex": [...]}, ...],
 "n": 604, "n_senses": 38}
```

- `sa` (or `en`, `bo`, `zh` in the reverse lexica): the headword. Sanskrit headwords are DCS lemmas, compounds
  keyed on their hyphenated members with the final member reduced to its stem (`kuśala-mūla`), or sandhied
  surface forms of compounds as they stand in the text.
- `n`: the number of attestations of the headword. `n_senses`: the number of senses.
- `senses`: ranked by `count`, the number of attestations of the sense. `ex` lists all of them.
- `ex[].id`: the sentence identifier. For Sanskrit–English, it gives the Mitrasaṃgraha split, the license group,
  the text file and the line. For Sanskrit–Tibetan and Sanskrit–Chinese, it is the identifier of the
  MITRA-parallel sentence pair (Sanskrit segment and Tibetan or Chinese segment), and `tgt_segnr` lists the
  target segments.
- `ex[].target`, `start`, `end`: the translation and the character offsets of the equivalent within it.
- `ex[].indices`: the 1-based positions of the headword in the analyzed Sanskrit sentence.
- `ex[].src`: the Sanskrit sentence (Sanskrit–English only).
- `ex[].low_coverage`: present and true when the aligner tagged few words of the sentence.
- `forms` (compound entries): the case forms that occur, with their number of attestations.

### Span rows

Each row is one aligned Sanskrit word (`type: token`) or compound (`type: phrase`), with `head`, `lemma`,
`sanskrit` (surface form), the equivalent (`en`, `bo` or `zh`), `start` / `end` in `target`, the sentence
identifier, the Sanskrit sentence (`src_sentence`) and the translation (`target`).

### Alignment rows

Each row is one sentence pair as aligned by the student. It contains `id`, `target`, the Sanskrit sentence
(`src_original`), the analyzer output (`toks`: surface form and lemma per word), the tagged model `output`,
the extracted spans (`aligns`: tag number → span) and the quality signals that the filters of the paper use:
`conf` (mean token log-probability), `ung_frac` (share of ungrounded spans), `coverage`, `verse`,
`numbering_bad`, `verbatim` and `flagged`.

## Repository contents

| Path | Content | License |
|---|---|---|
| `data/gold/` | 100 Sanskrit–English sentences aligned independently by three Sanskritists (A–C), the scorer and the outputs of every student | CC BY-NC-ND 4.0 |
| `data/teacher/` | teacher-labeled seeds, raw teacher calls, the 9,366-example training mixture, 1,489 held-out teacher labels, a second teacher run, and the predictions of every student | CC BY-NC-ND 4.0 |
| `data/excluded_texts.tsv` | Sanskrit texts excluded as reconstructions | CC BY-SA 4.0 |
| `code/` | annotation, teacher labeling, student training, mining, lexicon building, baselines, evaluation | MIT |

`data/gold/README.md`, `data/teacher/README.md` and `code/README.md` describe each part, and how to reproduce
the tables of the paper.

## Licenses

Every attestation reproduces its sentence pair, so the lexica inherit the licenses of their source texts.

- **Sanskrit–English** comes in two versions.
  - `sa-en-by-sa` contains only attestations from Mitrasaṃgraha texts under CC BY-SA (92.3% of the
    attestations) and is released under CC BY-SA 4.0.
  - The complete lexicon also contains attestations from texts under CC BY-NC-ND (4.3%) and from the
    validation and test splits of Mitrasaṃgraha (3.4%), for which no license is stated. It is released under
    CC BY-NC-ND 4.0.
- **Sanskrit–Tibetan and Sanskrit–Chinese** are released under CC BY-SA 4.0.
- **Gold and teacher data** are released under CC BY-NC-ND 4.0, since they contain lines from texts under
  that license.
- **Code** is released under MIT.

See `LICENSE.md`. If you use the data in a public-facing application, please acknowledge the Dharmamitra
project. We would also appreciate hearing from you at dharmamitra-project@gmail.com.

## Earlier release

Until October 2026 this repository held an earlier, unrelated release of span data (sa-bo, sa-zh, zh-bo; June
2026). It is preserved in the branch [`legacy-2026-06`](../../tree/legacy-2026-06) and the tag `legacy-2026-06`.
