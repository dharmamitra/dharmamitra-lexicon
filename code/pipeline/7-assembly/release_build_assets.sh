#!/bin/bash
# Build the release assets of MITRA-lexicon v1 (2026-10-19). Inputs are read-only.
set -euo pipefail
R=/mnt2/code/mitra-dictionaries/release-lexicon-v1-20261019
SET=/mnt2/code/mitra-dictionaries/paper-v4-20261017-norecon
W=$R/work; A=$R/assets
python3 $R/scripts/derive.py $W bidict > $W/derive.bidict.log 2>&1 &
python3 $R/scripts/derive.py $W spans  > $W/derive.spans.log 2>&1 &
python3 $R/scripts/derive.py $W mined  > $W/derive.mined.log 2>&1 &
for p in $(jobs -p); do wait $p; done
z() { zstd -q -19 --long=27 -T24 -f "$1" -o "$A/$2.zst"; }
# complete versions (sa-en: CC BY-NC-ND 4.0)
z $SET/skt_en_bidict.sa2en.ndjson mitra-lexicon-sa-en.sa2en.ndjson &
z $SET/skt_en_bidict.en2sa.ndjson mitra-lexicon-sa-en.en2sa.ndjson &
z $SET/skt_tib_bidict.sa2bo.ndjson mitra-lexicon-sa-bo.sa2bo.ndjson &
z $SET/skt_tib_bidict.bo2sa.ndjson mitra-lexicon-sa-bo.bo2sa.ndjson &
z $SET/skt_zh_bidict.sa2zh.ndjson mitra-lexicon-sa-zh.sa2zh.ndjson &
z $SET/skt_zh_bidict.zh2sa.ndjson mitra-lexicon-sa-zh.zh2sa.ndjson &
for p in $(jobs -p); do wait $p; done
z $SET/skt_en_spans.ndjson  mitra-lexicon-sa-en.spans.ndjson &
z $SET/skt_tib_spans.ndjson mitra-lexicon-sa-bo.spans.ndjson &
z $SET/skt_zh_spans.ndjson  mitra-lexicon-sa-zh.spans.ndjson &
z $W/sa-en.alignments.ndjson mitra-lexicon-sa-en.alignments.ndjson &
z $W/sa-bo.alignments.ndjson mitra-lexicon-sa-bo.alignments.ndjson &
for p in $(jobs -p); do wait $p; done
z $W/sa-zh.alignments.ndjson mitra-lexicon-sa-zh.alignments.ndjson &
# CC BY-SA 4.0 subset of sa-en
z $W/sa-en.by-sa.sa2en.ndjson mitra-lexicon-sa-en-by-sa.sa2en.ndjson &
z $W/sa-en.by-sa.en2sa.ndjson mitra-lexicon-sa-en-by-sa.en2sa.ndjson &
z $W/sa-en.by-sa.spans.ndjson mitra-lexicon-sa-en-by-sa.spans.ndjson &
z $W/sa-en.by-sa.alignments.ndjson mitra-lexicon-sa-en-by-sa.alignments.ndjson &
for p in $(jobs -p); do wait $p; done
cd $A && sha256sum *.zst > SHA256SUMS.txt && ls -la
touch $R/BUILD_DONE
