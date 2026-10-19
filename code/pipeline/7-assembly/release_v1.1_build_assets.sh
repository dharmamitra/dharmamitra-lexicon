#!/bin/bash
# MITRA-lexicon v1.1 (2026-10-20): v1.0 assets (hardlinked, unchanged) + Tibetan-Chinese. Inputs are read-only.
set -euo pipefail
R=/mnt2/code/mitra-dictionaries/release-lexicon-v1.1-20261020
V1=/mnt2/code/mitra-dictionaries/release-lexicon-v1-20261019/assets
Q=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4
A=$R/assets
for f in $V1/*.zst; do ln -f "$f" $A/; done
z() { zstd -q -19 --long=27 -T24 -f "$1" -o "$A/$2.zst"; }
z $Q/final-output/bo_zh_bidict.zh2bo.ndjson mitra-lexicon-zh-bo.zh2bo.ndjson &
z $Q/final-output/bo_zh_bidict.bo2zh.ndjson mitra-lexicon-zh-bo.bo2zh.ndjson &
z $Q/final-output/bo_zh_spans.ndjson mitra-lexicon-zh-bo.spans.ndjson &
cat $Q/zh-bo/mined.ndjson.part0 $Q/zh-bo/mined.ndjson.part1 | zstd -q -19 --long=27 -T24 -f -o $A/mitra-lexicon-zh-bo.alignments.ndjson.zst &
wait
cd $A && sha256sum *.zst > SHA256SUMS.txt && ls -la
touch $R/BUILD_DONE
