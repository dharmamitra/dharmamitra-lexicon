#!/usr/bin/env bash
# 2026-10-17: v4 sa-bo (qwen-dictionaries-v4, MITRA-parallel v3, no integrity gate) through the paper chain:
# postprocess_final -> full_attestations -> drop_subcompounds -> extend_compounds -> rekey_compounds (ONLY_PAIRS=skt_tib),
# then assembled with en (paper-final-output-20261017-stemext) and zh (paper-candidate-v4zh-20261017-stemext)
# into paper-v4-20261017-stemext. Files that later steps rewrite are COPIED, never hard-linked (flagged_*.tsv lesson).
set -euo pipefail
M=/mnt2/code/mitra-dictionaries; W=$M/v4bo-20261017; V4=$M/qwen-dictionaries-v4
OUT=$M/paper-v4-20261017-stemext
export ONLY_PAIRS=skt_tib REC_skt_tib=$V4/sa-bo/hybrid/final.ndjson
mkdir -p $W/v4out $W/pp $W/full $W/fullcpd $W/extcpd $W/stemext
rm -f $W/v4out/*
for f in skt_tib_bidict.sa2bo.ndjson skt_tib_bidict.bo2sa.ndjson skt_tib_spans.ndjson; do ln -f $V4/final-output/$f $W/v4out/; done
cd $M/postprocess
echo "### $(date +%T) postprocess"; python3 postprocess_final.py $W/v4out $W/pp
echo "### $(date +%T) full attestations"; python3 full_attestations.py $W/pp $W/full
[ -f $W/full/skt_tib_spans.ndjson ] || ln -f $W/pp/skt_tib_spans.ndjson $W/full/
echo "### $(date +%T) drop subcompounds"; python3 drop_subcompounds.py $W/full $W/fullcpd
echo "### $(date +%T) extend"; python3 extend_compounds.py $W/fullcpd $W/extcpd > $W/extend.log
echo "### $(date +%T) rekey"; python3 rekey_compounds.py $W/extcpd $W/stemext > $W/rekey.log
echo "### $(date +%T) assemble"
mkdir -p $OUT
EN=$M/paper-final-output-20261017-stemext; ZH=$M/paper-candidate-v4zh-20261017-stemext
for f in skt_en_bidict.sa2en.ndjson skt_en_bidict.en2sa.ndjson skt_en_spans.ndjson; do ln -f $EN/$f $OUT/$f; done
for f in skt_zh_bidict.sa2zh.ndjson skt_zh_bidict.zh2sa.ndjson skt_zh_spans.ndjson skt_zh_hirakawa_layer.ndjson; do ln -f $ZH/$f $OUT/$f; done
for f in skt_tib_bidict.sa2bo.ndjson skt_tib_bidict.bo2sa.ndjson skt_tib_spans.ndjson; do ln -f $W/stemext/$f $OUT/$f; done
cp $EN/flagged_skt_en.tsv $OUT/; cp $ZH/flagged_skt_zh.tsv $OUT/; cp $W/pp/flagged_skt_tib.tsv $OUT/
[ -d $OUT/pre-mdc-20260919 ] || cp -r --no-dereference $EN/pre-mdc-20260919 $OUT/
cp $W/pp/postprocess_stats.json $OUT/postprocess_stats.tib.json
cp $W/fullcpd/drop_subcompounds_stats.json $OUT/drop_subcompounds_stats.tib.json 2>/dev/null || true
cat > $OUT/README.md <<'R'
Paper dictionary set since 2026-10-17 (all v4): Sanskrit-English = paper-final-output-20261017-stemext, Sanskrit-Chinese =
paper-candidate-v4zh-20261017-stemext (v4 on MITRA-parallel v3), Sanskrit-Tibetan = v4 (qwen-dictionaries-v4, MITRA-parallel v3,
no integrity gate) through postprocess_final -> full_attestations -> drop_subcompounds -> extend_compounds -> rekey_compounds.
Derivation: ../v4bo-20261017/derive.sh. bidict/spans files are hard links into the source sets (do not edit in place).
R
echo "### $(date +%T) done"; ls -la $OUT
