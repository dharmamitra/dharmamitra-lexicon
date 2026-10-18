#!/usr/bin/env bash
# 2026-10-17: v4 sa-zh (MITRA-parallel v3 substrate, built early in qwen-dictionaries-v4-sazh-early) put through the
# paper's postprocess chain, then assembled with the unchanged en/bo files of paper-final-output-20261013-fullcpd.
set -euo pipefail
M=/mnt2/code/mitra-dictionaries; W=$M/v4zh-20261017; E=$M/qwen-dictionaries-v4-sazh-early
FC=$M/paper-final-output-20261013-fullcpd; D=$M/paper-candidate-v4zh-20261017
mkdir -p $W/v4out $W/pp $W/full $W/fullcpd $D; rm -f $W/v4out/*
for f in $E/final-output/skt_zh_*.ndjson; do ln -f $f $W/v4out/; done; ls -la $W/v4out
cd $M/postprocess
echo "### $(date +%T) postprocess"; ONLY_PAIRS=skt_zh python3 postprocess_final.py $W/v4out $W/pp
echo "### $(date +%T) full attestations"; ONLY_PAIRS=skt_zh python3 full_attestations.py $W/pp $W/full
cp -n $W/pp/skt_zh_spans.ndjson $W/full/ 2>/dev/null || true
echo "### $(date +%T) drop subcompounds"; ONLY_PAIRS=skt_zh python3 drop_subcompounds.py $W/full $W/fullcpd
echo "### $(date +%T) hirakawa layer"; cd $E/build
HIRA=$M/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv python3 build_hirakawa_layer.py \
  $E/sa-zh/hybrid/final.ndjson $W/fullcpd/skt_zh_bidict.sa2zh.ndjson $W/fullcpd/skt_zh_hirakawa_layer.ndjson
echo "### $(date +%T) assemble"
for f in $FC/*; do b=$(basename $f); [ -f $f ] && ! [[ $b == skt_zh_* ]] && ln -f $f $D/$b; done
cp -r $FC/pre-mdc-20260919 $D/ 2>/dev/null || true
for f in skt_zh_bidict.sa2zh.ndjson skt_zh_bidict.zh2sa.ndjson skt_zh_spans.ndjson skt_zh_hirakawa_layer.ndjson; do ln -f $W/fullcpd/$f $D/$f; done
cp $W/pp/flagged_skt_zh.tsv $D/flagged_skt_zh.tsv; cp $W/pp/postprocess_stats.json $D/postprocess_stats.zh.json
cp $W/full/full_attestations_stats.json $D/full_attestations_stats.zh.json; cp $W/fullcpd/drop_subcompounds_stats.json $D/drop_subcompounds_stats.zh.json
cat > $D/README.md <<'R'
CANDIDATE (2026-10-17), not the paper set: paper-final-output-20261013-fullcpd with the Sanskrit-Chinese pair replaced by
the v4 build (qwen-dictionaries-v4 mining on MITRA-parallel v3 beads, no lex50 pair filter, no integrity gate), built early in
qwen-dictionaries-v4-sazh-early and put through postprocess_final -> full_attestations -> drop_subcompounds (ONLY_PAIRS=skt_zh),
Hirakawa layer rebuilt. en/bo files are hard links to the fullcpd set. Derivation: ../v4zh-20261017/derive.sh.
R
echo "### $(date +%T) done"; ls -la $D
