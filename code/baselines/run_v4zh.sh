#!/usr/bin/env bash
# 2026-10-17: statistical baselines for sa-zh on the v4 corpus (MITRA-parallel v3, qwen-dictionaries-v4-sazh-early).
# Outputs work-v4zh/; refdict evals via build/eval-stemext-v4 (outputs notes/eval-stemext-v4zh-20261017/parts).
set -uo pipefail
cd "$(dirname "$0")"
M=/mnt2/code/mitra-dictionaries
export BASELINE_WORK=$PWD/work-v4zh CORPUS_SA_ZH=$M/qwen-dictionaries-v4-sazh-early/sa-zh/hybrid/final.ndjson
export EVAL_MINED_DIR=$M/paper-candidate-v4zh-20261017-stemext
export HIRA=$M/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv
mkdir -p $BASELINE_WORK; W=$BASELINE_WORK; EV=/home/sebastian/papers/mitra-dict-2026/build/eval-stemext-v4
d=sa-zh
python3 stat_baselines.py prep --direction $d > $W/prep_$d.log 2>&1 || { echo "prep $d FAILED"; exit 1; }
for t in eflomal fast_align mgiza; do
  python3 stat_baselines.py align --direction $d --tool $t > $W/align_${d}_$t.log 2>&1 || echo "align $d $t FAILED"
done
STUDENT_TAGS=abl-mitra9b-3ep python3 stat_baselines.py score --direction $d > $W/score_$d.log 2>&1 || echo "score $d FAILED"
for t in eflomal fast_align mgiza llm; do
  python3 stat_baselines.py dict --direction $d --tool $t >> $W/dict_$d.log 2>&1 || echo "dict $d $t FAILED"
  (cd $EV && EVAL_MINED=$W/$d.$t.bidict.ndjson EVAL_TAG=.$t python3 eval_sa_zh.py > $W/refdict_${d}_$t.log 2>&1) || echo "refdict $d $t FAILED"
done
echo "### done $(date +%T)"
