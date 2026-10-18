#!/usr/bin/env bash
# 2026-10-17: statistical baselines (sa-bo, sa-zh) on the v4 corpora with reconstructed Sanskrit texts removed
# (norecon-20261017, list postprocess/reconstructed_texts.tsv). Outputs work-norecon-{bo,zh}/; refdict evals via build/eval-norecon.
set -uo pipefail
cd "$(dirname "$0")"
M=/mnt2/code/mitra-dictionaries
export EVAL_MINED_DIR=$M/paper-v4-20261017-norecon
export HIRA=$M/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv
export CORPUS_SA_BO=$M/norecon-20261017/sa-bo.final.ndjson CORPUS_SA_ZH=$M/norecon-20261017/sa-zh.final.ndjson
EV=/home/sebastian/papers/mitra-dict-2026/build/eval-norecon
run() {  # direction short
  d=$1; s=$2; export BASELINE_WORK=$PWD/work-norecon-$s; W=$BASELINE_WORK; mkdir -p $W
  python3 stat_baselines.py prep --direction $d > $W/prep_$d.log 2>&1 || { echo "prep $d FAILED"; return 1; }
  for t in eflomal fast_align mgiza; do
    python3 stat_baselines.py align --direction $d --tool $t > $W/align_${d}_$t.log 2>&1 || echo "align $d $t FAILED"
  done
  STUDENT_TAGS=abl-mitra9b-3ep python3 stat_baselines.py score --direction $d > $W/score_$d.log 2>&1 || echo "score $d FAILED"
  for t in eflomal fast_align mgiza llm; do
    python3 stat_baselines.py dict --direction $d --tool $t >> $W/dict_$d.log 2>&1 || echo "dict $d $t FAILED"
    (cd $EV && EVAL_MINED=$W/$d.$t.bidict.ndjson EVAL_TAG=.$t python3 eval_sa_$s.py > $W/refdict_${d}_$t.log 2>&1) || echo "refdict $d $t FAILED"
  done
  echo "### $d done $(date +%T)"
}
run sa-bo bo & run sa-zh zh & wait
