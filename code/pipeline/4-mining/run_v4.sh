#!/usr/bin/env bash
# MITRA-dict v4 (2026-10-16): sa-bo + sa-zh dictionaries on MITRA-parallel v3 (qwen post-corrected beads).
#   substrate : prep_v3_substrate.py (same Sanskrit analysis / Chinese segmentation as v3)
#   delta     : delta.py -> only records whose prompt or segment numbers differ from v3 are mined; the rest reuse v3 rows
#   mine      : mine_v4.py, same checkpoint as v3, 2 GPUs data-parallel, chunked + resumable
#   build     : same steps as qwen-dictionaries-v3/run_split.sh minus the sa-zh lex50 pair filter and the integrity gate (all mined rows used) (sa-en unchanged, not rebuilt)
#   STAGES="build eval" ./run_v4.sh   # only some stages
set -uo pipefail
V4=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4; cd $V4; LOG=$V4/logs
export DICTSPAN_CKPT=/mnt/data/abl-mitra9b-3ep-mm GPU_UTIL=${GPU_UTIL:-0.85}
STAGES=${STAGES:-"mine combine build eval"}
stage() { echo; echo "### $(date '+%F %T') stage: $1"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

mine_delta() {  # direction-key dir
  local D=$1 OUT=$2 pids=()
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k python3 mine_v4.py --direction $D --part $k --nparts 2 --input $OUT/substrate.delta.ndjson \
      --out $OUT/mined_delta.ndjson > $LOG/mine_delta_${D}_p$k.log 2>&1 &
    pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "[$(date +%T)] mine delta $D exit $rc"; grep -h -E "QC|DONE|Traceback" $LOG/mine_delta_${D}_p*.log | tail -n 4
  return $rc
}

if has mine; then
  stage mine
  mine_delta sa-zh-v2 $V4/sa-zh || { echo "sa-zh mining failed"; exit 1; }
  until [ -f $V4/sa-bo/delta.stats.json ]; do sleep 30; done   # delta.py writes stats last
  cat $V4/sa-bo/delta.stats.json
  mine_delta sa-bo-v2 $V4/sa-bo || { echo "sa-bo mining failed"; exit 1; }
fi

if has combine; then
  stage combine
  for d in sa-zh sa-bo; do   # prepare_finals.py reads mined.ndjson.part0 + part1
    cat $V4/$d/reused.ndjson $V4/$d/mined_delta.ndjson.part0 > $V4/$d/mined.ndjson.part0
    cp $V4/$d/mined_delta.ndjson.part1 $V4/$d/mined.ndjson.part1
    wc -l $V4/$d/reused.ndjson $V4/$d/mined_delta.ndjson.part* $V4/$d/mined.ndjson.part*
  done
  # no pair-level lexical filter (lexfilter_sazh.py / lex50): v3 beads are already post-corrected + verified
  rm -f $V4/sa-zh/lex_drop_ids.txt
fi

if has build; then
  stage build
  cd $V4/build
  python3 prepare_finals.py sa-zh sa-bo || exit 1
  bash run_direction.sh sa-bo || exit 1
  bash run_direction.sh sa-zh || exit 1
  python3 dedup_ex_all.py $V4/final-output || exit 1
  HIRA=/mnt2/code/mitra-dictionaries/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv \
    python3 build_hirakawa_layer.py $V4/sa-zh/hybrid/final.ndjson $V4/final-output/skt_zh_bidict.sa2zh.ndjson $V4/final-output/skt_zh_hirakawa_layer.ndjson
  cd $V4; wc -l final-output/*bidict*.ndjson
fi

if has eval; then
  stage eval
  cd /home/sebastian/papers/mitra-dict-2026/build/eval
  for d in bo zh; do
    EVAL_MINED_DIR=$V4/final-output EVAL_TAG=.v4 HIRA=/mnt2/code/mitra-dictionaries/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv \
      python3 eval_sa_$d.py > $LOG/eval_sa_$d.log 2>&1 || echo "eval sa-$d failed"
  done
fi
echo "### $(date '+%F %T') ALL DONE"
