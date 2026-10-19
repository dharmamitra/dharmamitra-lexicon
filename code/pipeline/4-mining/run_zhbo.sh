#!/usr/bin/env bash
# zh-bo v4 (2026-10-19): MITRA-parallel v3 bo-zh pairs, mined with the 3-epoch student, built like sa-bo/sa-zh v4.
#   kundoku : Chinese segmentation as in the zh-bo seeds (kokuyaku model -> kana-stripped kanji runs)
#   mine    : mine_v4.py --direction zh-bo-v3 (2 GPUs data-parallel, resumable)
#   build   : prepare_finals.py zh-bo -> run_direction.sh zh-bo (bo_zh_* files only, no dedup)
#   webapp  : evidence file for the web interface (not part of the release)
#   STAGES="build" ./run_zhbo.sh
set -uo pipefail
V4=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4; cd $V4; LOG=$V4/logs
K=/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026/sa-zh
export DICTSPAN_CKPT=/mnt/data/abl-mitra9b-3ep-mm GPU_UTIL=${GPU_UTIL:-0.85}
STAGES=${STAGES:-"kundoku substrate mine build"}
stage() { echo; echo "### $(date '+%F %T') stage: $1"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
waitall() { local rc=0; for p in "$@"; do wait $p || rc=1; done; return $rc; }

if has kundoku; then
  stage kundoku
  pids=()
  for g in 0 1; do
    ( for s in $g $((g+2)); do
        CUDA_VISIBLE_DEVICES=$g python3 $K/kokuyaku_translate_hf.py --files zh-bo/kundoku/todo.$s.ndjson || exit 1
        python3 $K/segment_zh_from_japanese.py --files zh-bo/kundoku/todo.$s.ndjson || exit 1
      done ) > $LOG/zhbo_kundoku_g$g.log 2>&1 &
    pids+=($!)
  done
  waitall "${pids[@]}" || { echo "kundoku failed"; exit 1; }
fi

if has substrate; then
  stage substrate
  python3 prep_v3_zhbo_substrate.py write || exit 1
fi

if has mine; then
  stage mine
  pids=()
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k python3 mine_v4.py --direction zh-bo-v3 --part $k --nparts 2 \
      --out $V4/zh-bo/mined.ndjson > $LOG/mine_zh-bo-v3_p$k.log 2>&1 &
    pids+=($!)
  done
  waitall "${pids[@]}" || { echo "zh-bo mining failed"; exit 1; }
  grep -h -E "QC|DONE" $LOG/mine_zh-bo-v3_p*.log
fi

if has build; then
  stage build
  cd $V4/build
  python3 prepare_finals.py zh-bo || exit 1
  bash run_direction.sh zh-bo || exit 1
  # no dedup_ex_all.py: one attestation per occurrence, so count == len(ex) (as in the released sa-bo/sa-zh lexica)
  cd $V4; wc -l final-output/bo_zh_*.ndjson
fi
if has webapp; then
  stage webapp   # v9 webapp evidence: zh-bo only (other pairs as in webapp-v8)
  cd $V4/build
  mkdir -p $V4/webapp-v9
  OUT_DIR=$V4/webapp-v9 SPANS_FILE=$V4/final-output/bo_zh_spans.ndjson python3 build_zhbo_webapp_evidence.py || exit 1
  OUT_DIR=$V4/webapp-v9 KILL_DIR=$V4/final-output/killlists python3 filter_combined_stopwords.py zh-bo || exit 1
  ls -la $V4/webapp-v9
fi
echo "### $(date '+%F %T') ZH-BO ALL DONE"
