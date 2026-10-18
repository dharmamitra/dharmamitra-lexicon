#!/usr/bin/env bash
# v2 dictionary span-projection SFT + sa-en (Mitrasamgraha): 9B stage2 (ck6145)
# on five directions (sa-bo, sa-zh, zh-bo, en-bo, sa-en), ONE epoch, full
# finetune with 8-bit Adam on a single GPU, no periodic generation eval (final
# checkpoint written at the end of the epoch), exactly like the 2026-07-30 run
# that produced /mnt/data/qwen35-9b-dictspan-mm.
#
#   ./run_qwen_dictionaries_saen.sh              # full 1-epoch run
#   SMOKE=1 ./run_qwen_dictionaries_saen.sh      # ~3-step smoke
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DICTMINING_EXPERIMENT=SpanProjectionQwenDictSaenExperiment
export DICTMINING_BATCH=${DICTMINING_BATCH:-2}
export DICTMINING_NUM_EPOCHS=${DICTMINING_NUM_EPOCHS:-1}
export DICTMINING_NO_EVAL=${DICTMINING_NO_EVAL:-1}
export DICTMINING_REPORT_TO=${DICTMINING_REPORT_TO:-none}
if [ "${SMOKE:-0}" = "1" ]; then
    export DICTMINING_NUM_EPOCHS=0.02
fi
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
python run_dictmining.py
