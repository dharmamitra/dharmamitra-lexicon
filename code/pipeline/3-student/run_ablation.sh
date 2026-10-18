#!/usr/bin/env bash
# Base-model ablation for the MITRA-dict paper (2026-10-07): same five-direction span-projection data and recipe as
# /mnt/data/qwen35-9b-dictspan-saen-mm, three other base models; then held-out evaluation (span F1, link F1).
#   GPU1: MITRA Qwen3.5-2B  |  GPU0: vanilla Qwen3.5-9B-Base, then gemma2-mitra-base
set -uo pipefail
cd ~/code/lme
L=~/code/lme/logs/ablation; mkdir -p $L
Q=/mnt2/code/mitra-dictionaries/qwen-dictionaries
HELD=~/data/qwen-dictionaries-heldout500
TPL2B=${TPL2B:?set to a Qwen3.5-2B model template directory}
export DICTMINING_BATCH=2 DICTMINING_NUM_EPOCHS=1 DICTMINING_NO_EVAL=1 DICTMINING_REPORT_TO=none NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

train() {  # gpu experiment
  echo "[$(date +%H:%M)] train $2 on GPU$1"
  CUDA_VISIBLE_DEVICES=$1 DICTMINING_EXPERIMENT=$2 python run_dictmining.py > $L/train_$2.log 2>&1
  echo "[$(date +%H:%M)] train $2 exit $?"
}
evaluate() {  # gpu ckpt tag gpu_util
  echo "[$(date +%H:%M)] eval $3"
  ( cd $Q && CUDA_VISIBLE_DEVICES=$1 python eval_heldout.py --ckpt $2 --data $HELD/test.jsonl --tag $3 --gpu-util $4 ) > $L/eval_$3.log 2>&1
  echo "[$(date +%H:%M)] eval $3 exit $?"
}

(
  E=SpanProjectionDictSaenMitraQwen2BExperiment
  train 1 $E
  python ablation/convert_to_mm.py --src results/$E/2e-05/final --out /mnt/data/abl-mitra-qwen35-2b-dictspan-saen-mm --template $TPL2B > $L/convert_2b.log 2>&1
  evaluate 1 /mnt/data/abl-mitra-qwen35-2b-dictspan-saen-mm abl-mitra2b 0.35
) > $L/gpu1.log 2>&1 &
P1=$!

(
  E=SpanProjectionDictSaenVanillaQwen9BExperiment
  train 0 $E
  python ablation/convert_to_mm.py --src results/$E/2e-05/final --out /mnt/data/abl-vanilla-qwen35-9b-dictspan-saen-mm > $L/convert_vanilla9b.log 2>&1
  evaluate 0 /mnt/data/abl-vanilla-qwen35-9b-dictspan-saen-mm abl-vanilla9b 0.85
  while pgrep -f "hf download buddhist-nlp/gemma2-mitra-base" > /dev/null; do sleep 60; done
  E=SpanProjectionDictSaenGemmaMitraExperiment
  train 0 $E
  evaluate 0 ~/code/lme/results/$E/2e-05/final abl-gemma2 0.85
) > $L/gpu0.log 2>&1 &
P0=$!
wait $P1 $P0

echo "[$(date +%H:%M)] link-level scoring"
cd $Q/baselines
for d in sa-bo sa-zh sa-en; do
  STUDENT_TAGS=abl-mitra2b,abl-vanilla9b,abl-gemma2 python stat_baselines.py score --direction $d
done > $L/link_scores.log 2>&1
echo "[$(date +%H:%M)] ALL DONE"
