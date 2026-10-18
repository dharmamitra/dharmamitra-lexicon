#!/usr/bin/env bash
set -uo pipefail
cd ~/code/lme
L=~/code/lme/logs/ablation; Q=/mnt2/code/mitra-dictionaries/qwen-dictionaries
TPL2B=${TPL2B:?set to a Qwen3.5-2B model template directory}
E=SpanProjectionDictSaenMitraQwen2B3epExperiment; T=abl-mitra2b-3ep
export DICTMINING_BATCH=2 DICTMINING_NUM_EPOCHS=3 DICTMINING_NO_EVAL=1 DICTMINING_REPORT_TO=none NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
echo "[$(date +%H:%M)] train $E"
CUDA_VISIBLE_DEVICES=0 DICTMINING_EXPERIMENT=$E python run_dictmining.py > $L/train_$E.log 2>&1; echo "[$(date +%H:%M)] exit $?"
python ablation/convert_to_mm.py --src results/$E/2e-05/final --out /mnt/data/$T-mm --template $TPL2B > $L/convert_$T.log 2>&1
( cd $Q && CUDA_VISIBLE_DEVICES=0 python eval_heldout.py --ckpt /mnt/data/$T-mm --data ~/data/qwen-dictionaries-heldout500/test.jsonl --tag $T --gpu-util 0.85 ) > $L/eval_$T.log 2>&1
cd $Q/baselines
for d in sa-bo sa-zh sa-en; do STUDENT_TAGS=abl-mitra2b,abl-mitra2b-3ep,abl-vanilla9b,abl-gemma2 python stat_baselines.py score --direction $d; done > $L/link_scores_2b3ep.log 2>&1
echo "[$(date +%H:%M)] ALL DONE"
