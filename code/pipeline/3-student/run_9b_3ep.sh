#!/usr/bin/env bash
set -uo pipefail
cd ~/code/lme
L=~/code/lme/logs/ablation; Q=/mnt2/code/mitra-dictionaries/qwen-dictionaries
E=SpanProjectionQwenDictSaen3epExperiment; T=abl-mitra9b-3ep
GPUS=${GPUS:-0}; NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)
export DICTMINING_BATCH=${DICTMINING_BATCH:-2} DICTMINING_NUM_EPOCHS=3 DICTMINING_NO_EVAL=1 DICTMINING_REPORT_TO=none NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
echo "[$(date +%H:%M)] train $E on GPUs $GPUS"
if [ "$NPROC" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=$GPUS DICTMINING_EXPERIMENT=$E torchrun --nproc_per_node $NPROC --master_port 29611 run_dictmining.py > $L/train_$E.log 2>&1
else
  CUDA_VISIBLE_DEVICES=$GPUS DICTMINING_EXPERIMENT=$E python run_dictmining.py > $L/train_$E.log 2>&1
fi
echo "[$(date +%H:%M)] exit $?"
python ablation/convert_to_mm.py --src results/$E/2e-05/final --out /mnt/data/$T-mm > $L/convert_$T.log 2>&1
( cd $Q && CUDA_VISIBLE_DEVICES=${GPUS%%,*} python eval_heldout.py --ckpt /mnt/data/$T-mm --data ~/data/qwen-dictionaries-heldout500/test.jsonl --tag $T --gpu-util 0.85 ) > $L/eval_$T.log 2>&1
cd $Q/baselines
for d in sa-bo sa-zh sa-en; do STUDENT_TAGS=abl-mitra9b-3ep,abl-mitra2b-3ep python stat_baselines.py score --direction $d; done > $L/link_scores_9b3ep.log 2>&1
echo "[$(date +%H:%M)] ALL DONE"
