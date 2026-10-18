#!/usr/bin/env bash
# v4 round 2 (2026-10-16): continue run_v4.sh after MITRA-parallel v3 was rebuilt (triangulation + fix106 spans).
#   1. wait for the v3 rebuild (build_release.py) -> re-prep substrates into r2/ -> delta2 (records not yet covered)
#   2. wait for the running sa-zh miners (round 1) -> mine sa-zh delta2, then sa-bo delta2 (sa-bo round 1 never ran)
#   3. final_combine.py (rows for exactly the rebuilt records) -> build + eval via run_v4.sh stages
set -uo pipefail
V4=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4; cd $V4; LOG=$V4/logs
export DICTSPAN_CKPT=/mnt/data/abl-mitra9b-3ep-mm GPU_UTIL=${GPU_UTIL:-0.85}
REBUILD_PID=${REBUILD_PID:-1534804}; ZH_MINERS=${ZH_MINERS:-"1530114 1530115"}
stage() { echo; echo "### $(date '+%F %T') stage: $1"; }

stage wait_rebuild
while kill -0 $REBUILD_PID 2>/dev/null; do sleep 20; done
ls -la --time-style=+%T /mnt2/code/mitra-multilingual-matching/mitra-parallel-v3/; cat /mnt2/code/mitra-multilingual-matching/mitra-parallel-v3/build_stats.json

stage prep_r2
if ! { grep -q DONE $LOG/prep_r2_sa-zh.log 2>/dev/null && grep -q DONE $LOG/prep_r2_sa-bo.log 2>/dev/null; }; then
mkdir -p r2; cp prep_v3_substrate.py r2/
python3 -I r2/prep_v3_substrate.py sa-zh > $LOG/prep_r2_sa-zh.log 2>&1 &
python3 -I r2/prep_v3_substrate.py sa-bo > $LOG/prep_r2_sa-bo.log 2>&1
wait; tail -n1 $LOG/prep_r2_*.log
grep -q DONE $LOG/prep_r2_sa-zh.log && grep -q DONE $LOG/prep_r2_sa-bo.log || { echo "prep failed"; exit 1; }
fi

stage delta2
if [ ! -f sa-zh/delta2.stats.json ] || [ ! -f sa-bo/delta2.stats.json ]; then
python3 -I delta2.py sa-zh --cover sa-zh/substrate.delta.ndjson > $LOG/delta2_sa-zh.log 2>&1 &
python3 -I delta2.py sa-bo > $LOG/delta2_sa-bo.log 2>&1
wait; cat $LOG/delta2_*.log
[ -f sa-zh/delta2.stats.json ] && [ -f sa-bo/delta2.stats.json ] || { echo "delta2 failed"; exit 1; }
fi

mine2() {  # direction-key dir
  local D=$1 OUT=$2 pids=()
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k python3 mine_v4.py --direction $D --part $k --nparts 2 --input $OUT/substrate.delta2.ndjson \
      --out $OUT/mined_delta2.ndjson > $LOG/mine_delta2_${D}_p$k.log 2>&1 &
    pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "[$(date +%T)] mine delta2 $D exit $rc"; grep -h -E "QC|DONE|Traceback" $LOG/mine_delta2_${D}_p*.log | tail -n 4
  return $rc
}

stage wait_sa-zh_round1
for p in $ZH_MINERS; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
grep -h -E "QC|DONE|Traceback" $LOG/mine_delta_sa-zh-v2_p*.log | tail -n 4
grep -q DONE $LOG/mine_delta_sa-zh-v2_p0.log && grep -q DONE $LOG/mine_delta_sa-zh-v2_p1.log || { echo "sa-zh round 1 incomplete"; exit 1; }

stage mine_delta2
mine2 sa-zh-v2 $V4/sa-zh || { echo "sa-zh delta2 mining failed"; exit 1; }
mine2 sa-bo-v2 $V4/sa-bo || { echo "sa-bo delta2 mining failed"; exit 1; }

stage final_combine
python3 -I final_combine.py sa-zh substrate.delta.ndjson:mined_delta.ndjson substrate.delta2.ndjson:mined_delta2.ndjson &
python3 -I final_combine.py sa-bo substrate.delta2.ndjson:mined_delta2.ndjson
wait; cat sa-zh/combine.stats.json sa-bo/combine.stats.json || exit 1
rm -f sa-zh/lex_drop_ids.txt

STAGES="build eval" bash ./run_v4.sh
