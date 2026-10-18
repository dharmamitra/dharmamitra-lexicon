#!/usr/bin/env bash
set -uo pipefail
E=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4-sazh-early; cd $E
echo "### $(date +%T) combine"; python3 -I final_combine.py sa-zh substrate.delta.ndjson:mined_delta.ndjson substrate.delta2.ndjson:mined_delta2.ndjson || exit 1
rm -f sa-zh/lex_drop_ids.txt
cd build
echo "### $(date +%T) prepare"; python3 prepare_finals.py sa-zh || exit 1
echo "### $(date +%T) direction"; bash run_direction.sh sa-zh || exit 1
echo "### $(date +%T) dedup"; python3 dedup_ex_all.py $E/final-output || exit 1
echo "### $(date +%T) hirakawa"; HIRA=/mnt2/code/mitra-dictionaries/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv \
  python3 build_hirakawa_layer.py $E/sa-zh/hybrid/final.ndjson $E/final-output/skt_zh_bidict.sa2zh.ndjson $E/final-output/skt_zh_hirakawa_layer.ndjson || exit 1
wc -l $E/final-output/*.ndjson; echo "### $(date +%T) ALL DONE"
