#!/usr/bin/env bash
# Evaluation of the released lexicon set (paper-v4-20261017-norecon); outputs to $N.
set -uo pipefail
cd "$(dirname "$0")"
N=/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018; L=$N/logs; mkdir -p $L $N/parts
M=/mnt2/code/mitra-dictionaries
export EVAL_MINED_DIR=/mnt2/code/mitra-dictionaries/paper-v4-20261017-norecon
export HIRA=$M/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv
export CORPUS_SA_BO=$M/norecon-20261017/sa-bo.final.ndjson
export CORPUS_SA_ZH=$M/norecon-20261017/sa-zh.final.ndjson HIRAKAWA_FINAL=$M/norecon-20261017/sa-zh.final.ndjson
sh run_all.sh > $L/run_all.log 2>&1; echo "run_all $?"
NEGI_ATTESTED=1 EVAL_TAG=.attested python3 eval_sa_bo.py > $L/attested.log 2>&1; echo "attested $?"
python3 novel.py > $L/novel.log 2>&1; echo "novel $?"
python3 lemma_coverage.py > $L/lemma_coverage.log 2>&1; echo "lemma_coverage $?"
python3 attestation_composition.py > $L/attestation_composition.log 2>&1; echo "composition $?"
python3 hirakawa_compare.py > $L/hirakawa_compare.log 2>&1; echo "hirakawa_compare $?"
LAYER=$EVAL_MINED_DIR/skt_zh_hirakawa_layer.ndjson python3 hirakawa_gap/overview.py > $L/hirakawa_gap.log 2>&1; echo "hirakawa_gap $?"
python3 paper_numbers.py $EVAL_MINED_DIR $N > $L/paper-numbers.txt 2>&1; echo "paper_numbers $?"
python3 negi_recover.py > $L/negi_recover.log 2>&1; echo "negi_recover $?"
