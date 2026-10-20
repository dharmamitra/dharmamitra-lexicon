#!/bin/sh
# Reproduce everything (CPU only, ~1 minute). No arguments.
set -e
cd "$(dirname "$0")"
mkdir -p /home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/parts
for d in bo zh en; do
  python3 eval_sa_$d.py > /home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/parts/log-sa-$d.txt 2>&1
done
python3 make_report.py
