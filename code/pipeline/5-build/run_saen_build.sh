#!/usr/bin/env bash
# sa-en (Mitrasamgraha): spans (compound stitching) -> bidict -> rank-noise clean ->
# source-side stopword pass on BOTH sides (no Tibetan side, so kill_stopword_glosses.py
# does not apply -- same reasoning as the sa-zh case in run_direction.sh).
set -euo pipefail
cd "$(dirname "$0")"
export OUTDIR=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4/final-output
mkdir -p "$OUTDIR"
python build_saen_spans.py
python build_saen_bidict.py
python clean_rank_noise.py $OUTDIR/skt_en_bidict.sa2en.ndjson sa en ".;:?!"
python clean_rank_noise.py $OUTDIR/skt_en_bidict.en2sa.ndjson en sa ".;:?!"
python kill_src_stopwords.py $OUTDIR/skt_en_bidict.sa2en.ndjson sa en $OUTDIR/skt_en_bidict.en2sa.ndjson sa
python kill_src_stopwords.py $OUTDIR/skt_en_bidict.en2sa.ndjson en sa $OUTDIR/skt_en_bidict.sa2en.ndjson en
echo "### sa-en BUILD DONE ###"
