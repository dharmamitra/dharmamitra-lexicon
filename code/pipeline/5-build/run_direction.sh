#!/usr/bin/env bash
# spans -> bidict -> clean -> stopword-kill for one direction (dedup_ex_all runs
# once at the end of all directions, not here).
set -euo pipefail
D=$1
cd "$(dirname "$0")"
export OUTDIR=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4/final-output
export ENBO_LINKS=/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4/en-bo/enbo_links_all.json
mkdir -p "$OUTDIR"

python run_spans_parallel.py "$D"

case $D in
  sa-bo)
    python build_sabo_bidict.py
    python clean_rank_noise.py $OUTDIR/skt_tib_bidict.sa2bo.ndjson sa bo "/|"
    python clean_rank_noise.py $OUTDIR/skt_tib_bidict.bo2sa.ndjson bo sa "/|"
    python kill_stopword_glosses.py $OUTDIR/skt_tib_bidict.sa2bo.ndjson sa bo $OUTDIR/skt_tib_bidict.bo2sa.ndjson bo
    python kill_src_stopwords.py $OUTDIR/skt_tib_bidict.sa2bo.ndjson sa bo $OUTDIR/skt_tib_bidict.bo2sa.ndjson sa
    ;;
  zh-bo)
    python build_zhbo_bidict.py
    python clean_rank_noise.py $OUTDIR/bo_zh_bidict.zh2bo.ndjson zh bo "/|"
    python clean_rank_noise.py $OUTDIR/bo_zh_bidict.bo2zh.ndjson bo zh "/|"
    python kill_stopword_glosses.py $OUTDIR/bo_zh_bidict.zh2bo.ndjson zh bo $OUTDIR/bo_zh_bidict.bo2zh.ndjson bo
    python kill_src_stopwords.py $OUTDIR/bo_zh_bidict.zh2bo.ndjson zh bo $OUTDIR/bo_zh_bidict.bo2zh.ndjson zh
    ;;
  en-bo)
    python build_enbo_bidict.py
    python clean_rank_noise.py $OUTDIR/en_bo_bidict.en2bo.ndjson en bo "/|"
    python clean_rank_noise.py $OUTDIR/en_bo_bidict.bo2en.ndjson bo en "/|"
    python kill_stopword_glosses.py $OUTDIR/en_bo_bidict.en2bo.ndjson en bo $OUTDIR/en_bo_bidict.bo2en.ndjson bo
    python kill_src_stopwords.py $OUTDIR/en_bo_bidict.en2bo.ndjson en bo $OUTDIR/en_bo_bidict.bo2en.ndjson en
    ;;
  sa-zh)
    python build_sazh_bidict.py
    # boundary chars "/|" match how zh-bo treats its Chinese glosses; neither
    # sa2zh key uses them, so this only re-sorts and applies the PMI floor.
    python clean_rank_noise.py $OUTDIR/skt_zh_bidict.sa2zh.ndjson sa zh "/|"
    python clean_rank_noise.py $OUTDIR/skt_zh_bidict.zh2sa.ndjson zh sa "/|"
    # no Tibetan side here, so kill_stopword_glosses.py does not apply; both sides
    # are source-side and get the SA/ZH stopword pass.
    python kill_src_stopwords.py $OUTDIR/skt_zh_bidict.sa2zh.ndjson sa zh $OUTDIR/skt_zh_bidict.zh2sa.ndjson sa
    python kill_src_stopwords.py $OUTDIR/skt_zh_bidict.zh2sa.ndjson zh sa $OUTDIR/skt_zh_bidict.sa2zh.ndjson zh
    ;;
esac
echo "### $D BUILD DONE ###"
