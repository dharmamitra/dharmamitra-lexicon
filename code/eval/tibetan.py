#!/usr/bin/env python3
"""Light Wylie normalisation + strict / relaxed matching for the sa->bo evaluation.

tib_key(): (copied and extended from qwen-dictionaries/build/tib_norm.py::strip_particles;
the production module is NOT imported so that this evaluation is self-contained)
  0. repair of a Unicode->Wylie conversion artefact of the mined files: "ga[-]y" -> "g.y"
  1. shad / underscore / asterisk / pipe -> space, collapse whitespace, typographic apostrophes -> '
  2. repeat until stable:
       - drop trailing particle syllables (only while >1 syllable remains):
         gi kyi gyi yi 'i | gis kyis gyis yis | la na du tu su ru | las nas | dang | ni | kyang yang 'ang | 'am | 'o
         (exception: a final "las" is kept when preceded by a genitive (gi/kyi/gyi/yi/-'i) or by (')phrin,
          where it is the noun "action" and not the ablative particle)
       - strip clitics fused on the last syllable: 'i  'o  'am  'ang  'is  ('i'o -> first 'o then 'i)
  3. fused la-don / instrumental on the nominalisers, only if the expression has >1 syllable:
       par->pa  bar->ba  pas->pa  bas->ba  por->po  pos->po  bor->bo  bos->bo
No verb-stem normalisation, no removal of pa/ba/po nominalisers, Wylie case is preserved
(capital letters encode retroflexes / long vowels).
"""
import re

TRAIL = {"gi", "kyi", "gyi", "yi", "'i", "gis", "kyis", "gyis", "yis", "la", "na", "du", "tu", "su", "ru",
         "las", "nas", "dang", "ni", "kyang", "yang", "'ang", "'am", "'o", "'i'o"}
GEN = {"gi", "kyi", "gyi", "yi", "'i"}
LAS_NOUN_PREV = {"phrin", "'phrin"}
SUFFIX = ("'i", "'o", "'am", "'ang", "'is")
FUSED = {"par": "pa", "bar": "ba", "pas": "pa", "bas": "ba", "por": "po", "pos": "po", "bor": "bo", "bos": "bo"}
# function syllables (copied from kill_stopword_glosses.py FUNC) - used only to guard relaxed matching
FUNC = {'ni', 'na', 'no', 'nas', 'su', 'tu', 'du', 'ru', 'la', 'las', 'dang', 'gi', 'gis', 'gyi', 'gyis', 'kyi',
        'kyis', 'yi', 'yis', "'i", 'pa', "pa'i", 'ba', "ba'i", 'po', 'mo', 'par', 'bar', 'pas', 'bas', 'ste', 'te',
        'de', 'cing', 'zhing', 'shing', 'ces', 'zhes', 'yin', 'yod', 'med', 'do', "'o", 'so', 'to', 'ro', 'go',
        'ngo', 'lo', "'am", "'ang", 'yang', 'kyang', 'zhig', 'cig', 'gcig', 'rnams', 'dag', 'can', 'nyid', 'tsam',
        "'di", 'der', 'dir'}

_CLEAN = re.compile(r"[/_*|,;()\[\]]+")


def tib_key(bo):
    bo = (bo or "").replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    bo = bo.replace("ga[-]y", "g.y")     # conversion artefact in the mined files: g.yas is also written "ga[-]yas"
    bo = _CLEAN.sub(" ", bo)
    toks = bo.split()
    if not toks:
        return ""
    changed = True
    while changed:
        changed = False
        while len(toks) > 1 and toks[-1] in TRAIL:
            if toks[-1] == "las" and (toks[-2] in GEN or toks[-2].endswith("'i") or toks[-2] in LAS_NOUN_PREV):
                break                      # "X'i las" / "phrin las": las = the noun 'action', not the ablative
            toks.pop(); changed = True
        last = toks[-1]
        for suf in SUFFIX:
            if last.endswith(suf) and len(last) > len(suf):
                toks[-1] = last[:-len(suf)]; changed = True
                break
    if len(toks) > 1 and toks[-1] in FUSED:
        toks[-1] = FUSED[toks[-1]]
    return " ".join(toks)


def pad(k):
    return " " + k + " "


def has_content(k):
    return any(t not in FUNC for t in k.split())


def contains(big_padded, small_padded):
    """syllable-level contiguous containment (both arguments padded with spaces)."""
    return small_padded in big_padded
