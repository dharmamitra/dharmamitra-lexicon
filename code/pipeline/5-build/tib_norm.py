"""Tibetan (Wylie) normalization for the bilingual dicts.

strip_particles(bo): MERGE normalization - drop trailing grammatical particles and the
   apostrophe-clitic genitive/finalizer suffix on the last syllable, so case variants
   collapse to one lemma (snying rje'i / snying rje kyi -> snying rje). Keeps >=1 syllable.

verb_root(bo): the present/root form of a single-syllable inflected verb (byas -> byed),
   from tibetan-verbs-database. Used COMPLEMENTARILY (fold forms under the root, keep the
   originals), never to overwrite. Returns None if not a known distinct inflected form.
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    VERB_MAP = json.load(open(os.path.join(_HERE, "refresh2026/tib_verb_map.json"), encoding="utf-8"))
except Exception:
    VERB_MAP = {}

# trailing case / conjunction / topic particles (whole final syllables)
TRAIL = {"gi", "kyi", "gyi", "yi", "gis", "kyis", "gyis", "yis",
         "tu", "du", "su", "ru", "la", "na", "ra", "dang", "ni", "kyang", "yang",
         "'i", "'o", "'ang", "'am", "'i'o"}
# apostrophe-clitics fused onto the last syllable (rje'i -> rje, pa'i -> pa)
SUFFIX = ("'i", "'o", "'ang", "'am")


def strip_particles(bo):
    bo = (bo or "").strip()
    if not bo:
        return bo
    toks = bo.split()
    while len(toks) > 1 and toks[-1] in TRAIL:
        toks.pop()
    last = toks[-1]
    for suf in SUFFIX:
        if last.endswith(suf) and len(last) > len(suf):
            last = last[:-len(suf)]
            break
    toks[-1] = last
    out = " ".join(toks).strip()
    return out or bo


def verb_root(bo):
    bo = (bo or "").strip()
    if not bo or " " in bo:          # single syllable only (conservative)
        return None
    r = VERB_MAP.get(bo)
    return r if r and r != bo else None


def tib_lemma(bo):
    """Canonical lemma: strip trailing particles, then map a single-syllable inflected
    verb to its root. Mirrors the Sanskrit lemma (vs the surface form, kept separately)."""
    s = strip_particles(bo)
    return verb_root(s) or s
