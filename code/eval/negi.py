#!/usr/bin/env python3
"""Parser for Negi's Tibetan-Sanskrit dictionary (negi_wylie.txt) -> inverted Sanskrit -> {Tibetan} index.

File format (one entry per line):   <Wylie headword> TAB <entry body>
The body mixes: Devanagari grammar abbreviations (वि. पा. क्रि. ना. सं. ...), Sanskrit equivalents in
Devanagari (usually nominative), an em dash, then Tibetan (Wylie) + Sanskrit (Devanagari) citations
and source sigla with folio numbers; groups are separated by ';', sense numbers ('1.', 'i.') and '•'.

Extraction rule ("equivalent position"): within every group, only the text BEFORE the first em dash
is considered; groups whose pre-dash part contains Wylie are skipped unless they start with '='
(synonym cross reference followed by the Sanskrit equivalent). Parenthesised material is dropped.
Run this file directly to print parse statistics and the known-pair checks.
"""
import re, sys, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sa_key, stem_candidates

NEGI = "/mnt2/code/mitra-dictionaries/references/negi_wylie.txt"

# ---------------------------------------------------------------- Devanagari -> IAST
CONS = {'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ṅ', 'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'ñ',
        'ट': 'ṭ', 'ठ': 'ṭh', 'ड': 'ḍ', 'ढ': 'ḍh', 'ण': 'ṇ', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'ś',
        'ष': 'ṣ', 'स': 's', 'ह': 'h', 'ळ': 'ḷ'}
IVOW = {'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū', 'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ए': 'e',
        'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'}
MAT = {'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū', 'ृ': 'ṛ', 'ॄ': 'ṝ', 'ॢ': 'ḷ', 'े': 'e', 'ै': 'ai',
       'ो': 'o', 'ौ': 'au'}


def dev2iast(s):
    out = []
    pc = False
    for ch in s:
        if ch in CONS:
            if pc:
                out.append('a')
            out.append(CONS[ch]); pc = True
        elif ch in MAT:
            out.append(MAT[ch]); pc = False
        elif ch == '्':
            pc = False
        elif ch in IVOW:
            if pc:
                out.append('a')
            out.append(IVOW[ch]); pc = False
        elif ch in 'ंँ':
            if pc:
                out.append('a'); pc = False
            out.append('ṃ')
        elif ch == 'ः':
            if pc:
                out.append('a'); pc = False
            out.append('ḥ')
        elif ch == 'ऽ':
            if pc:
                out.append('a'); pc = False
            out.append("'")
        elif ch == '़':
            continue
        else:
            if pc:
                out.append('a'); pc = False
            out.append(ch)
    if pc:
        out.append('a')
    return ''.join(out)


# ---------------------------------------------------------------- body parsing
DEV = "ऀ-ॣ०-ॿ"          # Devanagari letters/signs without danda (।॥ = U+0964/5)
DEVCH = re.compile("[%s]" % DEV)
LATIN = re.compile(r"[A-Za-z]")
PAREN = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|<[^<>]*>")
# source siglum with folio/number:  अ.क.222ख/24.163   म.व्यु.937   लो.को.1824
SRC = re.compile(r"(?:[%s]{1,7}\.\s?){1,4}\s?\d[^\s;,]*" % DEV)
# any remaining multi-part dotted abbreviation (भू.का.कृ. / उ.प. / क्रि.वि.)
MULTIABBR = re.compile(r"(?:[%s]{1,7}\.){2,}" % DEV)
# single-token grammar / editorial abbreviations (whitelist, from a frequency list of the file)
ABBR1 = ["वि", "पा", "क्रि", "द्र", "ना", "सं", "कृ", "अव्य", "भूत", "भवि", "वर्त", "सक", "अक", "अवि", "प्र",
         "प्रा", "स्त्री", "पुं", "पु", "नपु", "त्रि", "तुल", "विधौ", "प्रत्य", "संक्षि", "उप", "सर्व", "द्वि", "ब", "अ"]
ABBR1_RE = re.compile(r"(?<![%s])(?:%s)\.(?![%s])" % (DEV, "|".join(sorted(ABBR1, key=len, reverse=True)), DEV))
NAME_RE = re.compile(r"(?<![%s])ना\.(?![%s])" % (DEV, DEV))
TERM_RE = re.compile(r"(?<![%s])पा\.(?![%s])" % (DEV, DEV))
POS_RE = re.compile(r"(?<![%s])(?:वि|क्रि|सं|कृ|अव्य|भूत|भवि|वर्त)\.(?![%s])" % (DEV, DEV))
SEE_RE = re.compile(r"(?<![%s])(?:द्र|तुल)\.(?![%s]).*$" % (DEV, DEV))
# group separators: ';'  bullets  sense numbers '1.' 'ii.'  (number must be followed by whitespace)
# Sigla of lexical works (kośas, Mahāvyutpatti, synonym and homonym lists): an equivalent whose group cites only
# these has no textual attestation in Negi.
SIGLUM = re.compile(r"((?:[%s]{1,7}\.\s?){1,4})\s?\d" % DEV)
def is_lexical_siglum(sg):
    sg = re.sub(r"\s", "", sg)
    if sg == "अभि.को.":          # Abhidharmakośa: a text, not a lexicon
        return False
    return sg.endswith("को.") or "व्यु." in sg or sg in ("ना.ना.",)
def group_attested(grp):
    """True if the group cites at least one non-lexical source (a text, with folio/verse)."""
    return any(not is_lexical_siglum(m.group(1)) for m in SIGLUM.finditer(grp))

SPLIT = re.compile(r";|•|(?:(?<=\s)|^)(?:\d{1,2}|[ivx]{1,4}|[IVX]{1,4})\.(?=\s)")
ITEM_OK = re.compile(r"^[a-zāīūṛṝḷṅñṭḍṇśṣṃḥ' ]+$")
KIND_SUFFIX = ("viśeṣaḥ", "viśeṣa", "bhedaḥ", "bheda", "viśeṣāḥ", "bhedāḥ")


def norm_wylie_head(hw):
    hw = hw.strip()
    hw = hw.split("/")[0]            # "brlags/ o nas" = headword "brlags" (+ cited form "brlags nas")
    hw = re.sub(r"[/*_]+", " ", hw)
    return re.sub(r"\s+", " ", hw).strip()


def _level(sep):
    sep = sep.strip()
    if sep == "•":
        return 0
    if sep == ";" or not sep:
        return None                      # sibling: stays on the current level
    tok = sep.rstrip(".")
    if tok.isdigit():
        return 2
    return 1 if tok.isupper() else 3     # I. II. -> 1 ; i. ii. -> 3


def parse_body(val, stats=None, attested=None):
    """-> (list of IAST equivalents (raw, un-normalised), list of '=' cross-reference targets).

    Hierarchy rule: markers define levels (bullet 0 < 'I.' 1 < '1.' 2 < 'i.' 3; ';' = sibling).
    If the header of an enclosing level already supplied an equivalent, the Devanagari found in
    its numbered children is Negi's Sanskrit *explanation* of the sub-sense (समाधिविशेषः, बोधिसत्त्वः
    after a proper name, ...) and is NOT taken as an equivalent."""
    val = val.replace("\\", " ")
    val = re.sub(r"(?<![A-Za-z'+])o\s?(?=[%s])" % DEV, "॰", val)      # 'o टम्' = abbreviated variant ending
    for _ in range(3):
        val = PAREN.sub(" ", val)
    equivs, xrefs = [], []
    sticky = None            # 'name' / 'term' / None, reset by bullets
    pos = 0
    pieces = []              # (separator that precedes the group, group text)
    sep = ""
    for m in SPLIT.finditer(val):
        pieces.append((sep, val[pos:m.start()]))
        sep = val[m.start():m.end()]
        pos = m.end()
    pieces.append((sep, val[pos:]))
    level = -1
    has_eq = {}              # level -> header of that level supplied an equivalent
    for pi, (sep, grp) in enumerate(pieces):
        lv = _level(sep)
        if lv is not None:
            level = lv
            for l in [x for x in has_eq if x >= lv]:
                del has_eq[l]
            if lv == 0:
                sticky = None
        pre = re.split(r"[—–]", grp, maxsplit=1)[0]
        pre = SEE_RE.sub(" ", pre)
        if NAME_RE.search(pre):
            sticky = "name"
        elif TERM_RE.search(pre):
            sticky = "term"
        elif POS_RE.search(pre):
            sticky = None
        pre = SRC.sub(" ", pre)
        pre = MULTIABBR.sub(" ", pre)
        pre = ABBR1_RE.sub(" ", pre)
        pre = pre.strip()
        if not pre:
            continue
        if LATIN.search(pre):
            if not pre.startswith("="):
                if stats is not None:
                    stats["groups_skipped_wylie_subentry"] += 1
                continue
            # '= wylie synonym(s) [Devanagari equivalent]'
            mdev = DEVCH.search(pre)
            wy = pre[1:mdev.start()] if mdev else pre[1:]
            for t in re.split(r"/_?|,", wy):
                t = norm_wylie_head(t)
                if t and LATIN.search(t):
                    xrefs.append(t)
            pre = pre[mdev.start():] if mdev else ""
            if LATIN.search(pre):          # Wylie again after the Devanagari: keep only first Devanagari stretch
                pre = LATIN.split(pre, maxsplit=1)[0]
        pre = pre.lstrip("= ")
        if not DEVCH.search(pre):
            continue
        if "।" in pre or "॥" in pre:
            if stats is not None:
                stats["groups_skipped_sanskrit_citation_with_danda"] += 1
            continue
        if any(has_eq.get(l) for l in has_eq if l < level):
            if stats is not None:
                stats["groups_skipped_explanatory_subsense"] += 1
            continue
        items = [x for x in re.split(r"[,/]", pre)]
        kept = 0
        for it in items:
            it = re.sub(r"[?*!\d\"“”‘’.:=+~^{}|-]", " ", it)
            it = re.sub(r"\s+", " ", it).strip()
            if not it or not DEVCH.search(it):
                continue
            ia = re.sub(r"\s+", " ", dev2iast(it)).strip().lower()
            if not ITEM_OK.match(ia):
                if stats is not None:
                    stats["items_rejected_nonIAST"] += 1
                continue
            if ia.count(" ") >= 4:              # >=5 words: a Sanskrit quotation, not an equivalent
                if stats is not None:
                    stats["items_dropped_long_quotation"] += 1
                continue
            if kept >= 1 and sticky in ("name", "term"):
                if stats is not None:
                    stats["items_dropped_name_or_term_descriptor"] += 1
                continue
            if kept >= 1 and ia.endswith(KIND_SUFFIX):
                if stats is not None:
                    stats["items_dropped_kind_descriptor"] += 1
                continue
            equivs.append(ia)
            if attested is not None:
                attested.append((pi, level))
            kept += 1
        if kept:
            has_eq[level] = True
    if attested is not None:
        # An equivalent counts as textually attested if its own group cites a text, or any group nested below it
        # (a header such as '• सं. धर्मः' is attested through the citations of its numbered sub-senses).
        # Sibling groups (';') at the same level are other equivalents and do not count.
        flags = []
        for pi, lv in attested:
            ok = group_attested(pieces[pi][1])
            j, inside = pi + 1, False
            while not ok and j < len(pieces):
                lj = _level(pieces[j][0])
                if lj is not None:
                    if lj <= lv:
                        break
                    inside = True
                if inside and group_attested(pieces[j][1]):
                    ok = True
                j += 1
            flags.append(ok)
        attested[:] = flags
    return equivs, xrefs


def load_negi(tib_norm_fn, stats=None):
    """Returns (pairs, info): pairs = list of (tib_headword_raw, tib_key, sanskrit_form_key)."""
    st = stats if stats is not None else collections.Counter()
    entries = {}     # raw head -> (equivs, xrefs, attested flags)
    order = []
    for line in open(NEGI, encoding="utf-8"):
        st["lines"] += 1
        if "\t" not in line:
            st["lines_without_tab"] += 1
            continue
        hw, val = line.rstrip("\n").split("\t", 1)
        hw = norm_wylie_head(hw)
        if not hw:
            st["lines_empty_headword"] += 1
            continue
        att = []
        eq, xr = parse_body(val, st, att)
        if hw in entries:
            st["duplicate_headword_lines_merged"] += 1
            entries[hw] = (entries[hw][0] + eq, entries[hw][1] + xr, entries[hw][2] + att)
        else:
            entries[hw] = (eq, xr, att)
            order.append(hw)
    pairs = []
    for hw in order:
        eq, xr, att = entries[hw]
        src = "own"
        if not eq and xr:
            st["entries_pure_xref"] += 1
            inh, inh_att = [], []
            for t in xr:
                if t in entries and entries[t][0]:
                    inh += entries[t][0]
                    inh_att += entries[t][2]
            if inh:
                eq, att, src = inh, inh_att, "xref"
                st["entries_pure_xref_resolved"] += 1
            else:
                st["entries_pure_xref_unresolved"] += 1
        if not eq:
            st["entries_without_sanskrit_equivalent"] += 1
            continue
        st["entries_with_sanskrit_equivalent"] += 1
        tk = tib_norm_fn(hw)
        # src: "own" = Negi gives a textual citation for this equivalent, "own-lex" = only lexical works
        # (or no source), "xref" / "xref-lex" = inherited from the entry the headword points to ('= ...')
        best = {}
        # (an inherited equivalent keeps the attestation status it has in the entry it is inherited from)
        for i, e in enumerate(eq):
            a = i < len(att) and att[i]
            best[e] = best.get(e, False) or a
        for e, a in best.items():
            k = sa_key(e)
            if k:
                pairs.append((hw, tk, k, src + ("" if a else "-lex")))
    st["pairs_tib_skt"] = len(pairs)
    return pairs, st


def build_index(pairs, mined_keys):
    """Invert to Sanskrit -> set(Tibetan keys). Reference forms are nominatives; each form is
    expanded with stem_candidates() and attached to every candidate that is a mined headword key.
    Returns (index, rule_counter, n_ref_single_forms, n_ref_single_forms_covered)."""
    index = collections.defaultdict(set)
    rules = collections.Counter()
    forms = set()
    covered = set()
    for hw, tk, k, src in pairs:
        if " " in k:
            continue
        forms.add(k)
        for cand, rule in stem_candidates(k):
            if cand in mined_keys:
                if tk not in index[cand]:
                    index[cand].add(tk)
                rules[rule] += 1
                covered.add(k)
    return index, rules, len(forms), len(covered)


if __name__ == "__main__":
    from tibetan import tib_key
    pairs, st = load_negi(tib_key)
    for k, v in sorted(st.items()):
        print(f"{k:45s} {v}")
    inv = collections.defaultdict(set)
    for hw, tk, k, src in pairs:
        for cand, rule in stem_candidates(k):
            inv[cand].add(tk)
    for sa, bo in [("karuṇā", "snying rje"), ("dharma", "chos"), ("śūnyatā", "stong pa nyid"),
                   ("bodhisattva", "byang chub sems dpa'")]:
        got = inv.get(sa_key(sa), set())
        print(f"CHECK {sa} -> {bo}: {'OK' if tib_key(bo) in got else 'MISSING'}  ({len(got)} Tibetan equivalents) e.g. {sorted(got)[:8]}")
