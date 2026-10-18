#!/usr/bin/env python3
"""English gloss normalisation for the sa->en evaluation (no external dependencies).

Pipeline for a string:  lowercase -> strip diacritics (NFD, drop combining marks) -> tokens [a-z]+ (possessive 's
dropped) -> British->American spelling folding (-our -> -or, -is(e|ed|ing|ation) -> -iz...) -> irregular-form map
(said->say, went->go, ...) -> Porter (1980) stemmer, original algorithm, implemented below.
nltk is NOT installed in the python used for this evaluation; the stemmer below was verified to be identical to
nltk.stem.PorterStemmer(mode='ORIGINAL_ALGORITHM') on the evaluation vocabulary (see verify_porter.py).
"""
import re, unicodedata

STOPWORDS = set("""a an the and or but if while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too very s t can will just don should
now i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
it its itself they them their theirs themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing would could shall may might must as also thus thou thee thy ye unto
upon o oh""".split())

IRREGULAR = {
    "said": "say", "says": "say", "went": "go", "gone": "go", "goes": "go", "came": "come", "saw": "see", "seen": "see",
    "gave": "give", "given": "give", "took": "take", "taken": "take", "made": "make", "done": "do", "told": "tell",
    "knew": "know", "known": "know", "thought": "think", "spoke": "speak", "spoken": "speak", "heard": "hear",
    "stood": "stand", "sat": "sit", "ate": "eat", "eaten": "eat", "drank": "drink", "drunk": "drink", "slept": "sleep",
    "fell": "fall", "fallen": "fall", "rose": "rise", "risen": "rise", "arose": "arise", "arisen": "arise",
    "born": "bear", "bore": "bear", "borne": "bear", "became": "become", "began": "begin", "begun": "begin",
    "brought": "bring", "bought": "buy", "caught": "catch", "chose": "choose", "chosen": "choose", "drew": "draw",
    "drawn": "draw", "drove": "drive", "driven": "drive", "fled": "flee", "flew": "fly", "flown": "fly",
    "fought": "fight", "found": "find", "forgot": "forget", "forgotten": "forget", "got": "get", "gotten": "get",
    "grew": "grow", "grown": "grow", "held": "hold", "kept": "keep", "laid": "lay", "led": "lead", "left": "leave",
    "lost": "lose", "meant": "mean", "met": "meet", "paid": "pay", "ran": "run", "sang": "sing", "sung": "sing",
    "sent": "send", "shone": "shine", "shook": "shake", "shaken": "shake", "slew": "slay", "slain": "slay",
    "sold": "sell", "sought": "seek", "spent": "spend", "stole": "steal", "stolen": "steal", "struck": "strike",
    "swore": "swear", "sworn": "swear", "taught": "teach", "threw": "throw", "thrown": "throw", "understood": "understand",
    "woke": "wake", "woken": "wake", "won": "win", "wore": "wear", "worn": "wear", "wrote": "write", "written": "write",
    "bound": "bind", "broke": "break", "broken": "break", "built": "build", "burnt": "burn", "dwelt": "dwell",
    "felt": "feel", "hid": "hide", "hidden": "hide", "lay": "lie", "lain": "lie", "lit": "light", "rode": "ride",
    "ridden": "ride", "sprang": "spring", "sprung": "spring", "stuck": "stick", "swept": "sweep", "wept": "weep",
    "men": "man", "women": "woman", "children": "child", "feet": "foot", "teeth": "tooth", "mice": "mouse",
    "kine": "cow", "oxen": "ox", "brethren": "brother", "people": "person", "better": "good", "best": "good",
    "worse": "bad", "worst": "bad",
}

_OUR = re.compile(r"^([a-z]{2,})our(s|ed|ing|able|ably|ite|ites|ful|fully|less|er|ers|hood)?$")
_ISE = re.compile(r"^([a-z]{2,})is(e|es|ed|ing|ation|ations|er|ers)$")
_TOK = re.compile(r"[a-z]+")


def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def spell(w):
    m = _OUR.match(w)
    if m:
        return m.group(1) + "or" + (m.group(2) or "")
    m = _ISE.match(w)
    if m:
        return m.group(1) + "iz" + m.group(2)
    return w


# ---------------------------------------------------------------- Porter stemmer (original 1980 algorithm)
_V = "aeiou"


def _cons(w, i):
    c = w[i]
    if c in _V:
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(w):
    n, i, L = 0, 0, len(w)
    while i < L and _cons(w, i):
        i += 1
    while i < L:
        while i < L and not _cons(w, i):
            i += 1
        if i >= L:
            break
        n += 1
        while i < L and _cons(w, i):
            i += 1
    return n


def _hasvowel(w):
    return any(not _cons(w, i) for i in range(len(w)))


def _dblc(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    if len(w) < 3:
        return False
    if not (_cons(w, len(w) - 1) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 3)):
        return False
    return w[-1] not in "wxy"


_S2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
       ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
       ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
       ("iviti", "ive"), ("biliti", "ble")]
_S3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
_S4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate",
       "iti", "ous", "ive", "ize"]


def porter(w):
    if len(w) <= 2:
        return w
    # 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # 1b
    flag = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _hasvowel(w[:-2]):
        w = w[:-2]; flag = True
    elif w.endswith("ing") and _hasvowel(w[:-3]):
        w = w[:-3]; flag = True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _dblc(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # 1c
    if w.endswith("y") and _hasvowel(w[:-1]):
        w = w[:-1] + "i"
    # 2
    for suf, rep in _S2:
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # 3
    for suf, rep in _S3:
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # 4
    best = None
    for suf in _S4:
        if w.endswith(suf) and (best is None or len(suf) > len(best)):
            best = suf
    if best:
        stem = w[:-len(best)]
        if _m(stem) > 1 and (best != "ion" or stem.endswith(("s", "t"))):
            w = stem
    # 5a
    if w.endswith("e"):
        stem = w[:-1]
        mm = _m(stem)
        if mm > 1 or (mm == 1 and not _cvc(stem)):
            w = stem
    # 5b
    if _m(w) > 1 and _dblc(w) and w.endswith("l"):
        w = w[:-1]
    return w


_scache = {}


def norm_token(t):
    r = _scache.get(t)
    if r is None:
        r = porter(IRREGULAR.get(t) or spell(t))
        _scache[t] = r
    return r


def tokens(s):
    s = fold(s).replace("’", "'").replace("'s ", " ")
    return _TOK.findall(s)


def stem_set(text):
    """all tokens of a reference text (stopwords included), normalised."""
    return {norm_token(t) for t in tokens(text)}


def gloss_stems(gloss):
    """(content stems, all stems) of a mined gloss."""
    toks = tokens(gloss)
    content = [norm_token(t) for t in toks if t not in STOPWORDS]
    return content, [norm_token(t) for t in toks]
