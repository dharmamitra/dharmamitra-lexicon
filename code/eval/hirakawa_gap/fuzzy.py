"""Fuzzy grounding of Hirakawa (zh, sa) entries in v3 lex50 pairs, with a false-positive control:
each matcher is also run against the Sanskrit of the pair 3 records away (same text, wrong pair)."""
import sys, os, re, json, collections, pickle, ast
sys.path.insert(0, "/home/sebastian/papers/mitra-dict-2026/build/eval")
import eval_sa_zh as Z
from multiprocessing import Pool
from rapidfuzz import fuzz
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.dirname(HERE)
FINAL = f"{S}/raw/v3lex50/sa-zh/hybrid/final.ndjson"
pairs, _ = Z.load_hirakawa(False)
H = collections.defaultdict(set)
for sa, zh in pairs:
    k = Z.zh_key(zh)
    if len(k) >= 2 and len(sa) >= 4: H[k].add(sa)
MAXL = max(len(k) for k in H)
LET = re.compile(r"[a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ]+")
FOLD = str.maketrans({"ā": "a", "ī": "i", "ū": "u", "ṝ": "ṛ", "ḹ": "ḷ"})
def fold(s): return "".join(LET.findall(s.lower().replace("ṁ", "ṃ"))).translate(FOLD)
SK = str.maketrans({"g": "k", "j": "c", "ḍ": "ṭ", "d": "t", "b": "p", "ṇ": "n", "ñ": "n", "ṅ": "n", "m": "n", "ś": "s", "ṣ": "s"})
def skel(s):
    s = fold(s).translate(SK)
    return re.sub(r"[aiuṛḷeoṃḥyvr]", "", s)          # drop vowels, anusvāra, visarga, glides, r (r/ḥ sandhi)
SUFFIX = ["bhūtatva", "tvāt", "tvena", "tva", "tā", "vat", "mat", "vant", "ka", "ika", "in", "tas", "bhūta", "bhāva"]
def strip_suffix(sa):
    for x in SUFFIX:
        if sa.endswith(x) and len(sa) - len(x) >= 5: return sa[:-len(x)]
    return sa
def m1(sa, src, words, lem):          # current: word-initial prefix / minus final letter / compound-internal for >= 8
    s2 = sa[:-1] if len(sa) >= 6 else None
    return any(w.startswith(sa) or (s2 and w.startswith(s2)) or (len(sa) >= 8 and s2 in w) for w in words)
def m2(sa, src, words, lem):          # folded, across word boundaries, minus final letter
    f = fold(sa); return len(f) >= 4 and f[:-1 if len(f) >= 6 else len(f)] in src["fold"]
def m3(sa, src, words, lem):          # consonant skeleton
    k = skel(sa); return len(k) >= 4 and k in src["skel"]
def m4(sa, src, words, lem):          # skeleton after stripping abstract/derivational suffixes
    k = skel(strip_suffix(sa)); return len(k) >= 4 and k in src["skel"]
def m5(sa, src, words, lem):          # DharmaNexus lemmas: single lemma or consecutive compound members joined, skeleton
    k = skel(strip_suffix(sa))
    return (len(k) >= 3 and k in lem["single"]) or (len(k) >= 4 and k in lem["joined"])
def m6(sa, src, words, lem):          # fuzzy partial ratio on folded strings
    f = fold(strip_suffix(sa))
    if len(f) < 6: return False
    return fuzz.partial_ratio(f, src["fold"], score_cutoff=88) > 0
METHODS = [("M1 current", m1), ("M2 folded", m2), ("M3 skeleton", m3), ("M4 +suffix strip", m4), ("M5 lemmas", m5), ("M6 fuzzy 88", m6)]
def prep(d):
    src = d["src_original"]; words = LET.findall(src.lower().replace("ṁ", "ṃ"))
    toks = d["toks"]; toks = ast.literal_eval(toks) if isinstance(toks, str) else toks
    single = {skel(l or u) for u, l in toks}; joined = set(); cur = []
    for u, l in toks:
        cur.append(l or u.rstrip("-"))
        if not u.endswith("-"):
            if len(cur) > 1:
                for i in range(len(cur)):
                    for j in range(i + 2, len(cur) + 1): joined.add(skel("".join(cur[i:j])))
            cur = []
    joined_s = "|".join(joined)
    return {"fold": fold(src), "skel": skel(src)}, words, {"single": single, "joined": joined_s}
ROWS = None
def init(rows):
    global ROWS; ROWS = rows
def work(i):
    d = ROWS[i]; c = ROWS[i + 3] if i + 3 < len(ROWS) else ROWS[i - 3]
    t = Z.zh_key(d["target"]); zs = set()
    for a in range(len(t)):
        for L in range(2, min(MAXL, len(t) - a) + 1):
            if t[a:a + L] in H: zs.add(t[a:a + L])
    if not zs: return []
    P = prep(d); Q = prep(c); out = []
    for z in zs:
        for sa in H[z]:
            r = tuple(f(sa, *P) for _, f in METHODS); q = tuple(f(sa, *Q) for _, f in METHODS)
            out.append((z, sa, r, q))
    return out
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(FINAL, encoding="utf-8")]
    real = collections.defaultdict(lambda: [False] * len(METHODS)); ctrl = collections.defaultdict(lambda: [False] * len(METHODS))
    with Pool(110, initializer=init, initargs=(rows,)) as pool:
        for res in pool.imap_unordered(work, range(len(rows)), chunksize=100):
            for z, sa, r, q in res:
                a = real[(z, sa)]; b = ctrl[(z, sa)]
                for k in range(len(METHODS)):
                    a[k] |= r[k]; b[k] |= q[k]
    pickle.dump({"real": dict(real), "ctrl": dict(ctrl), "names": [n for n, _ in METHODS]}, open(os.path.join(HERE, "fuzzy.pkl"), "wb"))
    ent = list(real)
    print(f"Hirakawa entries whose Chinese occurs in our pairs: {len(ent)}")
    for L in ("2", "3", "4+", "all"):
        E = [e for e in ent if L == "all" or (L == "4+" and len(e[0]) >= 4) or str(len(e[0])) == L]
        print(f"\n zh length {L}: {len(E)} entries   (grounded % | control false-positive %)")
        cum_r = [False] * len(E); cum_c = [False] * len(E)
        for k, n in enumerate(n for n, _ in METHODS):
            r = sum(real[e][k] for e in E); c = sum(ctrl[e][k] for e in E)
            cum_r = [x or real[e][k] for x, e in zip(cum_r, E)]; cum_c = [x or ctrl[e][k] for x, e in zip(cum_c, E)]
            print(f"   {n:18} {100*r/len(E):5.1f} | {100*c/len(E):5.1f}    cumulative {100*sum(cum_r)/len(E):5.1f} | {100*sum(cum_c)/len(E):5.1f}")
