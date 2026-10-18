#!/usr/bin/env python3
"""Hirakawa layer for sa-zh (2026-10-07; papers/mitra-dict-2026/notes/pipeline-facts.md §17).
For every entry of the re-OCR'd Hirakawa (hirakawa-reocr-2026/hirakawa_reocr.tsv): what evidence does MITRA-dict have?
  dict_exact          the (Sanskrit, Chinese) pair is a sense of the sa2zh dictionary (compound-aware key: hyphens
                      removed, vowel length folded, final ḥ/ṃ/m/s/r stripped; or the sandhi-insensitive consonant skeleton)
  dict_compositional  our headwords found inside Hirakawa's Sanskrit have Chinese senses covering >= 75% of his Chinese
  pair_attested       Chinese and Sanskrit co-occur in a sentence-pair group (records sharing a Chinese segment merged):
                      fuzzy partial ratio >= 80 on the folded Sanskrit for Chinese >= 3 chars; for 2-char Chinese a
                      folded substring match in >= 2 groups and more than twice the control count. Control = the same
                      test against the group 3 positions away.
This layer is EVIDENCE FOR HIRAKAWA'S ENTRIES, kept apart from the mined dictionary: do not merge it into sa2zh and do
not evaluate MITRA-dict against Hirakawa with it (circular).
usage: build_hirakawa_layer.py PAIRS(final.ndjson) SA2ZH(bidict) OUT(.ndjson)  [env HIRA]"""
import sys, os, re, json, collections, ast
from multiprocessing import Pool
sys.path.insert(0, "/home/sebastian/papers/mitra-dict-2026/build/eval")
os.environ.setdefault("HIRA", "/mnt2/code/mitra-dictionaries/mitra-dictionaries/non-english/hirakawa-reocr-2026/hirakawa_reocr.tsv")
import eval_sa_zh as Z
from rapidfuzz import fuzz
import ahocorasick
PAIRS, SA2ZH, OUT = sys.argv[1:4]
LET = re.compile(r"[a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ]+")
FOLD = str.maketrans({"ā": "a", "ī": "i", "ū": "u", "ṝ": "ṛ", "ḹ": "ḷ"})
SK = str.maketrans({"g": "k", "j": "c", "ḍ": "ṭ", "d": "t", "b": "p", "ṇ": "n", "ñ": "n", "ṅ": "n", "m": "n", "ś": "s", "ṣ": "s"})
SUFFIX = ["bhūtatva", "tvāt", "tvena", "tva", "tā", "vat", "mat", "vant", "ka", "ika", "in", "tas", "bhūta", "bhāva"]
def fold(s): return "".join(LET.findall(s.lower().replace("ṁ", "ṃ"))).translate(FOLD)
def skel(s): return re.sub(r"[aiuṛḷeoṃḥyvr]", "", fold(s).translate(SK))
def endstrip(f): return re.sub(r"(ḥ|ṃ|m|s|r)$", "", f)
def kf(s): return endstrip(fold(s.replace("-", "").replace(" ", "")))
def ks(s): k = skel(endstrip(fold(s.replace("-", "").replace(" ", "")))); return k if len(k) >= 3 else None
def strip_suffix(sa):
    for x in SUFFIX:
        if sa.endswith(x) and len(sa) - len(x) >= 5: return sa[:-len(x)]
    return sa
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
pairs, _ = Z.load_hirakawa(False)
H = collections.defaultdict(set)
for sa, zh in pairs:
    k = Z.zh_key(zh)
    if len(k) >= 2 and len(sa) >= 4: H[k].add(sa)
MAXL = max(len(k) for k in H)
# ---- dictionary side ----
A, B, top = set(), set(), {}
for l in open(SA2ZH, encoding="utf-8"):
    d = json.loads(l); k = kf(d["sa"]); k2 = ks(d["sa"]); zs = [Z.zh_key(s["zh"]) for s in d["senses"]]
    for z in zs:
        if z: A.add((z, k)); B.add((z, k2)) if k2 else None
    top.setdefault(k, []).extend(z for z in zs[:15] if z)
AC = ahocorasick.Automaton()
for k in top:
    if len(k) >= 3: AC.add_word(k, k)
AC.make_automaton()
def dict_status(z, sa):
    if (z, kf(sa)) in A or (ks(sa) and (z, ks(sa)) in B): return "dict_exact", None
    cov = set(); parts = []
    for _, m in AC.iter(fold(sa)):
        for o in top[m]:
            if o in z:
                i = z.find(o); cov.update(range(i, i + len(o))); parts.append((m, o))
    if len(z) >= 2 and len(cov) / len(z) >= 0.75: return "dict_compositional", sorted(set(parts))[:8]
    return None, None
# ---- pair side ----
rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
parent = list(range(len(rows))); seen = {}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, d in enumerate(rows):
    for zz in lit(d["tgt_segnr"]):
        if zz in seen: parent[find(i)] = find(seen[zz])
        else: seen[zz] = i
grp = collections.defaultdict(list)
for i in range(len(rows)): grp[find(i)].append(i)
G = [(Z.zh_key("\n".join(dict.fromkeys(rows[i]["target"] for i in idx))), fold(" ".join(rows[i]["src_original"] for i in idx)), rows[idx[0]]["id"])
     for _, idx in sorted(grp.items())]
del rows
def match(sa, src, two):
    if two:
        f = fold(sa); f = f[:-1] if len(f) >= 6 else f
        return len(f) >= 4 and f in src
    f = fold(strip_suffix(sa)); return len(f) >= 6 and fuzz.partial_ratio(f, src, score_cutoff=80) > 0
def work(i):
    t = G[i][0]; zs = set()
    for a in range(len(t)):
        for L in range(2, min(MAXL, len(t) - a) + 1):
            if t[a:a + L] in H: zs.add(t[a:a + L])
    j = i + 3 if i + 3 < len(G) else i - 3
    return [(z, sa, match(sa, G[i][1], len(z) == 2), match(sa, G[j][1], len(z) == 2), G[i][2]) for z in zs for sa in H[z]]
def ds_pair(j):
    z, sa, other = j
    a = dict_status(z, sa)[0] == "dict_compositional"
    b = other is not None and dict_status(z, other)[0] is not None
    return a, b
if __name__ == "__main__":
    import random
    real = collections.Counter(); ctrl = collections.Counter(); inpairs = set()
    with Pool(110) as pool:
        for res in pool.imap_unordered(work, range(len(G)), chunksize=50):
            for z, sa, r, c, pid in res:
                inpairs.add((z, sa)); real[(z, sa)] += r; ctrl[(z, sa)] += c
    E = [(z, sa) for z, sas in H.items() for sa in sas]
    rnd = random.Random(20261017); bylen = collections.defaultdict(list)
    for z, sa in E: bylen[len(z)].append(sa)
    n = att = catt = 0; jobs = []
    for z, sa in E:
        n += 1; r, c = real[(z, sa)], ctrl[(z, sa)]
        att += (r >= 2 and r > 2 * c) if len(z) == 2 else r >= 1
        catt += (c >= 2 and c > 2 * r) if len(z) == 2 else c >= 1
        other = None
        for _ in range(20):
            o = rnd.choice(bylen[len(z)])
            if o not in H[z]: other = o; break
        jobs.append((z, sa, other))
    with Pool(100) as pool:
        res = pool.map(ds_pair, jobs, chunksize=200)
    comp = sum(a for a, b in res); ccomp = sum(b for a, b in res)
    print(json.dumps({"entries": n, "attested_%": round(100*att/n,2), "control_attested_%": round(100*catt/n,2),
        "attested_among_zh_in_pairs_%": round(100*att/max(1,len(inpairs)),2), "control_among_zh_in_pairs_%": round(100*catt/max(1,len(inpairs)),2),
        "compositional_%": round(100*comp/n,2), "control_dict_other_sanskrit_%": round(100*ccomp/n,2)}))
