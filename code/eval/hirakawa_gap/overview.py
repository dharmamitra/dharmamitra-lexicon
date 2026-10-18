import json, pickle, collections, sys, re, ast
import os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# Hirakawa coverage buckets (tab:hirakawagap), moved from the scratchpad 2026-10-13. Env: EVAL_MINED_DIR, LAYER, CORPUS_SA_ZH, HIRA.
from fuzzy import H, Z, fold
import ahocorasick
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
P = "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/parts"
V = {"current": dict(fo=os.environ["EVAL_MINED_DIR"], pairs=os.environ["CORPUS_SA_ZH"], layer=os.environ["LAYER"],
                   ev=f"{P}/sa-zh.json")}
SC = pickle.load(open(os.path.join(HERE, "scan.pkl"), "rb")); canon = set(SC["occ"])
E = [(z, sa) for z, sas in H.items() for sa in sas]
AS = ahocorasick.Automaton(); fk = {}
for e in E:
    f = fold(e[1]); f = f[:-1] if len(f) >= 6 else f
    if len(f) >= 4: fk[e] = f; AS.add_word(f, f)
AS.make_automaton(); found = set()
for p in ("/mnt2/code/mitra-dictionaries/norecon-20261017/sa-zh.substrate.ndjson", "/mnt2/code/mitra-dictionaries/norecon-20261017/sa-bo.substrate.ndjson"):
    for l in open(p, encoding="utf-8"):
        i = l.find('"root_string": "'); j = l.find('", "par_string"', i)
        for _, f in AS.iter(fold(json.loads('"' + l[i + 16:j] + '"'))): found.add(f)
AZ = ahocorasick.Automaton()
for z in H: AZ.add_word(z, z)
AZ.make_automaton()
def text_of(s): return re.sub(r"_\d{3}$", "", s.split(":")[0])
out = {}
for name, v in V.items():
    R = {}
    h = s = a = 0
    for l in open(f"{v['fo']}/skt_zh_bidict.sa2zh.ndjson"): d = json.loads(l); h += 1; s += len(d["senses"]); a += d["n"]
    R["sa2zh headwords"] = h; R["zh2sa headwords"] = sum(1 for _ in open(f"{v['fo']}/skt_zh_bidict.zh2sa.ndjson")); R["senses"] = s; R["attestations"] = a
    zp = set(); npairs = 0; kosa = collections.Counter(); multi = 0
    for l in open(v["pairs"], encoding="utf-8"):
        d = json.loads(l); npairs += 1
        for _, z in AZ.iter(Z.zh_key(d["target"])): zp.add(z)
        segs = lit(d["tgt_segnr"]); ts = {text_of(x) for x in segs}; multi += len(ts) > 1
        if len(ts) == 1:
            t = next(iter(ts))
            if t in ("ZH_T29_1558", "ZH_T29_1559"): kosa[t] += len(lit(d["aligns"]))
    R["sentence pairs"] = npairs; R["pairs mixing translations"] = multi
    R["Kosa aligned words T1558 / T1559 (single-translation pairs)"] = f"{kosa['ZH_T29_1558']} / {kosa['ZH_T29_1559']}"
    ev = json.load(open(v["ev"])); st, rx = ev["variants"]["strict"], ev["variants"]["relaxed"]
    R["single-word headwords"] = ev["mined_single_word_headwords"]["all"]; R["|E| (judged by Hirakawa)"] = ev["E_headwords"]["all"]
    for k, lab in (("P@1", "P@1"), ("recall_all_macro", "recall")):
        R[f"Hirakawa strict {lab}"] = round(100 * st["all"][k], 1); R[f"Hirakawa relaxed {lab}"] = round(100 * rx["all"][k], 1)
    for b in ("1", "2-4", "5-19", "20-99", "100-999", ">=1000"): R[f"strict P@1, n {b}"] = round(100 * st[b]["P@1"], 1)
    L = {(d["zh"], d["sa"]): d for d in map(json.loads, open(v["layer"]))}
    c = collections.Counter()
    for e in E:
        d = L.get(e)
        if d and d.get("dict") == "dict_exact": k = "1 in dictionary (exact)"
        elif d and d.get("pair_attested"): k = "2 attested in our pairs"
        elif d: k = "3 composable from entries"
        elif e[0] not in zp: k = "5 Chinese nowhere in the canon" if e[0] not in canon else "4 Chinese not in our pairs"
        elif fk.get(e) in found: k = "6 both strings present, never together"
        else: k = "7 Sanskrit not in our corpus"
        c[k] += 1
    for k in sorted(c): R["Hirakawa: " + k] = round(100 * c[k] / len(E), 1)
    R["Hirakawa: covered (1+2+3)"] = round(100 * sum(c[k] for k in c if k[0] in "123") / len(E), 1)
    out[name] = R
keys = list(out["current"].keys())
print(f"{'':58}" + "".join(f"{n:>16}" for n in V))
for k in keys: print(f"{k:58}" + "".join(f"{str(out[n][k]):>16}" for n in V))
json.dump(out, open(os.environ.get("OUT", "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hirakawa-gap.json"), "w"), ensure_ascii=False, indent=1)
