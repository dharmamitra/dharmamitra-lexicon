"""Fresh hapax sample (2026-10-18): 20 random sa->X entries with n == 1 per pair from the released norecon lexica.
Writes notes/eval-lfix-20261018/hapax-resample/hapax-sa{en,bo,zh}-entries.md."""
import json, random, ast, re
M = "/mnt2/code/mitra-dictionaries"
P = f"{M}/paper-v4-20261017-norecon"
OUT = "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hapax-resample"
PAIRS = [("en", "skt_en", f"{M}/norecon-20261017/sa-en.mined.ndjson", "English"),
         ("bo", "skt_tib", f"{M}/norecon-20261017/sa-bo.final.ndjson", "Tibetan, Wylie"),
         ("zh", "skt_zh", f"{M}/norecon-20261017/sa-zh.final.ndjson", "Chinese")]
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
for tag, pfx, F, lang in PAIRS:
    once = []
    for l in open(f"{P}/{pfx}_bidict.sa2{tag}.ndjson", encoding="utf-8"):
        if '"n": 1,' in l:
            d = json.loads(l)
            if d["n"] == 1: once.append(d)
    S = random.Random(20261018).sample(once, 20)
    ids = {s["senses"][0]["ex"][0]["id"] for s in S}; R = {}
    for l in open(F, encoding="utf-8"):
        i = l[8:l.index('"', 8)]
        if i in ids: R[i] = json.loads(l)
    out = [f"# 20 random Sanskrit→{lang.split(',')[0]} entries attested exactly once (MITRA-lexicon, released sa→{tag}, norecon)\n\n"
           f"Population: {len(once)} sa→{tag} headwords attested once. Seed 20261018. Missing rows: {len(ids - set(R))}\n"]
    for k, e in enumerate(S, 1):
        s = e["senses"][0]; x = s["ex"][0]; r = R.get(x["id"], {})
        kind = "compound" if "-" in e["sa"] and " " not in e["sa"] else ("phrase / surface form" if " " in e["sa"] or len(x.get("indices") or []) > 1 else "single word")
        t = x["target"]; mark = t[:x["start"]] + "⟦" + t[x["start"]:x["end"]] + "⟧" + t[x["end"]:]
        toks = lit(r.get("toks", []))
        if "aligns" in r: al = {int(a): b for a, b in lit(r["aligns"]).items()}
        else:
            al = {}
            for m in re.finditer(r"<(\d+(?:,\d+)*)>(.*?)</\1>", r.get("output", "")):
                for j in m.group(1).split(","): al[int(j)] = m.group(2)
        tl = " ".join(f"[{j}]{w}" + (f"(lemma {lm})" if lm != w.rstrip('-') else "") + (f"→{al[j]}" if j in al else "") for j, (w, lm) in enumerate(toks, 1))
        out.append(f"## Entry {k:02d}: `{e['sa']}` → {s[tag]}  ({kind})\n\n- Case forms: {e.get('forms')}; token indices {x.get('indices')}\n"
                   f"- Sanskrit sentence: {r.get('src_original', x.get('src'))}\n- {lang} (aligned span in ⟦ ⟧): {mark}\n"
                   f"- Analyzer tokens (surface, lemma) → aligned spans: {tl}\n- Pair id: `{x['id']}`\n")
    open(f"{OUT}/hapax-sa{tag}-entries.md", "w", encoding="utf-8").write("\n".join(out))
    print(tag, len(once), len(ids - set(R)), flush=True)
