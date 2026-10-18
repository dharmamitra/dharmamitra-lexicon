import json, re, sys, collections, os, random
sys.path.insert(0, "/mnt2/code/mitra-dictionaries/qwen-dictionaries")
import eval_heldout as E
HELD = os.path.expanduser("~/data/qwen-dictionaries-heldout500")
CK = {"sa-bo": "abl-mitra9b-3ep", "sa-zh": "abl-mitra9b-3ep", "sa-en": "five-dir"}   # mining checkpoints (v3, 2026-10-13)
TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.S)
PRON = {"tad","idam","etad","yad","adas","mad","tvad","asmad","yuṣmad","kim","sva","bhavat","enad","anya","sarva","eka","ātman"}
PART = {"ca","vā","tu","hi","eva","api","iti","na","atha","iva","tathā","yathā","evam","tatra","atra","tatas","yadi","cet","vai","ha","u","nu","khalu","punar","tāvat","yāvat","tatas","mā","sma","kila","uta","aho","atas","yatas","yatra","tadā","yadā","idānīm","iha","tasmāt","su","sa","a","vi","pra","sam","ā"}
def src(row):
    it = row["input_text"]
    sa = dict(TAG.findall(re.search(r"^SANSKRIT: (.*)$", it, re.M).group(1)))
    le = dict(TAG.findall(re.search(r"^LEMMAS: (.*)$", it, re.M).group(1)))
    return sa, le
def kind(n, sa, le):
    w, l = sa.get(n, ""), le.get(n, "")
    k = []
    if w.endswith("-"): k.append("cpd-member")
    elif str(int(n)-1) in sa and sa[str(int(n)-1)].endswith("-"): k.append("cpd-final")
    if l in PRON: k.append("pronoun")
    elif l in PART: k.append("particle")
    return "+".join(k) or "plain word"
def analyse(rows, predkey="pred", label=""):
    out = {}
    for d in ("sa-bo", "sa-zh", "sa-en"):
        c = collections.Counter(); kinds = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
        base = collections.Counter()
        for r in rows:
            if r["lang"] != d: continue
            g, p = E.parse_spans(r["target_text"]), E.parse_spans(r[predkey])
            sa, le = src(r)
            for n in sa: base[kind(n, sa, le)] += 1
            for n in set(g) | set(p):
                if n in g and n in p:
                    if g[n] == p[n]: cat = "agree"
                    elif g[n] in p[n]: cat = "boundary: student longer"
                    elif p[n] in g[n]: cat = "boundary: student shorter"
                    elif E.jaccard(g[n], p[n]) >= 0.5: cat = "boundary: other overlap"
                    else: cat = "different span"
                elif n in g: cat = "student omits"
                else: cat = "student adds"
                c[cat] += 1
                if cat != "agree":
                    kinds[cat][kind(n, sa, le)] += 1
                    ex[cat].append((r, n, sa.get(n), le.get(n), g.get(n), p.get(n)))
        out[d] = (c, kinds, ex, base)
    return out
def show(res, nex=0, seed=1):
    for d, (c, kinds, ex, base) in res.items():
        dis = sum(v for k, v in c.items() if k != "agree")
        tb = sum(base.values())
        print(f"\n##### {d}: {c['agree']} agree, {dis} disagreements; source words by kind: " + ", ".join(f"{k} {100*v/tb:.0f}%" for k, v in base.most_common()))
        for cat, v in sorted(((k, v) for k, v in c.items() if k != "agree"), key=lambda kv: -kv[1]):
            ks = ", ".join(f"{k} {100*n/v:.0f}%" for k, n in kinds[cat].most_common(4))
            print(f"  {cat:28} {v:5d} ({100*v/dis:4.1f}%)   [{ks}]")
            rnd = random.Random(seed); sample = rnd.sample(ex[cat], min(nex, len(ex[cat])))
            for r, n, w, l, g, p in sample:
                tgt = re.sub(r"</?\d+>", "", r["target_text"])
                print(f"      {w} ({l})  T: {g!r}  S: {p!r}   || {tgt[:110]}")
if __name__ == "__main__":
    nex = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = []
    P = {t: [json.loads(l) for l in open(f"{HELD}/test.preds-{t}.jsonl", encoding="utf-8")] for t in set(CK.values())}
    for k, r in enumerate(P["five-dir"]):
        rows.append(P[CK[r["lang"]]][k])
    print("=========== STUDENT vs TEACHER, all 1,489 held-out rows")
    show(analyse(rows), nex)
    if len(sys.argv) > 2:
        # teacher run 2 vs run 1 on the 155 rows
        raw2 = {}
        for d in CK:
            for l in open(f"heldout/run2/{d}/gemini_raw.ndjson", encoding="utf-8"):
                x = json.loads(l); raw2[(d, x["id"])] = x["tagged"]
        rr = []
        for r in rows:
            if (r["lang"], r["id"]) in raw2:
                r2 = dict(r); r2["pred"] = raw2[(r["lang"], r["id"])]; rr.append(r2)
        print("\n=========== TEACHER run 2 vs run 1 (155 rows)  ['student' = run 2]")
        show(analyse(rr), 0)
        print("\n=========== STUDENT vs TEACHER on the same 155 rows")
        show(analyse([r for r in rows if (r["lang"], r["id"]) in raw2]), 0)
