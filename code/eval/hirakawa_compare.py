#!/usr/bin/env python3
"""Close comparison of the Hirakawa digitization with the Sanskrit-Chinese dictionary of MITRA-dict
(the counterpart of the Negi analyses: negi_recover.py, negi_sources.py, the headword bands).

1. Hirakawa equivalents that no mined sense matches (relaxed), for headwords in E: does the Chinese string occur in
   our Chinese at all, and together with the Sanskrit lemma? Is it a phonetic transcription?
2. Chinese headwords by frequency band: Hirakawa headwords by occurrence in our Chinese corpus vs. our zh->sa
   headwords by attestations.
Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hirakawa-compare.json, /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hirakawa-missing-samples.md
"""
import collections, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key
import novel, eval_sa_zh as Z

FINAL = os.environ.get("HIRAKAWA_FINAL", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-zh/hybrid/final.ndjson")
TAG = os.environ.get("EVAL_TAG", "")  # output suffix, e.g. .v3
OUT = "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018"
# characters typical of Buddhist phonetic transcription (a heuristic, not exhaustive)
TRANSCR = set("阿婆迦羅囉哆多陀那摩訶薩埵毘鞞跋提波般若耶伽闍吒茶拏彌蜜尼梨利唎離頗槃盤叉沙娑舍奢輸首呾怛他咀嚕盧樓魯喃南曩嚩縛囀嚩鉢缽尸翅柘遮旃栴檀健犍揵乾俱拘鳩究瞿屈掘羯竭吉祇耆嚧醯呬兮枳羝底帝坻諦邏洛勒刺剌磨麼莫謨母牟悉瑟徵蘇修須輸蹉嗟差闡懺車奴怒弩褥耨那捺訥歌柯哥迦蘭攔藍爛闌")


def transcription(zh):
    return len(zh) >= 3 and sum(c in TRANSCR for c in zh) / len(zh) >= 0.75


def main():
    rnd = random.Random(42)
    cap = novel.run("sa-zh")
    recs = [r for r in cap["recs"] if r.get("ref_missing") is not None]
    missing = [(r["sa"], r["n"], x, [g for g, *_ in r["senses"][:3]]) for r in recs for x in r["ref_missing"]]
    matched = [(r["sa"], r["n"], x) for r in recs for x in cap["index"][sa_key(r["sa"])] if x not in r["ref_missing"]]
    print(f"Hirakawa equivalents for headwords in E: missing {len(missing)}, matched {len(matched)}")
    zk = lambda z: Z.zh_key(Z.zh_plain(z))
    want = collections.defaultdict(set)
    for sa, n, x, _ in missing + [(a, b, c, None) for a, b, c in matched]:
        want[zk(x)].add(sa_key(sa))
    maxlen = max(len(k) for k in want)
    str_count, cooc, lem_count = collections.Counter(), collections.Counter(), collections.Counter()
    nrows = 0
    lem_re = re.compile(r"""\[\s*(?:'[^']*'|"[^"]*"),\s*(?:'([^']*)'|"([^"]*)")\s*\]""")
    for line in open(FINAL, encoding="utf-8"):
        d = json.loads(line)
        nrows += 1
        t = Z.zh_key(Z.zh_plain(d.get("target", "")))
        found = set()
        for i in range(len(t)):
            for L in range(1, min(maxlen, len(t) - i) + 1):
                if t[i:i + L] in want:
                    found.add(t[i:i + L])
        toks = d.get("toks") or []
        lems = {sa_key(x[1] or x[0] or "") for x in toks} if isinstance(toks, list) else \
               {sa_key(a or b) for a, b in lem_re.findall(toks)}
        for l in lems:
            lem_count[l] += 1
        for k in found:
            str_count[k] += 1
            for l in want[k] & lems:
                cooc[(l, k)] += 1
    res = {}
    for label, pool in (("missing", missing), ("matched", [(a, b, c, None) for a, b, c in matched])):
        c = collections.Counter()
        for sa, n, x, _ in pool:
            k = zk(x)
            c["all"] += 1
            c["transcription"] += transcription(k)
            c["string_in_corpus"] += str_count[k] > 0
            c["cooccurs_with_lemma"] += cooc[(sa_key(sa), k)] > 0
            c["len1"] += len(k) == 1
        res[label] = dict(c)
        print(f"{label:8} {c['all']:6d} | transcription {100*c['transcription']/c['all']:5.1f}% | single character "
              f"{100*c['len1']/c['all']:5.1f}% | string occurs in our Chinese {100*c['string_in_corpus']/c['all']:5.1f}% | "
              f"co-occurs with the Sanskrit lemma {100*c['cooccurs_with_lemma']/c['all']:5.1f}%")
    # categories of the missing ones
    cat = collections.Counter(); ex = collections.defaultdict(list)
    for sa, n, x, top in missing:
        k = zk(x)
        if transcription(k):
            cc = "phonetic transcription"
        elif str_count[k] == 0:
            cc = "string never in our Chinese"
        elif cooc[(sa_key(sa), k)] == 0:
            cc = "in our Chinese, never with the lemma"
        else:
            cc = "co-occurs with the lemma"
        cat[cc] += 1; ex[cc].append((sa, n, x, top, cooc[(sa_key(sa), k)]))
    res["missing_categories"] = dict(cat)
    print({k: f"{v} ({100*v/len(missing):.1f}%)" for k, v in cat.most_common()})
    L = ["# Hirakawa equivalents MITRA-dict lacks: samples (seed 42)", ""]
    for cc, rows in ex.items():
        L += [f"## {cc} ({len(rows)})", ""]
        for sa, n, x, top, co in rnd.sample(rows, min(25, len(rows))):
            L.append(f"- **{sa}** (n={n}) : Hirakawa `{x}` | co-occurrences {co} | ours: " + "; ".join(f"`{g}`" for g in top))
        L.append("")
    open(os.path.join(OUT, f"hirakawa-missing-samples{TAG}.md"), "w", encoding="utf-8").write("\n".join(L))
    # headword bands
    hpairs, _ = Z.load_hirakawa(True)
    hheads = {zk(z) for _, z in hpairs if zk(z)}
    hcount = collections.Counter()
    maxh = max(len(k) for k in hheads)
    hset = hheads
    for line in open(FINAL, encoding="utf-8"):
        t = Z.zh_key(Z.zh_plain(json.loads(line).get("target", "")))
        found = set()
        for i in range(len(t)):
            for L_ in range(1, min(maxh, len(t) - i, 12) + 1):
                if t[i:i + L_] in hset:
                    found.add(t[i:i + L_])
        for k in found:
            hcount[k] += 1
    ours = collections.Counter()
    for line in open(os.path.join(C.MINED_DIR, "skt_zh_bidict.zh2sa.ndjson"), encoding="utf-8"):
        r = json.loads(line)
        ours[zk(r["zh"])] += r["n"]
    bands = [(0, 0), (1, 1), (2, 4), (5, 19), (20, 99), (100, 999), (1000, 10**9)]
    lab = lambda n: next((f"{a}" if a == z else f"{a}-{z}" if z < 10**9 else f">={a}") for a, z in bands if a <= n <= z)
    hb = collections.Counter(lab(hcount[k]) for k in hheads)
    ob = collections.Counter(lab(n) for n in ours.values())
    oin = collections.Counter(lab(n) for k, n in ours.items() if k in hheads)
    print(f"Hirakawa Chinese headwords with a usable pair: {len(hheads)} | ours zh->sa: {len(ours)}")
    for a, z in bands:
        b = lab(a)
        print(f"  {b:9} Hirakawa (occurrences in our Chinese) {hb[b]:6d} | ours (attestations) {ob[b]:6d}, of which Hirakawa headwords {oin[b]:6d}")
    res["bands"] = {"hirakawa": dict(hb), "ours": dict(ob), "ours_in_hirakawa": dict(oin),
                    "hirakawa_headwords": len(hheads), "our_headwords": len(ours)}
    json.dump(res, open(os.path.join(OUT, f"hirakawa-compare{TAG}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
