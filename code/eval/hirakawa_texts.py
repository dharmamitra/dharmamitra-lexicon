#!/usr/bin/env python3
"""Which Chinese texts do the Hirakawa equivalents that MITRA-dict lacks come from?

Hirakawa (in our digitization) gives no sources, so we locate each equivalent in the full Taishō canon (DharmaNexus
Chinese segments, ZH_T*). Only equivalents of two or more characters that occur in at most MAXTEXTS texts are used
(specific strings); each spreads a weight of 1 over the texts it occurs in. The same is done for the equivalents
we share with Hirakawa, as a baseline, and every text is marked by whether it is part of our Sanskrit-Chinese corpus.
Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hirakawa-texts.json and a printed ranking.
"""
import collections, glob, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import sa_key
import novel, eval_sa_zh as Z

SEG = os.path.expanduser("~/data/dharmanexus-chinese/segments")
FINAL = "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-zh/hybrid/final.ndjson"
MAXTEXTS = 20
WANT, MAXLEN = set(), 8


def text_id(fn):
    m = re.match(r"(ZH_T\d+_[0-9A-Za-z]+?)(?:_\d{3})?\.json$", fn)
    return m.group(1) if m else None


def scan(path):
    found = set()
    for s in json.load(open(path, encoding="utf-8")):
        t = Z.zh_key(Z.zh_plain(s.get("original", "")))
        for i in range(len(t)):
            for L in range(2, min(MAXLEN, len(t) - i) + 1):
                if t[i:i + L] in WANT:
                    found.add(t[i:i + L])
    return text_id(os.path.basename(path)), found


def main():
    global WANT
    cap = novel.run("sa-zh")
    zk = lambda z: Z.zh_key(Z.zh_plain(z))
    miss, match = set(), set()
    for r in cap["recs"]:
        if r.get("ref_missing") is None:
            continue
        m = {zk(x) for x in r["ref_missing"]}
        for x in cap["index"][sa_key(r["sa"])]:
            k = zk(x)
            if len(k) >= 2:
                (miss if k in m else match).add(k)
    WANT = miss | match
    files = [p for p in glob.glob(os.path.join(SEG, "ZH_T*.json")) if text_id(os.path.basename(p))]
    print(f"strings: missing {len(miss)}, matched {len(match)}; Taishō files {len(files)}", flush=True)
    occ = collections.defaultdict(set)
    with Pool(C.NPROC) as pool:
        for tid, found in pool.imap_unordered(scan, files, chunksize=20):
            for k in found:
                occ[k].add(tid)
    ours = set()
    for line in open(FINAL, encoding="utf-8"):
        i = line.find('"tgt_segnr": ["')
        ours.add(text_id(line[i + 15:line.find(":", i + 15)] + ".json") or "")
    meta = {}
    try:
        for f in json.load(open(os.path.join(os.path.dirname(SEG), "ZH_files.json"), encoding="utf-8")):
            tid = text_id(f.get("filename", "") + ".json") or text_id(f.get("textname", "") + ".json")
            if tid and tid not in meta:
                meta[tid] = f.get("displayName") or f.get("textname")
    except Exception:
        pass
    res = {}
    for label, pool_ in (("missing", miss), ("matched", match)):
        w = collections.Counter()
        n_none = n_spec = 0
        for k in pool_:
            ts = occ.get(k, set())
            if not ts:
                n_none += 1
                continue
            if len(ts) <= MAXTEXTS:
                n_spec += 1
                for t in ts:
                    w[t] += 1 / len(ts)
        tot = sum(w.values())
        inours = sum(v for t, v in w.items() if t in ours)
        res[label] = dict(strings=len(pool_), not_in_taisho=n_none, specific=n_spec,
                          share_weight_in_our_corpus=inours / tot if tot else None,
                          top=[(t, meta.get(t, ""), round(v, 1), t in ours) for t, v in w.most_common(40)])
        print(f"\n== {label}: {len(pool_)} strings (2+ chars); not in the Taishō {n_none} ({100*n_none/len(pool_):.1f}%); "
              f"specific (≤{MAXTEXTS} texts) {n_spec}; weight falling on texts in our sa-zh corpus: {100*inours/tot:.1f}%")
        acc = 0
        for t, v in w.most_common(30):
            acc += v
            print(f"   {t:16} {v:7.1f} {100*acc/tot:5.1f}%  {'OURS' if t in ours else '    '}  {meta.get(t, '')[:60]}")
    json.dump(res, open("/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/hirakawa-texts.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
