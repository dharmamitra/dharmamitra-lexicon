"""(moved from the scratchpad 2026-10-13; corpus and dictionary from CORPUS_SA_BO / EVAL_MINED_DIR) Negi vs MITRA-dict (v3 sa-bo) headword bands (recomputed 2026-10-08; the 2026-10-06 scratch script is lost).
Negi headwords = tib_key of Negi Tibetan headwords; occurrence in our Tibetan = sentence pairs of sa-bo final whose
target contains the key as a contiguous syllable sequence (<= 12 syllables; upper bound)."""
import sys, json, re, collections
sys.path.insert(0, "/home/sebastian/papers/mitra-dict-2026/build/eval")
from multiprocessing import Pool
from tibetan import tib_key
import negi
pairs, _ = negi.load_negi(tib_key)
NK = {p[1] for p in pairs if p[1]}
NK12 = {tuple(k.split()) for k in NK if 1 <= len(k.split()) <= 12}
print("Negi headword keys", len(NK), "usable", len(NK12), flush=True)
def syl(t): return [x for x in re.split(r"[\s/_|]+", t) if x]
def work(line):
    d = json.loads(line); s = syl(d["target"]); found = set()
    for i in range(len(s)):
        for L in range(1, min(12, len(s) - i) + 1):
            g = tuple(s[i:i + L])
            if g in NK12: found.add(g)
    return found
cnt = collections.Counter()
with Pool(110) as pool:
    for f in pool.imap_unordered(work, open(__import__("os").environ["CORPUS_SA_BO"], encoding="utf-8"), chunksize=500):
        cnt.update(f)
band = lambda n: "0" if n == 0 else "1" if n == 1 else "2-4" if n <= 4 else "5-19" if n <= 19 else ">=20"
nb = collections.Counter(band(cnt.get(k, 0)) for k in NK12)
print("Negi headwords by occurrence in our Tibetan:", {k: f"{v} ({100*v/len(NK12):.0f}%)" for k, v in sorted(nb.items())})
ob = collections.Counter(); five = 0; ge20 = ge20n = tot = 0
for l in open(__import__("os").environ["EVAL_MINED_DIR"] + "/skt_tib_bidict.bo2sa.ndjson", encoding="utf-8"):
    d = json.loads(l); tot += 1; n = d["n"]; ob[band(n)] += 1
    five += len(syl(d["bo"])) >= 5
    if n >= 20:
        ge20 += 1; ge20n += tib_key(d["bo"]) in NK
print(f"ours bo2sa headwords {tot}; attested <=4: {100*(ob['1']+ob['2-4'])/tot:.0f}%; >=5 syllables {100*five/tot:.0f}%; >=20 attestations {ge20}, of which Negi headwords {100*ge20n/ge20:.0f}%")
print(f"ratio to Negi headwords (78,972): {tot/78972:.1f}")
