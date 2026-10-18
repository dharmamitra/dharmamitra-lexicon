#!/usr/bin/env python3
"""Remove Tibetan grammatical 'stopword' spans (yod do, yin pa, ste, rnams ...) that
survive PMI because the model misaligns them the same way across many lemmas. A span
is a stopword gloss iff FUNCTION-SYLLABLE-ONLY and spread thin (no single head >=70%)
and promiscuous (>=10 heads). Keeps concentrated correspondences (iti->zhes, 0.94).

Usage: kill_stopword_glosses.py <gloss_file> <gloss_head_key> <gloss_sense_key> <head_file> <head_key>
  gloss_file: the X->Tibetan direction (senses' gloss is Tibetan)
  head_file : the Tibetan->X direction (headword is Tibetan)
Backups: .prestop.bak
"""
import json, collections, shutil, os, sys

GF, GHK, GSK, HF, HK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
FUNC = {'ni','na','no','nas','su','tu','du','ru','la','las','dang','gi','gis','gyi','gyis','kyi','kyis',
        'yi','yis',"'i",'pa',"pa'i",'ba',"ba'i",'po','mo','par','bar','pas','bas','ste','te','de','cing',
        'zhing','shing','ces','zhes','yin','yod','med','do',"'o",'so','to','ro','go','ngo','lo',"'am",
        "'ang",'yang','kyang','zhig','cig','gcig','rnams','dag','can','nyid','tsam',"'di",'der','dir'}


def func_only(b):
    toks = b.replace("/", " ").replace("|", " ").split()
    return bool(toks) and all(t in FUNC for t in toks)


src = GF + ".prenoise.bak" if os.path.exists(GF + ".prenoise.bak") else GF
dist = collections.defaultdict(collections.Counter)
for l in open(src, encoding="utf-8"):
    d = json.loads(l)
    for s in d["senses"]:
        dist[s[GSK]][d[GHK]] += s["count"]

kill = set()
for bo, c in dist.items():
    if not func_only(bo):
        continue
    tot = sum(c.values())
    if len(c) >= 10 and c.most_common(1)[0][1] / tot < 0.70:
        kill.add(bo)
top = sorted(kill, key=lambda b: -sum(dist[b].values()))[:18]
print(f"stopword glosses to kill: {len(kill)}  top: {', '.join(repr(b) for b in top)}")
# Persist for the webapp-evidence filter (spans bypass this cleaner entirely).
KILLDIR = os.path.join(os.path.dirname(os.path.abspath(GF)), "killlists")
os.makedirs(KILLDIR, exist_ok=True)
_tag = os.path.basename(GF).replace(".ndjson", "")
json.dump(sorted(kill), open(os.path.join(KILLDIR, f"gloss.{_tag}.bo.json"), "w",
                             encoding="utf-8"), ensure_ascii=False)
print(f"kill list -> killlists/gloss.{_tag}.bo.json")


def apply_gloss(path, skey):
    shutil.copyfile(path, path + ".prestop.bak")
    drop = tot = 0
    with open(path + ".tmp", "w", encoding="utf-8") as o:
        for l in open(path, encoding="utf-8"):
            d = json.loads(l); tot += len(d["senses"])
            keep = [s for s in d["senses"] if s[skey] not in kill]
            drop += len(d["senses"]) - len(keep)
            if not keep:
                continue
            d["senses"] = keep; d["n"] = sum(s["count"] for s in keep); d["n_senses"] = len(keep)
            o.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(f"  {os.path.basename(path)}: dropped {drop}/{tot} senses")


def apply_head(path, hkey):
    shutil.copyfile(path, path + ".prestop.bak")
    drop = tot = 0
    with open(path + ".tmp", "w", encoding="utf-8") as o:
        for l in open(path, encoding="utf-8"):
            d = json.loads(l); tot += 1
            if d[hkey] in kill:
                drop += 1; continue
            o.write(l)
    os.replace(path + ".tmp", path)
    print(f"  {os.path.basename(path)}: dropped {drop}/{tot} headwords")


apply_gloss(GF, GSK)
apply_head(HF, HK)
