#!/usr/bin/env python3
"""Source-side counterpart to kill_stopword_glosses.py (Sanskrit / Chinese / English).

kill_stopword_glosses.py only filters the TIBETAN side -- its FUNC set is Wylie -- so the
SA_STOP / ZH_STOP / EN_STOP lists that generate_seeds.py fed the seed model had no
enforcement anywhere in the build. This adds it.

It does NOT blanket-delete the stoplists, and it does not reuse the Tibetan test as-is.
Source tokens are SUPPLIED (Sanskrit morphological analysis, Chinese segmentation, English
whitespace) rather than CHOSEN by the model, so they do not show the promiscuous-latching
failure the Tibetan filter targets; most listed stopwords here are real lexical entries
(na->mi/ma, yathaa->ji ltar, kascit->kha cig, 若->gal te, 得->thob, what->gang, may->shog).
Applying the Tibetan test unchanged kills all of those.

So a headword dies only when all three hold:
  1. promiscuous   -- >= MINGLOSS distinct glosses
  2. unfocused     -- no single gloss holds >= TOP1MAX of its mass
  3. unreciprocated-- count-weighted mean P(head | gloss) in the reverse direction < REVMIN
Test 3 is what separates a real entry with many synonyms (kascit -> kha cig / 'ga' zhig /
la la, revP 0.71) from a latch (vaa -> ldang / langs / 'byung, revP 0.11).

Usage: kill_src_stopwords.py <head_file> <head_key> <sense_key> <rev_file> <lang>
  head_file : the X->Y direction whose HEADWORD is language X  (e.g. sa2bo, head_key=sa)
  rev_file  : the Y->X direction, where X appears as a GLOSS   (e.g. bo2sa)
Env: DRY=1 report only; MINGLOSS(10), TOP1MAX(0.35), REVMIN(0.15).
Backups: .presrcstop.bak
"""
import collections, json, os, shutil, sys

HF, HK, SK, RF, LANG = sys.argv[1:6]
DRY = os.environ.get("DRY") == "1"
MINGLOSS = int(os.environ.get("MINGLOSS", "10"))
TOP1MAX = float(os.environ.get("TOP1MAX", "0.35"))
REVMIN = float(os.environ.get("REVMIN", "0.15"))

# Verbatim from generate_seeds.py's SA_STOP / ZH_STOP / EN_STOP.
STOP = {
    "sa": set("ca vā na hi tu eva api iti atha khalu punar sma ha cet tathā yathā tatra "
              "tadā ataḥ kim cit cid tat sa te ye yad idam etad kaścit".split()),
    "zh": set("之 其 者 而 也 矣 於 于 以 為 是 故 所 若 乃 即 亦 復 已 耳 焉 哉 乎 與 及 則 "
              "但 更 皆 悉 諸 等 中 時 當 應 得 有 無 不".split()),
    "en": set("the a an of to in and or is are was were be been that this these those it its "
              "by with for as on at from into then thus so such which who whom what when will "
              "shall would should may might have has had do does not".split()),
}[LANG]
norm = (lambda h: h.lower()) if LANG == "en" else (lambda h: h)


def snapshot(p):
    """Prefer the pre-noise-clean copy for statistics (fuller distribution), matching
    kill_stopword_glosses.py's convention."""
    return p + ".prenoise.bak" if os.path.exists(p + ".prenoise.bak") else p


fwd = collections.defaultdict(collections.Counter)
for l in open(snapshot(HF), encoding="utf-8"):
    d = json.loads(l)
    if norm(d[HK]) in STOP:
        for s in d["senses"]:
            fwd[d[HK]][s[SK]] += s["count"]

rev = {}
for l in open(snapshot(RF), encoding="utf-8"):
    d = json.loads(l)
    rev[d[SK]] = {s[HK]: s["count"] for s in d["senses"]}

kill, report = set(), []
for h, c in fwd.items():
    tot = sum(c.values())
    top, tn = c.most_common(1)[0]
    top1 = tn / tot
    revp = 0.0
    for g, cnt in c.items():
        r = rev.get(g)
        if r:
            revp += (cnt / tot) * (r.get(h, 0) / max(sum(r.values()), 1))
    doomed = len(c) >= MINGLOSS and top1 < TOP1MAX and revp < REVMIN
    if doomed:
        kill.add(h)
    report.append((doomed, h, tot, len(c), top1, revp, [x[0] for x in c.most_common(3)]))

print(f"[{LANG}] {os.path.basename(HF)}: {len(fwd)} stopword headwords "
      f"({sum(r[2] for r in report)} occurrences) -> killing {len(kill)}")
for doomed, h, tot, ng, t1, rp, g in sorted(report, key=lambda x: -x[2]):
    if doomed or tot > 500:
        print(f"   {'KILL' if doomed else 'keep'} {h!r:10} n={tot:<7} gl={ng:<4} "
              f"top1={t1:.2f} revP={rp:.2f}  {g}")
# Persist the kill set so the webapp-evidence stage can apply the same filter: the
# spans -> *_combined.ndjson path bypasses the bidict cleaners entirely, so without
# this the app keeps serving headwords the dictionary has dropped.
KILLDIR = os.path.join(os.path.dirname(os.path.abspath(HF)), "killlists")
os.makedirs(KILLDIR, exist_ok=True)
tag = os.path.basename(HF).replace(".ndjson", "")
json.dump(sorted(kill), open(os.path.join(KILLDIR, f"src.{tag}.{LANG}.json"), "w",
                             encoding="utf-8"), ensure_ascii=False)
print(f"   kill list -> killlists/src.{tag}.{LANG}.json")
if DRY:
    print("   [dry run - bidicts unchanged; kill list written]")
    sys.exit(0)
if not kill:
    print("   [nothing to kill]")
    sys.exit(0)


def backup(path):
    # Overwrites, like clean_rank_noise.py / kill_stopword_glosses.py do: the pipeline is
    # rebuild-oriented (build_*_bidict.py regenerates from spans, then each cleaner
    # re-snapshots), so a stale backup is worse than a refreshed one. The true original
    # is always .prehybrid.bak.
    shutil.copyfile(path, path + ".presrcstop.bak")


def apply_head(path, hkey):
    backup(path)
    drop = tot = 0
    with open(path + ".tmp", "w", encoding="utf-8") as o:
        for l in open(path, encoding="utf-8"):
            d = json.loads(l); tot += 1
            if d[hkey] in kill:
                drop += 1; continue
            o.write(l)
    os.replace(path + ".tmp", path)
    print(f"  {os.path.basename(path)}: dropped {drop}/{tot} headwords")


def apply_gloss(path, skey):
    backup(path)
    drop = tot = 0
    with open(path + ".tmp", "w", encoding="utf-8") as o:
        for l in open(path, encoding="utf-8"):
            d = json.loads(l); tot += len(d["senses"])
            keep = [s for s in d["senses"] if s[skey] not in kill]
            drop += len(d["senses"]) - len(keep)
            if not keep:
                continue
            d["senses"] = keep
            d["n"] = sum(s["count"] for s in keep)
            d["n_senses"] = len(keep)
            o.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(f"  {os.path.basename(path)}: dropped {drop}/{tot} senses")


apply_head(HF, HK)
apply_gloss(RF, HK)
