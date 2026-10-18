#!/usr/bin/env python3
"""Derive the release files of MITRA-lexicon v1 from the paper set (read-only inputs).

- sa-en CC BY-SA subset: keeps only attestations whose id starts with train/cc-by-sa/ (license field "cc-by-sa").
  Mitrasamgraha val/test rows (no license tag) and cc-by-nc-nd rows go only into the complete (CC BY-NC-ND) version.
  Sense counts, n, n_senses are recomputed; compound forms are recounted from the span rows (field `form`) of the
  retained attestations; senses/entries left without attestations are dropped.
- Alignment outputs (mined files): the lemma-encoding repair of the lexica is applied to the lemma of each token
  (toks[i][1]); sentences are untouched. Multi-part files are concatenated.

Usage: derive.py OUTDIR"""
import json, sys, os, collections
SET = "/mnt2/code/mitra-dictionaries/paper-v4-20261017-norecon"
MINED = "/mnt2/code/mitra-dictionaries/norecon-20261017"
OUT = sys.argv[1]
st = collections.Counter()

def fix(s):
    return s.replace("kﾱp", "kḷp").replace("rﾱ", "ṛ") if s and "ﾱ" in s else s

FORM = None
def load_forms():
    """(sentence id, token indices) -> surface key of the compound before stem keying (span field `form`, else `head`)."""
    global FORM
    FORM = {}
    for l in open(f"{SET}/skt_en_spans.ndjson", encoding="utf-8"):
        if not l.startswith('{"type": "phrase"'):
            continue
        d = json.loads(l)
        if d.get("head_kind") == "surface":       # sandhied-surface index of the same compound, not a case form
            continue
        k = (d["sentence_id"], tuple(d["indices"]))
        v = fix(d.get("form") or d["head"])
        if FORM.get(k, v) != v: st["form_conflict"] += 1
        FORM[k] = v

def forms_of(ex):
    return FORM.get((ex["id"], tuple(ex["indices"])), "?")

def bysa_bidict(src, dst, check_forms):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
        for l in f:
            e = json.loads(l)
            if check_forms and "forms" in e:
                if FORM is None: load_forms()
                # self-check: forms must be reproducible from the attestations of the full entry
                full = collections.Counter(forms_of(x) for s in e["senses"] for x in s["ex"])
                st["forms_check_ok" if dict(full) == e["forms"] else "forms_check_FAIL"] += 1
            senses = []
            for s in e["senses"]:
                ex = [x for x in s["ex"] if x.get("license") == "cc-by-sa"]
                if ex:
                    s = dict(s, ex=ex, count=len(ex)); senses.append(s)
            st[f"{os.path.basename(dst)}:entries_in"] += 1
            st[f"{os.path.basename(dst)}:att_full"] += e["n"]
            for s_ in e["senses"]:
                for x in s_["ex"]: st[f"{os.path.basename(dst)}:att_license={x.get('license')}"] += 1
            if not senses:
                st[f"{os.path.basename(dst)}:entries_dropped"] += 1; continue
            senses.sort(key=lambda s: -s["count"])
            e2 = {k: v for k, v in e.items() if k not in ("senses", "forms")}
            e2["n"] = sum(s["count"] for s in senses); e2["n_senses"] = len(senses)
            if "forms" in e:
                fc = collections.Counter(forms_of(x) for s in senses for x in s["ex"])
                fc.pop("?", None)                         # attestation without a compound span row: not counted
                if fc and len(fc) > 1 or e["sa"] not in fc:      # same rule as rekey_compounds.py
                    e2["forms"] = dict(fc.most_common())
            e2["senses"] = senses
            # keep the original key order: key, senses, n, n_senses, forms
            order = [k for k in e.keys() if k in e2]
            o.write(json.dumps({k: e2[k] for k in order}, ensure_ascii=False) + "\n")
            st[f"{os.path.basename(dst)}:att_kept"] += e2["n"]

def bysa_lines(src, dst, test):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
        for l in f:
            st[f"{os.path.basename(dst)}:in"] += 1
            if test(l):
                o.write(l); st[f"{os.path.basename(dst)}:kept"] += 1

def fix_mined(srcs, dst):
    with open(dst, "w", encoding="utf-8") as o:
        for src in srcs:
            for l in open(src, encoding="utf-8"):
                if "ﾱ" in l:
                    d = json.loads(l)
                    toks = d.get("toks") or []
                    if any("ﾱ" in (t[1] or "") for t in toks):
                        d["toks"] = [[t[0], fix(t[1])] + list(t[2:]) for t in toks]
                        st[f"{os.path.basename(dst)}:lemma_fixed_rows"] += 1
                        l = json.dumps(d, ensure_ascii=False) + "\n"
                o.write(l); st[f"{os.path.basename(dst)}:rows"] += 1

os.makedirs(OUT, exist_ok=True)
step = sys.argv[2] if len(sys.argv) > 2 else "all"
if step in ("all", "bidict"):
    bysa_bidict(f"{SET}/skt_en_bidict.sa2en.ndjson", f"{OUT}/sa-en.by-sa.sa2en.ndjson", True)
    bysa_bidict(f"{SET}/skt_en_bidict.en2sa.ndjson", f"{OUT}/sa-en.by-sa.en2sa.ndjson", False)
if step in ("all", "spans"):
    bysa_lines(f"{SET}/skt_en_spans.ndjson", f"{OUT}/sa-en.by-sa.spans.ndjson",
               lambda l: '"license": "cc-by-sa"' in l)
if step in ("all", "mined"):
    fix_mined([f"{MINED}/sa-en.mined.ndjson"], f"{OUT}/sa-en.alignments.ndjson")
    fix_mined([f"{MINED}/sa-bo.mined.part0", f"{MINED}/sa-bo.mined.part1"], f"{OUT}/sa-bo.alignments.ndjson")
    fix_mined([f"{MINED}/sa-zh.mined.part0", f"{MINED}/sa-zh.mined.part1"], f"{OUT}/sa-zh.alignments.ndjson")
    bysa_lines(f"{OUT}/sa-en.alignments.ndjson", f"{OUT}/sa-en.by-sa.alignments.ndjson",
               lambda l: l.startswith('{"id": "train/cc-by-sa/'))
json.dump(dict(st), open(f"{OUT}/derive_stats.{step}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(dict(st), indent=1, ensure_ascii=False))
