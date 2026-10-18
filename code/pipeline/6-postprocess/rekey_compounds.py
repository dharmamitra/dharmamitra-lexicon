#!/usr/bin/env python3
"""Key compound entries on their stem form (2026-10-17, user decision).

The span builders key a compound phrase on its desandhied surface, i.e. with the case ending of the last member
(kuśala-mūlāni, kuśala-mūlam, kuśala-mūlaiḥ, ... are separate headwords; for compounds with >= 20 attestations the
largest form holds only 50-55% of the attestations). This re-keys every hyphenated compound attestation on
  non-final members as they stand in the compound + the stem of the final member
where the stem is the ByT5/DCS lemma of the final member (mūlāni -> mūla, yaśāḥ -> yaśas, vihārī -> vihārin), except when
that lemma is a DCS verb (participles are lemmatised to the root: heyāni -> hā, varjitam -> varjay); then the a-stem ending
is stripped from the surface (heyāni -> heya, varjitam -> varjita). Non-compound entries, sandhied surface entries
(head_kind "surface") and collocations with spaces are unchanged.

Mapping is per attestation: (sentence id, Sanskrit token indices) of every attestation is looked up in the spans file.
Entries and senses that become identical are merged (counts = len(ex), attestations re-sorted by length of the
translation, senses re-ranked by count, ties in first-seen order); every re-keyed sa->X entry gets
"forms": {surface key: attestations}. Spans: `head` of the re-keyed phrase spans becomes the stem key, the old head is
kept as `form`. Other files are hard-linked.

Usage: rekey_compounds.py SRC_DIR DST_DIR      (ONLY_PAIRS=skt_zh,... to restrict)
"""
import json, os, re, sys, collections
from multiprocessing import Pool

SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]
VERBS = set(open(os.path.join(HERE, "dcs_verbs.txt")).read().split())
A_END = sorted("ebhyaḥ ābhyām ānām eṣu ayoḥ asya āni āya ena aiḥ āt āḥ aḥ am au e ā o".split(), key=len, reverse=True)


def is_compound_head(h):
    return "-" in h and " " not in h


def stem_key(head, lemma):
    hm = [m for m in head.split("-") if m]
    lm = [m for m in lemma.split("-") if m]
    if len(hm) < 2 or not lm:          # malformed heads ('--rāja') stay as they are
        return head
    last_s, last_l = hm[-1], lm[-1]
    if last_l in VERBS:                       # participle / verbal noun lemmatised to the root
        stem = last_s
        for e in A_END:
            if last_s.endswith(e) and len(last_s) - len(e) >= 2:
                stem = last_s[:-len(e)] + "a"; break
    else:
        stem = last_l
    return "-".join(hm[:-1] + [stem])


def span_map(pfx):
    """(sentence_id, indices) -> stem key, and head -> Counter(stem key) as fallback."""
    m, byhead = {}, collections.defaultdict(collections.Counter)
    with open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f:
        for l in f:
            if '"phrase"' not in l:
                continue
            r = json.loads(l)
            if r.get("type") != "phrase" or r.get("head_kind") == "surface" or not is_compound_head(r["head"]):
                continue
            k = stem_key(r["head"], r.get("lemma") or r["head"])
            m[(r["sentence_id"], tuple(r["indices"]))] = k
            byhead[r["head"]][k] += 1
    return m, {h: c.most_common(1)[0][0] for h, c in byhead.items()}


def rewrite_spans(pfx):
    st = collections.Counter()
    with open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f, open(f"{DST}/{pfx}_spans.ndjson", "w", encoding="utf-8") as o:
        for l in f:
            if '"phrase"' in l:
                r = json.loads(l)
                if r.get("type") == "phrase" and r.get("head_kind") != "surface" and is_compound_head(r["head"]):
                    k = stem_key(r["head"], r.get("lemma") or r["head"])
                    if k != r["head"]:
                        r["form"] = r["head"]; r["head"] = k; st["spans_rekeyed"] += 1
                    l = json.dumps(r, ensure_ascii=False) + "\n"
            o.write(l)
    return st


def finish_senses(senses):
    for s in senses:
        s["ex"].sort(key=lambda x: len(x.get("target", "")))      # stable: ties keep corpus order
        s["count"] = len(s["ex"])
    senses.sort(key=lambda s: -s["count"])
    return senses


def process_pair(pfx, tag):
    M, H = span_map(pfx)
    st = collections.Counter()

    def key_of(sa, x):
        k = M.get((x["id"], tuple(x.get("indices") or ())))
        if k is None:
            k = H.get(sa); st["att_fallback_head"] += 1 if k else 0; st["att_unmapped"] += 0 if k else 1
            k = k or sa
        return k

    # ---- Sanskrit -> X
    E, order = {}, []
    for l in open(f"{SRC}/{pfx}_bidict.sa2{tag}.ndjson", encoding="utf-8"):
        e = json.loads(l); sa = e["sa"]; st["sa_in"] += 1
        groups = collections.defaultdict(list)
        if is_compound_head(sa):
            for s in e["senses"]:
                for x in s["ex"]:
                    groups[key_of(sa, x)].append((s, x))
        else:
            for s in e["senses"]:
                for x in s["ex"]:
                    groups[sa].append((s, x))
        for k, items in groups.items():
            if k not in E:
                E[k] = {"sa": k, "senses": {}, "forms": collections.Counter()}; order.append(k)
            T = E[k]
            for s, x in items:
                g = s[tag]
                if g not in T["senses"]:
                    T["senses"][g] = {kk: vv for kk, vv in s.items() if kk not in ("ex", "count", "relaxed")}
                    T["senses"][g]["ex"] = []
                    if "relaxed" in s: T["senses"][g]["relaxed"] = 0
                T["senses"][g]["ex"].append(x)
                if is_compound_head(sa): T["forms"][sa] += 1
            for s in e["senses"]:          # carry relaxed counts proportionally to the attestations moved
                if "relaxed" in s and s["relaxed"]:
                    moved = sum(1 for ss, _ in items if ss is s)
                    if moved and "relaxed" in T["senses"].get(s[tag], {}):
                        T["senses"][s[tag]]["relaxed"] += round(s["relaxed"] * moved / max(len(s["ex"]), 1))
    out = []
    for k in order:
        T = E[k]; senses = finish_senses(list(T["senses"].values()))
        e = {"sa": k, "senses": senses, "n": sum(s["count"] for s in senses), "n_senses": len(senses)}
        if T["forms"] and (len(T["forms"]) > 1 or k not in T["forms"]):
            e["forms"] = dict(T["forms"].most_common())
        out.append(e)
    out.sort(key=lambda e: -e["n"])
    with open(f"{DST}/{pfx}_bidict.sa2{tag}.ndjson", "w", encoding="utf-8") as o:
        for e in out:
            o.write(json.dumps(e, ensure_ascii=False) + "\n")
    st["sa_out"] = len(out)
    st["att_sa2x"] = sum(e["n"] for e in out)

    # ---- X -> Sanskrit
    nrev = att = 0
    with open(f"{SRC}/{pfx}_bidict.{tag}2sa.ndjson", encoding="utf-8") as f, \
            open(f"{DST}/{pfx}_bidict.{tag}2sa.ndjson", "w", encoding="utf-8") as o:
        for l in f:
            e = json.loads(l); S, so = {}, []
            for s in e["senses"]:
                sa = s["sa"]
                for x in s["ex"]:
                    k = key_of(sa, x) if is_compound_head(sa) else sa
                    if k not in S:
                        S[k] = {kk: vv for kk, vv in s.items() if kk not in ("ex", "count", "relaxed")}
                        S[k]["sa"] = k; S[k]["ex"] = []; so.append(k)
                    S[k]["ex"].append(x)
            senses = finish_senses([S[k] for k in so])
            e["senses"] = senses; e["n"] = sum(s["count"] for s in senses); e["n_senses"] = len(senses)
            nrev += 1; att += e["n"]
            o.write(json.dumps(e, ensure_ascii=False) + "\n")
    st["rev"] = nrev; st["att_x2sa"] = att
    st.update(rewrite_spans(pfx))
    return pfx, dict(st)


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with Pool(len(PAIRS)) as p:
        res = dict(p.starmap(process_pair, PAIRS))
    done = {f"{pfx}_{s}" for pfx, _ in PAIRS for s in ("spans.ndjson",)} | \
           {f"{pfx}_bidict.{a}.ndjson" for pfx, t in PAIRS for a in (f"sa2{t}", f"{t}2sa")}
    for f in os.listdir(SRC):
        if f not in done and os.path.isfile(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            os.link(f"{SRC}/{f}", f"{DST}/{f}")
    if os.path.isdir(f"{SRC}/pre-mdc-20260919") and not os.path.exists(f"{DST}/pre-mdc-20260919"):
        __import__("shutil").copytree(f"{SRC}/pre-mdc-20260919", f"{DST}/pre-mdc-20260919", symlinks=True)   # relative links
    json.dump(res, open(f"{DST}/rekey_compounds_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
