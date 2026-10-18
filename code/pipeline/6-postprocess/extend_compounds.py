#!/usr/bin/env python3
"""Form compound entries that the span builders miss for two mechanical reasons (2026-10-17, user decision).

R1 (all pairs)  untagged formative: a compound whose untagged members are final members from a closed list of formatives
                (SUFFIX: -tva, -tā, -mātra), which the model mostly leaves untagged, and whose tagged members form one unit
                (a single tagged member, an existing phrase over exactly those members, or, for Chinese, pieces joined
                as in R2). Example: duḥkha-nirantara-tva : 苦受無間 (before: only duḥkha-nirantara). Where the target
                renders the formative right after the span, the span is extended over it: Tibetan nyid (also "pa/ba nyid")
                for -tva/-tā and tsam for -mātra, Chinese 性 for -tva/-tā (chos nyid for dharma-tā, not chos).
                Hand check 2026-10-17: -ādi/-ādika (text has 等 that the model left untagged; ādi-buddha) and -vat (gloss
                lacks "like"/如) gave too narrow glosses and are not used.
R2 (sa-zh only) enumeration comma: at least three members, every member tagged, and the Chinese pieces separated only by 、
                (時節、處所、勝負差別 for kāla-deśa-prakarṣa-bheda), which the builder rejects as punctuation. Two-member
                cases were mostly pieces of different list items (母、殺 from 殺母、殺父) and are not used.
The new phrase spans the target from the first to the last tagged piece (the gloss is that verbatim slice, as in the
builders; English lower-cased as en_key), head = desandhied surface of the full compound, lemma = member lemmas, and
carries "rule": "suffix" | "enum". Full compounds only (as drop_subcompounds.py): a phrase span of the same sentence whose
indices are a proper subset of the new compound is removed, together with its attestations; a sandhied surface span over
such a subset is moved to the new indices. Nothing is added where a phrase already covers the compound or more.
Counts are recomputed (count == len(ex)). Records (token lists) come from the mined rows of each pair (env REC_<pfx>).

Usage: extend_compounds.py SRC_DIR DST_DIR     (ONLY_PAIRS=..., REC_skt_zh=..., REC_skt_tib=..., REC_skt_en=...)
"""
import json, os, sys, ast, re, collections, unicodedata
from multiprocessing import Pool

SRC, DST = sys.argv[1], sys.argv[2]
M = "/mnt2/code/mitra-dictionaries"
PAIRS = [("skt_en", "en"), ("skt_tib", "bo"), ("skt_zh", "zh")]
if os.environ.get("ONLY_PAIRS"):
    PAIRS = [p for p in PAIRS if p[0] in os.environ["ONLY_PAIRS"].split(",")]
REC = {"skt_zh": os.environ.get("REC_skt_zh", f"{M}/qwen-dictionaries-v3/sa-zh/hybrid/final.ndjson"),
       "skt_tib": os.environ.get("REC_skt_tib", f"{M}/qwen-dictionaries-v3/sa-bo/hybrid/final.ndjson"),
       "skt_en": os.environ.get("REC_skt_en", f"{M}/qwen-dictionaries/sa-en/mined.ndjson")}
SUFFIX = {"tva", "tā", "mātra"}
EXT = {"bo": {"tva": re.compile(r"^[ _]?(?:(?:pa|ba) )?nyid(?![a-z'])"), "tā": re.compile(r"^[ _]?(?:(?:pa|ba) )?nyid(?![a-z'])"),
              "mātra": re.compile(r"^[ _]?tsam(?![a-z'])")},
       "zh": {"tva": re.compile(r"^性"), "tā": re.compile(r"^性")}}
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
norm = lambda s: unicodedata.normalize("NFC", s).replace("ṁ", "ṃ").replace("Ṁ", "Ṃ")


def en_key(s):
    s = " ".join(s.split()).lower()
    return s[:-1] if s.endswith(",") else s


def chains(T):
    i = 0
    while i < len(T):
        j = i
        while j < len(T) - 1 and T[j][0].endswith("-"):
            j += 1
        if j > i:
            yield list(range(i + 1, j + 2))
        i = j + 1


def process_pair(pfx, tag):
    st = collections.Counter()
    tok = collections.defaultdict(dict)                  # sid -> idx -> token span record
    phr = collections.defaultdict(dict)                  # sid -> indices -> phrase span (non-surface)
    sur = collections.defaultdict(dict)                  # sid -> indices -> surface span
    for l in open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8"):
        r = json.loads(l); sid = r["sentence_id"]; ix = tuple(r["indices"])
        if r["type"] == "token": tok[sid][ix[0]] = r
        elif r.get("head_kind") == "surface": sur[sid][ix] = r
        else: phr[sid][ix] = r
    new, drop, move, moved_to = [], set(), {}, set()     # new phrase spans; (head, sid, ix) dropped; (head, sid, ix) -> new ix
    for l in open(REC[pfx], encoding="utf-8"):
        d = json.loads(l); sid = d["id"]
        if sid not in tok:
            continue
        T = lit(d["toks"]); A = tok[sid]; tgt = d["target"]
        for ch in chains(T):
            al = [k for k in ch if k in A]; un = [k for k in ch if k not in A]
            if not al or any(set(ch) <= set(ix) for ix in phr[sid]):
                continue                                 # nothing tagged, or the compound (or more) is already an entry
            pieces = sorted((A[k]["start"], A[k]["end"]) for k in al)
            gaps = [tgt[pieces[x][1]:pieces[x + 1][0]] for x in range(len(pieces) - 1)]
            enum = tag == "zh" and len(al) >= 3 and all(g in ("", "、") for g in gaps)
            if un:
                if not all(T[k - 1][1] in SUFFIX for k in un):
                    continue
                if al != ch[:len(al)]:
                    continue                             # untagged formatives must be final members
                if not (len(al) == 1 or tuple(al) in phr[sid] or enum):
                    continue
                rule = "suffix"
            else:
                if not enum:
                    continue
                rule = "enum"
            s, e = pieces[0][0], pieces[-1][1]
            if un and tag in EXT:
                rx = EXT[tag].get(T[un[0] - 1][1])
                m = rx.match(tgt[e:]) if rx else None
                if m: e += m.end(); st["formative_rendered"] += 1
            base = phr[sid].get(tuple(al)) or A[al[0]]
            r = {k: v for k, v in base.items() if k not in ("type", "head", "lemma", "sanskrit", tag, "indices", "start", "end", "rule", "head_kind")}
            head = norm("".join(T[k - 1][0] for k in ch))
            r.update({"type": "phrase", "head": head, "lemma": norm("-".join(T[k - 1][1] for k in ch)), "sanskrit": head,
                      tag: tgt[s:e], "indices": ch, "start": s, "end": e, "rule": rule})
            new.append(r); st[f"new_{rule}"] += 1
            if un: st.update(f"suffix_{T[k - 1][1]}" for k in un)
            for ix in list(phr[sid]):
                if set(ix) < set(ch):
                    drop.add((phr[sid][ix]["head"], sid, ix)); st["phrase_dropped"] += 1
            for ix in list(sur[sid]):
                if set(ix) < set(ch):
                    if (sid, tuple(ch)) in moved_to: drop.add((sur[sid][ix]["head"], sid, ix)); st["surface_dropped"] += 1
                    else: move[(sur[sid][ix]["head"], sid, ix)] = tuple(ch); moved_to.add((sid, tuple(ch))); st["surface_moved"] += 1
    # ---- spans
    with open(f"{SRC}/{pfx}_spans.ndjson", encoding="utf-8") as f, open(f"{DST}/{pfx}_spans.ndjson", "w", encoding="utf-8") as o:
        for l in f:
            if '"phrase"' in l:
                r = json.loads(l); k = (r["head"], r["sentence_id"], tuple(r["indices"]))
                if r["type"] == "phrase" and k in drop:
                    continue
                if r["type"] == "phrase" and k in move:
                    r["indices"] = list(move[k]); l = json.dumps(r, ensure_ascii=False) + "\n"
            o.write(l)
        for r in new:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    # ---- dictionaries
    gl = (lambda s: en_key(s)) if tag == "en" else (lambda s: s)

    def ex_of(r):
        x = {"id": r["sentence_id"]}
        if tag == "en":
            x.update(target=r["target"], start=r["start"], end=r["end"], src=r.get("src_sentence", ""), license=r.get("license", ""))
            if r.get("low_coverage"): x["low_coverage"] = True
        else:
            x.update(tgt_segnr=r["tgt_segnr"], target=r["target"], start=r["start"], end=r["end"])
        x["indices"] = r["indices"]
        return x

    add_fw = collections.defaultdict(lambda: collections.defaultdict(list))
    add_rv = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in new:
        g = gl(r[tag]); x = ex_of(r)
        if not g: continue
        add_fw[r["head"]][g].append((x, r["rule"])); add_rv[g][r["head"]].append((x, r["rule"]))

    def fix(senses, key, head):
        out = []
        for s in senses:
            ex = []; h = head if head is not None else s["sa"]
            for x in s["ex"]:
                k = (h, x["id"], tuple(x.get("indices") or ()))
                if k in drop: st["att_dropped"] += 1; continue
                if k in move: x = dict(x, indices=list(move[k]))
                ex.append(x)
            if ex:
                s["ex"] = ex; s["count"] = len(ex); out.append(s)
        return out

    def merge(e, senses, key, adds):
        idx = {s[key]: s for s in senses}
        for g, xs in adds.items():
            if g not in idx:
                idx[g] = {key: g, "count": 0, "ex": []}; senses.append(idx[g])
            idx[g]["ex"] += [x for x, _ in xs]
            if tag == "zh": idx[g]["relaxed"] = idx[g].get("relaxed", 0) + sum(1 for _, ru in xs if ru == "enum")
        for s in senses:
            s["ex"].sort(key=lambda x: len(x.get("target", ""))); s["count"] = len(s["ex"])
        senses.sort(key=lambda s: -s["count"])
        e["senses"] = senses; e["n"] = sum(s["count"] for s in senses); e["n_senses"] = len(senses)
        return e

    for d1, k1, k2, adds in ((f"sa2{tag}", "sa", tag, add_fw), (f"{tag}2sa", tag, "sa", add_rv)):
        out = []; seen = set()
        for l in open(f"{SRC}/{pfx}_bidict.{d1}.ndjson", encoding="utf-8"):
            e = json.loads(l); h = e[k1]; seen.add(h)
            senses = fix(e["senses"], k2, h if k1 == "sa" else None)
            e = merge(e, senses, k2, adds.get(h, {}))
            if e["senses"]: out.append(e)
            else: st[f"{d1}_headwords_emptied"] += 1
        for h in adds:
            if h not in seen:
                out.append(merge({k1: h}, [], k2, adds[h])); st[f"{d1}_headwords_new"] += 1
        out.sort(key=lambda e: -e["n"])
        with open(f"{DST}/{pfx}_bidict.{d1}.ndjson", "w", encoding="utf-8") as o:
            for e in out:
                o.write(json.dumps(e, ensure_ascii=False) + "\n")
        st[f"{d1}_headwords"] = len(out); st[f"{d1}_att"] = sum(e["n"] for e in out)
    return pfx, dict(st)


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with Pool(len(PAIRS)) as p:
        res = dict(p.starmap(process_pair, PAIRS))
    done = {f"{pfx}_spans.ndjson" for pfx, _ in PAIRS} | {f"{pfx}_bidict.{a}.ndjson" for pfx, t in PAIRS for a in (f"sa2{t}", f"{t}2sa")}
    for f in os.listdir(SRC):
        if f not in done and os.path.isfile(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            os.link(f"{SRC}/{f}", f"{DST}/{f}")
    if os.path.isdir(f"{SRC}/pre-mdc-20260919") and not os.path.exists(f"{DST}/pre-mdc-20260919"):
        __import__("shutil").copytree(f"{SRC}/pre-mdc-20260919", f"{DST}/pre-mdc-20260919", symlinks=True)
    json.dump(res, open(f"{DST}/extend_compounds_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
