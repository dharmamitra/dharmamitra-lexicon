#!/usr/bin/env python3
"""zh-bo span builder. Chinese compounds are based on the ACTUAL Chinese string
(par_string / src_original), NOT the reordered par_zh_segmented: segmented tokens
are aligned back to the real Chinese via difflib, and only tokens that are truly
ADJACENT in the real Chinese are merged -> the head is always a real substring of
the actual Chinese. Tibetan gloss = the covering Tibetan span (length-guarded).
Sub-units are emitted back-to-front for searchability. Shad stripped & salvaged.
"""
import json, difflib

BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries"
FINAL = __import__("os").environ.get("SPANS_IN") or f"{BASE}/zh-bo/hybrid/final.ndjson"
SPANS = __import__("os").environ.get("SPANS_OUT") or f"{BASE}/zh-bo/hybrid/zh-bo.spans.ndjson"
MAXSUB = 6
n_tok = n_phr = n_fail = n_span = 0


def trim_shad(target, s, e):
    while s < e and target[s] in "/| \t": s += 1
    while e > s and target[e - 1] in "/| \t": e -= 1
    cut = min([p for p in (target.find("/", s, e), target.find("|", s, e)) if p != -1], default=-1)
    if cut != -1:
        e = cut
        while e > s and target[e - 1] in " \t": e -= 1
    return (s, e) if e > s else None


def offsets(target, aligns, ntok):
    global n_fail, n_span
    cur = 0; out = {}
    for k, span in aligns.items():
        if not (1 <= int(k) <= ntok):
            continue
        n_span += 1
        p = target.find(span, cur)
        if p < 0:
            p = target.find(span)
            if p < 0:
                n_fail += 1; continue
        out[int(k)] = (span, p, p + len(span)); cur = p + len(span)
    return out


def align_to_chinese(toks, par):
    # map each segmented token to a [start,end) char span in the REAL Chinese (par)
    surfs = [t[0].strip() for t in toks]
    sm = difflib.SequenceMatcher(None, " ".join(surfs), par, autojunk=False)
    jr = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                jr[i1 + k] = j1 + k
    spans = []; pos = 0
    for s in surfs:
        rs = [jr[k] for k in range(pos, pos + len(s)) if k in jr]
        spans.append((min(rs), max(rs) + 1) if rs else None); pos += len(s) + 1
    return spans


def sub_runs(run):
    out = set(); L = len(run)
    for a in range(L):
        for b in range(a + 1, L):
            if b - a + 1 > MAXSUB:
                break
            out.add(tuple(run[a:b + 1]))
    return out


seen_pairs = set()
with open(SPANS, "w", encoding="utf-8") as o:
    for line in open(FINAL, encoding="utf-8"):
        r = json.loads(line)
        key = (r["id"], r["target"], tuple(tuple(t) for t in r["toks"]))  # drop exact-duplicate pairs
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        toks = r["toks"]; tgt = r["target"]; par = r["src_original"]
        off = offsets(tgt, r["aligns"], len(toks))
        loc = {"sentence_id": r["id"], "src_segnr": r["src_segnr"], "tgt_segnr": r["tgt_segnr"],
               "source": r["source"], "target": tgt, "src_sentence": par}
        cspan = align_to_chinese(toks, par)
        # token spans (head = the token's actual Chinese substring when alignable, else the word)
        for n in sorted(off):
            t = trim_shad(tgt, off[n][1], off[n][2])
            if not t: continue
            s, e = t
            zh = par[cspan[n - 1][0]:cspan[n - 1][1]] if cspan[n - 1] else toks[n - 1][0]
            o.write(json.dumps({"type": "token", "head": zh, "zh": zh, "bo": tgt[s:e],
                                "indices": [n], "start": s, "end": e, **loc}, ensure_ascii=False) + "\n"); n_tok += 1
        # phrase spans: group aligned tokens that are ADJACENT in the real Chinese
        aligned = sorted((n for n in off if cspan[n - 1]), key=lambda n: cspan[n - 1][0])
        runs = []; i = 0
        while i < len(aligned):
            run = [aligned[i]]; j = i
            while j + 1 < len(aligned) and cspan[aligned[j + 1] - 1][0] == cspan[run[-1] - 1][1]:
                run.append(aligned[j + 1]); j += 1
            if len(run) >= 2:
                runs.append(run)
            i = j + 1
        for run in runs:
            for sub in sub_runs(run):
                cs = cspan[sub[0] - 1][0]; ce = cspan[sub[-1] - 1][1]
                head = par[cs:ce]
                if not head or any(c in head for c in " \t，。、！？；："):
                    continue
                ts = min(off[n][1] for n in sub); te = max(off[n][2] for n in sub)
                t = trim_shad(tgt, ts, te)
                if not t:
                    continue
                bs, be = t
                exp = sum(off[n][2] - off[n][1] for n in sub)
                if be - bs > 3 * max(exp, 1):    # Tibetan scattered -> not a coherent gloss
                    continue
                o.write(json.dumps({"type": "phrase", "head": head, "zh": head, "bo": tgt[bs:be],
                                    "indices": list(sub), "start": bs, "end": be, **loc},
                                   ensure_ascii=False) + "\n"); n_phr += 1

print(f"token spans: {n_tok}   phrase spans: {n_phr}")
print(f"cursor-match failures: {n_fail}/{n_span} ({100*n_fail/max(n_span,1):.3f}%)")
print(f"-> {SPANS}")
