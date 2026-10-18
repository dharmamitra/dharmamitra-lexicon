#!/usr/bin/env python3
"""sa-bo span builder with compound stitching.

For each sentence: offsets into the Tibetan (Wylie) target (forward cursor on the
sentence-ordered aligns; dup-id guard). Emits:
  - token spans  (head = LEMMA; the single-word dictionary)
  - phrase spans (head = DESANDHIED SURFACE form, NOT lemma) for compounds:
      * Sanskrit compound members are marked by a trailing '-' on the unsandhied
        surface (ātma-, dharma-). Consecutive members are stitched even across
        Tibetan connective/case particles (dang, gi, gyi, kyi, la, ...).
      * ALL contiguous sub-compounds are emitted so sub-terms are searchable
        (pudgala dharma nairātmya -> dharma nairātmya -> nairātmya, etc.).
Phrase Tibetan side = the verbatim target slice (incl. bridged particles).
The bidict build keys phrases on `head`, so they are fully bidirectional.
"""
import json

BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
FINAL = __import__("os").environ.get("SPANS_IN") or f"{BASE}/sa-zh/hybrid/final.ndjson"
SPANS = __import__("os").environ.get("SPANS_OUT") or f"{BASE}/sa-zh/hybrid/sa-zh.spans.ndjson"
MAXSUB = 8   # cap sub-compound length (bounds explosion on long runs)
BRIDGE = set()   # Chinese target: no connective particles between chars, merge only when adjacent
n_tok = n_phr = n_fail = n_span = 0
n_relax = __import__("collections").Counter()


import difflib
BOUND = set(" \t/|0123456789()[]#.,")


def align_tokens(toks, root):
    # map each token to a [start,end) char range in root_string via difflib over the
    # space-joined unsandhied surfaces (handles sandhi differences). Returns list.
    surfs = [t[0].strip().rstrip("-").strip() for t in toks]
    sm = difflib.SequenceMatcher(None, " ".join(surfs), root, autojunk=False)
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


def word_surface(root, sp_a, sp_b, exp_len):
    # snap the aligned [a..b] range to root word boundaries -> the actual sandhi'd
    # compound surface. Reject implausible matches (whole-sentence / wrong length).
    if not sp_a or not sp_b:
        return None
    s, e = sp_a[0], sp_b[1]
    while s > 0 and root[s - 1] not in BOUND: s -= 1
    while e < len(root) and root[e] not in BOUND: e += 1
    surf = root[s:e]
    if " " in surf or not surf or not (0.6 <= len(surf) / max(exp_len, 1) <= 1.8):
        return None
    return surf


def trim_shad(target, s, e):
    # strip leading/trailing shad + whitespace; truncate at first internal shad
    # (a span should never cross a clause boundary). Returns (s,e) or None if empty.
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


def is_member(n, toks):                      # compound member: unsandhied surface ends with '-'
    return toks[n - 1][0].rstrip().endswith("-")


def surf(n, toks):                           # desandhied surface, trailing '-' stripped
    s = toks[n - 1][0].strip()
    return s[:-1].strip() if s.endswith("-") else s


def maximal_runs(off, target, toks):
    idx = sorted(off); res = []; i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1 and off[idx[j]][2] <= off[idx[j + 1]][1]:
            gap = target[off[idx[j]][2]:off[idx[j + 1]][1]]
            if gap.strip() == "" or (is_member(idx[j], toks) and all(w in BRIDGE for w in gap.split())):
                j += 1
            else:
                break
        if j > i:
            res.append(idx[i:j + 1])
        i = j + 1
    return res


def sub_runs(run, toks):
    # all contiguous sub-runs (len 2..MAXSUB) whose non-final tokens are compound
    # members, plus the maximal run itself (collocation). Returns set of tuples.
    out = set(); L = len(run)
    for a in range(L):
        for b in range(a + 1, L):
            if b - a + 1 > MAXSUB:
                break
            if not is_member(run[b - 1], toks):   # token before the new last must be a member
                break
            out.add(tuple(run[a:b + 1]))
    out.add(tuple(run))
    return out


# ---- relaxed compound windows ("M", decided 2026-10-07; papers/mitra-dict-2026/notes/pipeline-facts.md §17) ----
# Chinese often renders a Sanskrit compound with its members reordered (pañca-bhaya-samatikrānta -> 離五怖), with a
# grammatical particle between members (nāga-rāja-paripṛcchā -> 龍王所問) or with a member left untranslated
# (cintā-maya-prajñā -> 思慧). The strict runs above reject all three. For windows of consecutive tokens whose
# non-final tokens are compound members (len 2..MAXSUB) that the strict pass did not emit, accept the window when
# the aligned Chinese pieces form one compact block in any order, with at most one 1-character gap made of a
# particle in RELAX_FUNC, at most one unaligned interior member, and no punctuation. Free gaps were rejected
# (they pull in neighbouring content words). Emitted with "rule": order | particle | unaligned | overlap.
RELAX = __import__("os").environ.get("SAZH_RELAX", "1") == "1"
RELAX_FUNC = set("之所而為其及與於諸能得等者中")
RELAX_PUNCT = __import__("re").compile(r"[\s，。、；：？！「」『』（）()《》〈〉…·,.;:!?\u3000/|]")


def relaxed_windows(off, tgt, toks, done):
    N = len(toks); out = []
    for a in range(1, N):
        for b in range(a + 1, min(N, a + MAXSUB - 1) + 1):
            if not is_member(b - 1, toks):
                break
            win = tuple(range(a, b + 1))
            if win in done:
                continue
            al = [n for n in win if n in off]
            miss = [n for n in win if n not in off]
            if len(miss) > 1 or (miss and miss[0] in (win[0], win[-1])) or len(al) < 2:
                continue
            seq = [off[n] for n in al]
            reordered = any(seq[i][2] > seq[i + 1][1] for i in range(len(seq) - 1))
            sp = sorted((x[1], x[2]) for x in seq)
            end = sp[0][1]; ok = True; gap = False; overlap = False
            for s0, e0 in sp[1:]:
                if s0 > end:
                    g = tgt[end:s0]
                    if len(g) > 1 or g not in RELAX_FUNC:
                        ok = False; break
                    gap = True
                elif s0 < end:
                    overlap = True
                end = max(end, e0)
            if not ok or RELAX_PUNCT.search(tgt[sp[0][0]:end]):
                continue
            t = trim_shad(tgt, sp[0][0], end)
            if not t:
                continue
            rule = "unaligned" if miss else "particle" if gap else "order" if reordered else "overlap"
            out.append((win, t[0], t[1], rule))
    return out


seen_pairs = set()
with open(SPANS, "w", encoding="utf-8") as o:
    for line in open(FINAL, encoding="utf-8"):
        r = json.loads(line)
        key = (r["id"], r["target"], tuple(tuple(t) for t in r["toks"]))  # drop exact-duplicate pairs
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        toks = r["toks"]; tgt = r["target"]; off = offsets(tgt, r["aligns"], len(toks))
        loc = {"sentence_id": r["id"], "src_segnr": r["src_segnr"], "tgt_segnr": r["tgt_segnr"],
               "source": r["source"], "target": tgt, "src_sentence": r["src_original"]}
        for n in sorted(off):
            _, s, e = off[n]; sf, lem = toks[n - 1]
            t = trim_shad(tgt, s, e)
            if not t: continue
            s, e = t
            o.write(json.dumps({"type": "token", "head": lem, "lemma": lem, "sanskrit": sf, "zh": tgt[s:e],
                                "indices": [n], "start": s, "end": e, **loc}, ensure_ascii=False) + "\n"); n_tok += 1
        mruns = maximal_runs(off, tgt, toks)
        root = r["src_original"]
        tokspans = align_tokens(toks, root) if any(is_member(run[0], toks) for run in mruns) else None
        for run in mruns:
            # actual word boundaries of this compound in the Sanskrit (for surface sub-strings)
            word = None
            if tokspans is not None and is_member(run[0], toks):
                sa, sb = tokspans[run[0] - 1], tokspans[run[-1] - 1]
                if sa and sb:
                    ws, we = sa[0], sb[1]
                    while ws > 0 and root[ws - 1] not in BOUND: ws -= 1
                    while we < len(root) and root[we] not in BOUND: we += 1
                    if " " not in root[ws:we]: word = (ws, we)
            for sub in sub_runs(run, toks):
                t = trim_shad(tgt, off[sub[0]][1], off[sub[-1]][2])
                if not t: continue
                s, e = t; bo = tgt[s:e]
                comp = len(sub) > 1 and all(toks[n - 1][0].rstrip().endswith("-") for n in sub[:-1])
                if comp:   # real compound -> hyphen-joined, no spaces (ātma-dharma-upacāraḥ)
                    concat = "".join(toks[n - 1][0].strip() for n in sub).rstrip("-")
                    lemma = "-".join(toks[n - 1][1] for n in sub)
                else:      # non-compound collocation -> keep the real multi-word form with spaces
                    concat = " ".join(surf(n, toks) for n in sub)
                    lemma = " ".join(toks[n - 1][1] for n in sub)
                o.write(json.dumps({"type": "phrase", "head": concat, "lemma": lemma, "sanskrit": concat,
                                    "zh": bo, "indices": list(sub), "start": s, "end": e, **loc},
                                   ensure_ascii=False) + "\n"); n_phr += 1
                # ALSO index this sub-compound under its ACTUAL sandhi'd surface substring
                if word and tokspans[sub[0] - 1] and tokspans[sub[-1] - 1]:
                    left = word[0] if sub[0] == run[0] else tokspans[sub[0] - 1][0]
                    right = word[1] if sub[-1] == run[-1] else (tokspans[sub[-1]][0] if tokspans[sub[-1]] else tokspans[sub[-1] - 1][1])
                    actual = root[left:right].strip()
                    exp = sum(len(surf(n, toks)) for n in sub)
                    if actual and " " not in actual and 0.6 <= len(actual) / max(exp, 1) <= 1.8 and actual != concat:
                        o.write(json.dumps({"type": "phrase", "head": actual, "head_kind": "surface",
                                            "lemma": lemma, "sanskrit": actual, "zh": bo,
                                            "indices": list(sub), "start": s, "end": e, **loc},
                                           ensure_ascii=False) + "\n"); n_phr += 1
        if RELAX and off:
            done = {tuple(sub) for run in mruns for sub in sub_runs(run, toks)}
            for win, s, e, rule in relaxed_windows(off, tgt, toks, done):
                concat = "".join(toks[n - 1][0].strip() for n in win).rstrip("-")
                lemma = "-".join(toks[n - 1][1] for n in win)
                o.write(json.dumps({"type": "phrase", "head": concat, "lemma": lemma, "sanskrit": concat,
                                    "zh": tgt[s:e], "indices": list(win), "start": s, "end": e, "rule": rule, **loc},
                                   ensure_ascii=False) + "\n"); n_phr += 1; n_relax[rule] += 1

print(f"token spans: {n_tok}   phrase spans: {n_phr}   of which relaxed (M): {dict(n_relax)}")
print(f"cursor-match failures: {n_fail}/{n_span} ({100*n_fail/max(n_span,1):.3f}%)")
print(f"-> {SPANS}")
