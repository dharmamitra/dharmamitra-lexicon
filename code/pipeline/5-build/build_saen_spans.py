#!/usr/bin/env python3
"""sa-en span builder with compound stitching (English-target variant of
build_sabo_spans.py / build_sazh_spans.py).

Input : sa-en/mined.ndjson (qwen span-projection output on Mitrasamgraha).
Gates : verbatim copy, numbering ok, conf >= -0.45, ung_frac <= 0.25. The
        coverage<30% flag is NOT a gate here (literary English translations
        legitimately leave many Sanskrit words untagged); it is recorded per span
        as low_coverage so the bidict/webapp can weight or filter.
Offsets: taken from the model's own tag positions in `output` (English repeats
        words like "form" often, so first-occurrence search is unsafe); cursor
        search is the fallback.
Emits (sa-en/hybrid/sa-en.spans.ndjson):
  - token spans  (head = LEMMA; the single-word dictionary)
  - phrase spans (head = DESANDHIED SURFACE compound, hyphen-joined) when
    consecutive compound members (trailing '-' on the unsandhied surface) all
    align and their English spans form one unit: the covering English substring
    may contain ONLY glue words (of, the, and, ...) / light punctuation between the
    member spans, in ANY order (English inverts compounds: jagat-nivāsa ->
    "abode of the universe"), and no other aligned span may fall inside it.
    All contiguous sub-compounds are emitted so sub-terms are searchable.
    Non-compound collocations merge only when literally adjacent + in order.
  - head-only phrases (head_kind="head_only"): compound whose members are all
    untagged and only the head member carries a MULTI-WORD English span (the
    seed policy: a compound rendered as one English unit is tagged once under the
    head) -> the whole compound is indexed under that span.
  - surface variants (head_kind="surface"): the actual sandhi'd compound
    substring of the source sentence, as in the sa-bo/sa-zh builders.
"""
import json, os, re, sys, difflib
from multiprocessing import Pool

BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
MINED = os.environ.get("SPANS_IN") or f"{BASE}/sa-en/mined.ndjson"
SPANS = os.environ.get("SPANS_OUT") or f"{BASE}/sa-en/hybrid/sa-en.spans.ndjson"
WORKERS = int(os.environ.get("WORKERS", "64"))
MAXSUB = 8
MAXGLUE = 4
GLUE = {"of", "the", "a", "an", "and", "or", "in", "for", "to", "with", "by", "on", "at",
        "from", "'s", "s", "who", "which", "that", "is", "are", "was", "were", "be", "as",
        "one", "his", "her", "its", "their", "own", "very"}
TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)
WORD = re.compile(r"[A-Za-zĀ-ſḀ-ỿ']+")
EDGE = " \t,.;:!?'\"()[]“”‘’—–-"
CLAUSE = set(".;:!?")
BOUND = set(" \t/|0123456789()[]#.,")


# ---------- Sanskrit surface alignment (verbatim from the sa-bo builder) ----------
def align_tokens(toks, root):
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


def is_member(n, toks):
    return toks[n - 1][0].rstrip().endswith("-")


def surf(n, toks):
    s = toks[n - 1][0].strip()
    return s[:-1].strip() if s.endswith("-") else s


# ---------- English offsets ----------
def offsets_from_output(output, target, aligns, ntok):
    """Positions from the tagged output when its detagged text equals the target."""
    s = re.sub(r"\s+", " ", output).strip()
    s = re.sub(r"<(\d+)>\s+", r"<\1>", s); s = re.sub(r"\s+</(\d+)>", r"</\1>", s)
    if re.sub(r"</?\d+>", "", s) != target:
        return None
    out, pos_out, pos_tgt, seen = {}, 0, 0, set()
    for m in TAG.finditer(s):
        pos_tgt += len(re.sub(r"</?\d+>", "", s[pos_out:m.start()]))
        n, span = int(m.group(1)), m.group(2)
        if 1 <= n <= ntok and n not in seen and span and aligns.get(str(n)) == span \
                and target[pos_tgt:pos_tgt + len(span)] == span:
            out[n] = (span, pos_tgt, pos_tgt + len(span)); seen.add(n)
        pos_tgt += len(span); pos_out = m.end()
    return out


def offsets_cursor(target, aligns, ntok):
    cur, out, fail = 0, {}, 0
    for k, span in aligns.items():
        n = int(k)
        if not (1 <= n <= ntok):
            continue
        p = target.find(span, cur)
        if p < 0:
            p = target.find(span)
            if p < 0:
                fail += 1; continue
        out[n] = (span, p, p + len(span)); cur = p + len(span)
    return out, fail


def trim(target, s, e):
    while s < e and target[s] in EDGE: s += 1
    while e > s and target[e - 1] in EDGE: e -= 1
    if e <= s or any(c in CLAUSE for c in target[s:e]):
        return None
    return s, e


# ---------- merging ----------
def cover_ok(run, off, target, compound):
    """Covering English substring of the run's spans; ok iff leftover text is only
    glue/punct (compound) or empty (collocation, in order), and no foreign span leaks."""
    starts = [off[n][1] for n in run]; ends = [off[n][2] for n in run]
    lo, hi = min(starts), max(ends)
    for m, (_, s, e) in off.items():
        if m not in run and s < hi and e > lo:
            return None                                   # leak of another aligned word
    if not compound:
        if any(off[run[i]][2] > off[run[i + 1]][1] for i in range(len(run) - 1)):
            return None                                   # must be in order
    mask = bytearray(b" " * (hi - lo))
    for n in run:
        _, s, e = off[n]
        mask[s - lo:e - lo] = b"x" * (e - s)
    leftover = "".join(ch for ch, m in zip(target[lo:hi], mask) if m == 32)
    words = WORD.findall(leftover)
    if not compound:
        return (lo, hi) if not words and not leftover.strip() else None
    if len(words) > MAXGLUE or any(w.lower() not in GLUE for w in words):
        return None
    if any(c in CLAUSE for c in leftover):
        return None
    return lo, hi


def maximal_runs(off, target, toks):
    idx = sorted(off); res = []; i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            cand = idx[i:j + 2]
            comp = all(is_member(n, toks) for n in cand[:-1])
            if cover_ok(cand, off, target, comp):
                j += 1
            else:
                break
        if j > i:
            res.append(idx[i:j + 1])
        i = j + 1
    return res


def sub_runs(run, toks):
    out = set(); L = len(run)
    for a in range(L):
        for b in range(a + 1, L):
            if b - a + 1 > MAXSUB:
                break
            if not is_member(run[b - 1], toks):
                break
            out.add(tuple(run[a:b + 1]))
    out.add(tuple(run))
    return out


def head_only_compounds(off, toks):
    """Compounds where no member is aligned and the head carries a multi-word span."""
    res = []; i = 1; N = len(toks)
    while i <= N:
        if is_member(i, toks):
            j = i
            while j <= N and is_member(j, toks): j += 1
            if j <= N and j in off and not any(n in off for n in range(i, j)) \
                    and len(off[j][0].split()) >= 2:
                res.append(list(range(i, j + 1)))
            i = j + 1
        else:
            i += 1
    return res


def compound_forms(sub, toks):
    comp = len(sub) > 1 and all(is_member(n, toks) for n in sub[:-1])
    if comp:
        concat = "".join(toks[n - 1][0].strip() for n in sub).rstrip("-")
        lemma = "-".join(toks[n - 1][1] for n in sub)
    else:
        concat = " ".join(surf(n, toks) for n in sub)
        lemma = " ".join(toks[n - 1][1] for n in sub)
    return comp, concat, lemma


def surface_variant(root, tokspans, sub, run, word, concat, toks):
    if not (word and tokspans[sub[0] - 1] and tokspans[sub[-1] - 1]):
        return None
    left = word[0] if sub[0] == run[0] else tokspans[sub[0] - 1][0]
    right = word[1] if sub[-1] == run[-1] else (tokspans[sub[-1]][0] if sub[-1] < len(tokspans) and tokspans[sub[-1]] else tokspans[sub[-1] - 1][1])
    actual = root[left:right].strip()
    exp = sum(len(surf(n, toks)) for n in sub)
    if actual and " " not in actual and 0.6 <= len(actual) / max(exp, 1) <= 1.8 and actual != concat:
        return actual
    return None


def word_bounds(root, tokspans, run):
    sa, sb = tokspans[run[0] - 1], tokspans[run[-1] - 1]
    if not (sa and sb):
        return None
    ws, we = sa[0], sb[1]
    while ws > 0 and root[ws - 1] not in BOUND: ws -= 1
    while we < len(root) and root[we] not in BOUND: we += 1
    return (ws, we) if " " not in root[ws:we] else None


# ---------- per row ----------
def process(line):
    r = json.loads(line)
    st = {"rows": 0, "gated": 0, "tok": 0, "phr": 0, "headonly": 0, "surface": 0, "fail": 0, "spans": 0}
    if not r["verbatim"] or r["numbering_bad"] or r["conf"] < -0.45 or r["ung_frac"] > 0.25:
        st["gated"] = 1
        return [], st
    st["rows"] = 1
    toks, tgt, root = r["toks"], r["target"], r["src_original"]
    off = offsets_from_output(r["output"], tgt, r["aligns"], len(toks))
    if off is None or len(off) < len(r["aligns"]):
        off, st["fail"] = offsets_cursor(tgt, r["aligns"], len(toks))
    st["spans"] = len(r["aligns"])
    parts = r["id"].split("/")
    loc = {"sentence_id": r["id"], "split": parts[0], "license": parts[1] if len(parts) == 3 else "",
           "source": "qwen-v2-saen", "low_coverage": bool(r["coverage"] < 0.30),
           "conf": r["conf"], "target": tgt, "src_sentence": root}
    out = []
    for n in sorted(off):
        _, s, e = off[n]; sf, lem = toks[n - 1]
        t = trim(tgt, s, e)
        if not t: continue
        s, e = t
        out.append({"type": "token", "head": lem, "lemma": lem, "sanskrit": sf, "en": tgt[s:e],
                    "indices": [n], "start": s, "end": e, **loc}); st["tok"] += 1

    mruns = maximal_runs(off, tgt, toks)
    honly = head_only_compounds(off, toks)
    need_surface = any(is_member(run[0], toks) for run in mruns) or honly
    tokspans = align_tokens(toks, root) if need_surface else None

    def emit_phrase(sub, run, s, e, kind=None):
        t = trim(tgt, s, e)
        if not t: return
        s, e = t
        comp, concat, lemma = compound_forms(sub, toks)
        row = {"type": "phrase", "head": concat, "lemma": lemma, "sanskrit": concat, "en": tgt[s:e],
               "indices": list(sub), "start": s, "end": e, **loc}
        if kind: row["head_kind"] = kind
        out.append(row); st["phr"] += 1
        if comp and tokspans is not None:
            word = word_bounds(root, tokspans, run)
            actual = surface_variant(root, tokspans, sub, run, word, concat, toks)
            if actual:
                out.append({**row, "head": actual, "head_kind": "surface", "sanskrit": actual})
                st["surface"] += 1

    for run in mruns:
        for sub in sub_runs(run, toks):
            comp = all(is_member(n, toks) for n in sub[:-1])
            c = cover_ok(list(sub), off, tgt, comp)
            if not c: continue
            emit_phrase(sub, run, c[0], c[1])
    for run in honly:
        _, s, e = off[run[-1]]
        emit_phrase(tuple(run), run, s, e, kind="head_only"); st["headonly"] += 1
    return out, st


def main():
    os.makedirs(os.path.dirname(SPANS), exist_ok=True)
    tot = {}
    seen = set(); dup = 0
    with open(SPANS, "w", encoding="utf-8") as o, Pool(WORKERS) as pool:
        for rows, st in pool.imap(process, open(MINED, encoding="utf-8"), chunksize=200):
            for k, v in st.items(): tot[k] = tot.get(k, 0) + v
            if rows:
                key = (rows[0]["target"], rows[0]["src_sentence"])   # exact duplicate pairs
                if key in seen:
                    dup += 1; continue
                seen.add(key)
                for row in rows:
                    o.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows used: {tot['rows']}  gated out: {tot['gated']}  duplicate pairs skipped: {dup}")
    print(f"token spans: {tot['tok']}  phrase spans: {tot['phr']} (head-only compounds: {tot['headonly']}, "
          f"surface variants: {tot['surface']})")
    print(f"cursor-match failures: {tot['fail']}/{tot['spans']}")
    print("->", SPANS)


if __name__ == "__main__":
    main()
