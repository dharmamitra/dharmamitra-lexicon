#!/usr/bin/env python3
"""Statistical word-alignment baselines (eflomal, fast_align) for the MITRA-dict paper.

Two questions, per direction (sa-bo, sa-zh, sa-en):
  1. agreement with the held-out teacher labels (~/data/qwen-dictionaries-heldout500/test.jsonl),
     link-level precision / recall / F1, for the student checkpoints and the baselines alike;
  2. a dictionary built from the baseline links with the same lemma -> sense counting as
     MITRA-dict, evaluated against the reference lexica with build/eval/eval_sa_*.py.

Tokenization (identical for every system, so links are comparable):
  source  = Sanskrit lemma per analyzer token (the dictionary's headword unit)
  target  = Tibetan: Wylie syllables (tsheg / space / shad split)
            Chinese: characters (punctuation kept as tokens but never scored / never a sense)
            English: \\w+ words and punctuation marks, lower-cased
The baselines are trained on the full corpus that the student aligned (sa-bo 905k, sa-zh 178k,
sa-en 400k pairs) with the held-out pairs appended, and read their links off those lines.

  python stat_baselines.py prep  --direction sa-bo
  python stat_baselines.py align --direction sa-bo --tool eflomal|fast_align
  python stat_baselines.py score --direction sa-bo            # needs student preds files
  python stat_baselines.py dict  --direction sa-bo --tool eflomal|fast_align|llm
"""
import argparse, collections, json, os, random, re, subprocess, sys, time
from pathlib import Path

Q = Path("/mnt2/code/mitra-dictionaries/qwen-dictionaries")
WORK = Path(os.environ.get("BASELINE_WORK", Q / "baselines" / "work"))   # work-v2: sa-bo/sa-zh on MITRA-parallel v2 (2026-10-13)
HELD = Path(os.path.expanduser("~/data/qwen-dictionaries-heldout500"))
TOOLS = Path(os.path.expanduser("~/code/statistical-alignment/tools"))
FAST_ALIGN, ATOOLS = TOOLS / "fast_align/build/fast_align", TOOLS / "fast_align/build/atools"
CORPUS = {"sa-bo": Q / "sa-bo/hybrid/final.ndjson", "sa-zh": Q / "sa-zh/hybrid/final.ndjson",
          "sa-en": Q / "sa-en/mined.ndjson"}
for _d, _v in (("sa-bo", "CORPUS_SA_BO"), ("sa-zh", "CORPUS_SA_ZH")):   # v2 corpora (qwen-dictionaries-v3) via env
    if os.environ.get(_v):
        CORPUS[_d] = Path(os.environ[_v])
SENSE_FIELD = {"sa-bo": "bo", "sa-zh": "zh", "sa-en": "en"}
JOIN = {"sa-bo": " ", "sa-zh": "", "sa-en": " "}
TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)
PUNCT_ZH = set("，。、；：？！「」『』（）《》〈〉【】…—～·　 ,.;:?!()[]\"'")


# ------------------------------------------------------------------ tokenizers
def tok_bo_wylie(s):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^\s/|_]+", s)]


def tok_bo_uni(s):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^\s་།༎༏༐༑༔]+", s)]


def tok_zh(s):
    return [(c, i, i + 1) for i, c in enumerate(s) if not c.isspace()]


def tok_en(s):
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", s)]


TOK = {"sa-bo": tok_bo_wylie, "sa-zh": tok_zh, "sa-en": tok_en}


def is_punct(direction, tok):
    if direction == "sa-zh":
        return tok in PUNCT_ZH or not any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in tok)
    if direction == "sa-en":
        return not re.match(r"\w", tok)
    return False


def spans_from_tagged(tagged):
    """{n: (start, end)} in detagged coordinates, using the tag positions."""
    out, pos, plain = {}, 0, []
    i = 0
    for m in re.finditer(r"<(/?)(\d+)>", tagged):
        plain.append(tagged[i:m.start()]); pos += m.start() - i; i = m.end()
        n = int(m.group(2))
        if not m.group(1):
            out.setdefault(n, [pos, None])
        elif n in out and out[n][1] is None:
            out[n][1] = pos
    plain.append(tagged[i:])
    return {n: (s, e) for n, (s, e) in out.items() if e is not None and e > s}, "".join(plain)


def links_from_tagged(tagged, target_tokens):
    """teacher/student tagged line -> {(i, j)} over target token indices overlapping each span."""
    spans, plain = spans_from_tagged(tagged)
    links = set()
    for n, (a, b) in spans.items():
        for j, (_, s, e) in enumerate(target_tokens):
            if s < b and e > a:
                links.add((n - 1, j))
    return links, plain


# ------------------------------------------------------------------------ prep
def corpus_rows(direction):
    for line in open(CORPUS[direction], encoding="utf-8"):
        d = json.loads(line)
        toks = d.get("toks") or []
        src = [(t[1] or t[0] or "").strip() or "<unk>" for t in toks]
        tgt = d.get("target") or ""
        if not src or not tgt.strip():
            continue
        yield d, src, tgt


def heldout_rows(direction):
    rows = [json.loads(l) for l in open(HELD / "test.jsonl", encoding="utf-8")]
    return [(k, r) for k, r in enumerate(rows) if r["lang"] == direction]


def parse_block(input_text):
    lem = re.search(r"^LEMMAS: (.*)$", input_text, re.M).group(1)
    return [m.group(2) for m in TAG.finditer(lem)]


def prep(direction):
    WORK.mkdir(parents=True, exist_ok=True)
    src_f = open(WORK / f"{direction}.src", "w", encoding="utf-8")
    trg_f = open(WORK / f"{direction}.trg", "w", encoding="utf-8")
    meta_f = open(WORK / f"{direction}.corpus_meta.jsonl", "w", encoding="utf-8")   # target + llm links per line
    tok = TOK[direction]
    n = 0
    for d, src, tgt in corpus_rows(direction):
        tt = tok(tgt)
        if not tt:
            continue
        src_f.write(" ".join(w.replace(" ", "_") for w in src) + "\n")
        trg_f.write(" ".join(w for w, _, _ in tt) + "\n")
        # LLM links from the mined aligns (span text -> first occurrence, as in the span build)
        llm = []
        for k, span in (d.get("aligns") or {}).items():
            i = int(k) - 1
            p = tgt.find(span) if span else -1
            if p < 0 or not (0 <= i < len(src)):
                continue
            llm += [(i, j) for j, (_, s, e) in enumerate(tt) if s < p + len(span) and e > p]
        meta_f.write(json.dumps({"target": tgt, "llm": llm, "ok": bool(d.get("verbatim")) and not d.get("numbering_bad")},
                                ensure_ascii=False) + "\n")
        n += 1
    n_corpus = n
    # held-out rows appended; gold + target tokens recorded for scoring
    held = []
    raw_bo = None
    if direction == "sa-bo":
        raw_bo = {(str(d["src_segnr"]), str(d["tgt_segnr"])): d["target"]
                  for d in map(json.loads, open(Q / "heldout/sa-bo/gemini_raw.ndjson", encoding="utf-8"))}
    skipped = 0
    for k, r in heldout_rows(direction):
        src = parse_block(r["input_text"])
        _, plain = spans_from_tagged(r["target_text"])
        if direction == "sa-bo":
            wy = raw_bo[(str(r["src_segnr"]), str(r["tgt_segnr"]))]
            tt_al = tok_bo_wylie(wy)              # aligner side (Wylie)
            tt = tok_bo_uni(plain)                # scoring side (Unicode)
            if len(tt_al) != len(tt):
                skipped += 1
                continue
        else:
            tt = tok(plain); tt_al = tt
        if not tt:
            skipped += 1; continue
        gold, _ = links_from_tagged(r["target_text"], tt)
        src_f.write(" ".join(w.replace(" ", "_") for w in src) + "\n")
        trg_f.write(" ".join(w for w, _, _ in tt_al) + "\n")
        held.append({"row": k, "line": n, "n_src": len(src), "tokens": [w for w, _, _ in tt],
                     "punct": [is_punct(direction, w) for w, _, _ in tt], "gold": sorted(gold),
                     "gold_tagged": r["target_text"]})
        n += 1
    src_f.close(); trg_f.close(); meta_f.close()
    json.dump({"n_corpus": n_corpus, "n_total": n, "held": held}, open(WORK / f"{direction}.held.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"[{direction}] corpus {n_corpus} lines + {len(held)} held-out ({skipped} skipped) -> {WORK}", file=sys.stderr)


# ----------------------------------------------------------------------- align
def parse_links_file(p):
    out = []
    for ln in open(p, encoding="utf-8"):
        s = set()
        for tk in ln.split():
            a, b = tk.split("-"); s.add((int(a), int(b)))
        out.append(s)
    return out


def gdfa(fwd, rev, ls, lt):
    inter, union = fwd & rev, fwd | rev
    al = set(inter)
    sa = {i for i, _ in al}; ta = {j for _, j in al}
    nbr = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    added = True
    while added:
        added = False
        for (i, j) in list(al):
            for di, dj in nbr:
                ni, nj = i + di, j + dj
                if 0 <= ni < ls and 0 <= nj < lt and (ni, nj) not in al and (ni, nj) in union:
                    if ni not in sa or nj not in ta:
                        al.add((ni, nj)); sa.add(ni); ta.add(nj); added = True
    for (i, j) in union:
        if i not in sa and j not in ta:
            al.add((i, j)); sa.add(i); ta.add(j)
    return al


MGIZA = TOOLS / "mgiza/mgizapp/bin"
A3 = re.compile(r"(\S+) \(\{([\d ]*)\}\)")


def a3_to_links(parts, out, swap):
    """merge mgiza's A3.final.part* by sentence number and write one 'i-j ...' line per sentence."""
    sent = {}
    for p in parts:
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        for k in range(0, len(lines) - 2, 3):
            m = re.match(r"# Sentence pair \((\d+)\)", lines[k])
            if not m:
                continue
            links = set()
            for i, mm in enumerate(A3.finditer(lines[k + 2])):
                if i == 0:                      # NULL
                    continue
                for j in mm.group(2).split():
                    a, b = i - 1, int(j) - 1
                    links.add((b, a) if swap else (a, b))
            sent[int(m.group(1))] = links
    n = max(sent) if sent else 0
    with open(out, "w", encoding="utf-8") as f:
        for k in range(1, n + 1):
            f.write(" ".join(f"{i}-{j}" for i, j in sorted(sent.get(k, ()))) + "\n")
    return n


def run_mgiza(direction, ncpus=20):
    import shutil, glob as _glob
    d = WORK / f"{direction}.mgiza"; d.mkdir(exist_ok=True)
    S, T = d / "src.txt", d / "trg.txt"
    shutil.copy(WORK / f"{direction}.src", S); shutil.copy(WORK / f"{direction}.trg", T)
    run = lambda cmd, **kw: subprocess.run([str(c) for c in cmd], check=True, cwd=d, **kw)
    run([MGIZA / "plain2snt", S, T])          # writes src.vcb, trg.vcb, src_trg.snt, trg_src.snt (extension stripped)
    for X in (S, T):
        run([MGIZA / "mkcls", "-c50", "-n2", f"-p{X}", f"-V{d / X.stem}.vcb.classes"])
    outs = {}
    for name, (a, b) in {"fwd": (S, T), "rev": (T, S)}.items():
        snt = d / f"{a.stem}_{b.stem}.snt"
        cooc = d / f"{name}.cooc"
        va, vb = d / f"{a.stem}.vcb", d / f"{b.stem}.vcb"
        run([MGIZA / "snt2cooc", cooc, va, vb, snt])
        with open(d / f"{name}.log", "w") as log:
            run([MGIZA / "mgiza", "-ncpus", ncpus, "-c", snt, "-s", va, "-t", vb,
                 "-CoocurrenceFile", cooc, "-o", d / name, "-m1", "5", "-m2", "0", "-mh", "5", "-m3", "3", "-m4", "3",
                 "-onlyaldumps", "1", "-nodumps", "1"], stdout=log, stderr=subprocess.STDOUT)
        parts = sorted(_glob.glob(str(d / f"{name}.A3.final.part*")))
        outs[name] = WORK / f"{direction}.mgiza.{name}"
        n = a3_to_links(parts, outs[name], swap=(name == "rev"))
        print(f"[{direction}/mgiza] {name}: {n} sentences from {len(parts)} parts", file=sys.stderr)
    return outs["fwd"], outs["rev"]


def write_links(path, links):
    with open(path, "w", encoding="utf-8") as f:
        for s in links:
            f.write(" ".join(f"{i}-{j}" for i, j in sorted(s)) + "\n")


def align(direction, tool):
    src, trg = WORK / f"{direction}.src", WORK / f"{direction}.trg"
    out = WORK / f"{direction}.{tool}.links"
    t0 = time.time()
    if tool == "eflomal":
        import eflomal
        fwd, rev = WORK / f"{direction}.eflomal.fwd", WORK / f"{direction}.eflomal.rev"
        with open(src, encoding="utf-8") as sf, open(trg, encoding="utf-8") as tf:
            eflomal.Aligner(model=3).align(sf, tf, links_filename_fwd=str(fwd), links_filename_rev=str(rev), quiet=True)
    elif tool == "fast_align":
        bitext = WORK / f"{direction}.bitext"
        with open(bitext, "w", encoding="utf-8") as bf:
            for a, b in zip(open(src, encoding="utf-8"), open(trg, encoding="utf-8")):
                bf.write(f"{a.rstrip()} ||| {b.rstrip()}\n")
        fwd, rev = WORK / f"{direction}.fast_align.fwd", WORK / f"{direction}.fast_align.rev"
        with open(fwd, "w") as f:
            subprocess.run([str(FAST_ALIGN), "-i", str(bitext), "-d", "-o", "-v"], stdout=f, stderr=subprocess.DEVNULL, check=True)
        with open(rev, "w") as f:
            subprocess.run([str(FAST_ALIGN), "-i", str(bitext), "-d", "-o", "-v", "-r"], stdout=f, stderr=subprocess.DEVNULL, check=True)
    elif tool == "mgiza":
        fwd, rev = run_mgiza(direction)
    else:
        raise SystemExit(f"unknown tool {tool}")
    print(f"[{direction}/{tool}] aligned in {time.time()-t0:.0f}s; symmetrizing", file=sys.stderr)
    ls = [len(l.split()) for l in open(src, encoding="utf-8")]
    lt = [len(l.split()) for l in open(trg, encoding="utf-8")]
    F, R = parse_links_file(fwd), parse_links_file(rev)
    sym = [gdfa(f, r, a, b) for f, r, a, b in zip(F, R, ls, lt)]
    write_links(out, sym)
    print(f"[{direction}/{tool}] {len(sym)} lines -> {out} ({time.time()-t0:.0f}s)", file=sys.stderr)


# ----------------------------------------------------------------------- score
def prf(c, g, p):
    r = c / g if g else 0.0; pr = c / p if p else 0.0
    return 100 * pr, 100 * r, (200 * r * pr / (r + pr) if r + pr else 0.0)


def score_system(held, sys_links, boot=1000):
    """sys_links: list aligned with held (set of (i,j)). Returns dict of metrics with bootstrap CI on F1."""
    per = []
    for h, L in zip(held, sys_links):
        punct = h["punct"]
        gold = {(i, j) for i, j in map(tuple, h["gold"]) if not punct[j]}
        pred = {(i, j) for i, j in L if 0 <= j < len(punct) and not punct[j] and 0 <= i < h["n_src"]}
        tagged_src = {i for i, _ in gold}
        pred_on_tagged = {(i, j) for i, j in pred if i in tagged_src}
        per.append((len(gold & pred), len(gold), len(pred), len(gold & pred_on_tagged), len(pred_on_tagged)))
    tot = [sum(x[k] for x in per) for k in range(5)]
    p, r, f = prf(tot[0], tot[1], tot[2])
    p_t, _, f_t = prf(tot[3], tot[1], tot[4])
    rng = random.Random(0); fs = []
    for _ in range(boot):
        s = [per[rng.randrange(len(per))] for _ in per]
        fs.append(prf(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s))[2])
    fs.sort()
    return dict(n=len(per), gold=tot[1], pred=tot[2], correct=tot[0], precision=p, recall=r, f1=f,
                f1_ci=(fs[int(0.025 * boot)], fs[int(0.975 * boot) - 1]),
                precision_on_tagged_source=p_t, f1_on_tagged_source=f_t)


def score(direction):
    H = json.load(open(WORK / f"{direction}.held.json", encoding="utf-8"))
    held = H["held"]
    systems = {}
    # student checkpoints: predictions on the held-out rows
    # extra student checkpoints (e.g. the base-model ablation) via STUDENT_TAGS="tag1,tag2"
    for tag in ("four-dir", "five-dir") + tuple(t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t):
        p = HELD / f"test.preds-{tag}.jsonl"
        if not p.exists():
            continue
        preds = [json.loads(l) for l in open(p, encoding="utf-8")]
        L = []
        for h in held:
            r = preds[h["row"]]
            tt = (tok_bo_uni if direction == "sa-bo" else TOK[direction])(spans_from_tagged(h["gold_tagged"])[1])
            links, plain = links_from_tagged(r["pred"], tt)
            # if the student did not copy verbatim, positions are unreliable: fall back to text search
            if re.sub(r"\s+", " ", plain).strip() != re.sub(r"\s+", " ", spans_from_tagged(h["gold_tagged"])[1]).strip():
                target = spans_from_tagged(h["gold_tagged"])[1]
                links = set()
                for m in TAG.finditer(r["pred"]):
                    n, span = int(m.group(1)), m.group(2).strip()
                    q = target.find(span) if span else -1
                    if q >= 0:
                        links |= {(n - 1, j) for j, (_, s, e) in enumerate(tt) if s < q + len(span) and e > q}
            L.append(links)
        systems[f"student ({tag})"] = L
    # teacher self-agreement is 100 by construction; baselines:
    for tool in ("fast_align", "mgiza", "eflomal"):
        p = WORK / f"{direction}.{tool}.links"
        if not p.exists():
            continue
        allL = parse_links_file(p)
        systems[tool] = [allL[h["line"]] for h in held]
    res = {name: score_system(held, L) for name, L in systems.items()}
    print(f"\n=== {direction}: link-level agreement with held-out teacher labels (n={len(held)} sentences, "
          f"{sum(len(h['gold']) for h in held)} gold links) ===")
    print(f"{'system':22} {'P':>6} {'R':>6} {'F1':>6}   95% CI F1       | P / F1 on teacher-tagged source words")
    for name, m in res.items():
        print(f"{name:22} {m['precision']:6.1f} {m['recall']:6.1f} {m['f1']:6.1f}   [{m['f1_ci'][0]:5.1f}, {m['f1_ci'][1]:5.1f}]   | "
              f"{m['precision_on_tagged_source']:5.1f} / {m['f1_on_tagged_source']:5.1f}")
    json.dump(res, open(WORK / f"{direction}.scores.json", "w"), indent=1)
    return res


# ------------------------------------------------------------------------ dict
def build_dict(direction, tool):
    """lemma -> sense counts from the corpus links (held-out lines excluded), MITRA-dict bidict shape."""
    H = json.load(open(WORK / f"{direction}.held.json", encoding="utf-8"))
    n_corpus = H["n_corpus"]
    field, join = SENSE_FIELD[direction], JOIN[direction]
    tok = TOK[direction]
    meta = open(WORK / f"{direction}.corpus_meta.jsonl", encoding="utf-8")
    srcs = open(WORK / f"{direction}.src", encoding="utf-8")
    links_it = (parse_links_file(WORK / f"{direction}.{tool}.links") if tool != "llm" else None)
    counts = collections.defaultdict(collections.Counter)
    examples = {}
    t0 = time.time()
    for ln in range(n_corpus):
        m = json.loads(meta.readline())
        src = srcs.readline().split()
        if tool == "llm":
            if not m["ok"]:
                continue
            L = {tuple(x) for x in m["llm"]}
        else:
            L = links_it[ln]
        tt = tok(m["target"])
        by_i = collections.defaultdict(list)
        for i, j in L:
            if 0 <= i < len(src) and 0 <= j < len(tt) and not is_punct(direction, tt[j][0]):
                by_i[i].append(j)
        for i, js in by_i.items():
            js.sort()
            # longest contiguous run of linked target tokens = the sense string
            best, cur = [js[0]], [js[0]]
            for j in js[1:]:
                cur = cur + [j] if j == cur[-1] + 1 else [j]
                if len(cur) > len(best):
                    best = cur
            if len(best) > 8:
                continue
            a, b = tt[best[0]][1], tt[best[-1]][2]
            sense = m["target"][a:b] if direction != "sa-en" else m["target"][a:b].lower()
            sense = " ".join(sense.split())
            if not sense:
                continue
            lemma = src[i].replace("_", " ")
            counts[lemma][sense] += 1
            examples.setdefault((lemma, sense), {"target": m["target"], "start": a, "end": b})
    out = WORK / f"{direction}.{tool}.bidict.ndjson"
    with open(out, "w", encoding="utf-8") as f:
        for lemma in sorted(counts, key=lambda k: -sum(counts[k].values())):
            senses = [{field: s, "count": c, "ex": [examples[(lemma, s)]]} for s, c in counts[lemma].most_common()]
            f.write(json.dumps({"sa": lemma, "n": sum(counts[lemma].values()), "senses": senses}, ensure_ascii=False) + "\n")
    print(f"[{direction}/{tool}] {len(counts)} headwords -> {out} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "align", "score", "dict"])
    ap.add_argument("--direction", required=True, choices=list(CORPUS))
    ap.add_argument("--tool", default="eflomal")
    a = ap.parse_args()
    {"prep": lambda: prep(a.direction), "align": lambda: align(a.direction, a.tool),
     "score": lambda: score(a.direction), "dict": lambda: build_dict(a.direction, a.tool)}[a.cmd]()


if __name__ == "__main__":
    main()
