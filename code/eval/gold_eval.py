"""Sanskrit-English expert gold alignments: inter-annotator agreement and scores of the lexicon's spans.

Inputs (data/gold/): sa-en_gold_alignments.jsonl (annotations of A, B, C), items.json (the 100 sentences with
English word tokens and the span `pred` from which the lexicon was built), baseline_links.jsonl (links of
fast_align, MGIZA++ and eflomal on the same sentences), genre.json.

Units are the annotator's units: every analysed Sanskrit word and every maximal compound stack.
A decision is either `none` or an English word range.
  agreement (per annotator pair, units both annotated):
      status agreement and Cohen's kappa on aligned/none, exact agreement on the decision,
      span F1 = 2 * #identical spans / (#spans A + #spans B), mean Jaccard when both aligned;
      Krippendorff's alpha (nominal) on the decision over all annotators.
  model (the span the lexicon was built from, `pred` of the item) against each annotator:
      exact-span precision / recall / F1, and soft F1 (Jaccard of the word ranges as partial credit);
      by unit kind and by genre; bootstrap CI over sentences; also against the consensus (all three agree)
      and the majority (two of three) of the annotators.
  links (word units only): links (Sanskrit token, English \\w+ token) from the gold span, against the
      student links (pred) and the statistical aligners' links.
  content_words: the same without closed-class words (FUNCTION).

Usage: python3 gold_eval.py [OUT.json]   (default gold-eval.json)
"""
import collections, csv, itertools, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GOLD_DIR", os.path.join(HERE, "..", "..", "data", "gold"))
GOLD = os.path.join(DATA, "sa-en_gold_alignments.jsonl")
ITEMS = os.path.join(DATA, "items.json")
BASELINES = os.path.join(DATA, "baseline_links.jsonl")
GENRE = os.path.join(DATA, "genre.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else "gold-eval.json"
FULL = ("A", "B", "C")                                   # the three annotators, each did all 100 sentences
# closed-class lemmas (pronouns, particles, conjunctions, the copula) that the guidelines leave unaligned
FUNCTION = set("tad idam etad yad adas kim mad tvad asmad yuṣmad bhavat ca na eva api iva vā hi tu iti tatra tadā tathā "
               "yathā tatas tasmāt atha yatra yadā kva kutra kadā sma ha vai u as".split())


def load():
    items = {it["id"]: it for it in json.load(open(ITEMS)) if it.get("pool") == "agreement"}
    ann = collections.defaultdict(dict)                       # item -> annotator -> uid -> decision
    for l in open(GOLD):
        r = json.loads(l)
        if r["pool"] != "agreement":
            continue
        d = {}
        for u in r["units"]:
            g = u["gold"]
            if g is None:
                continue
            d[u["uid"]] = None if g["status"] == "none" else (g["start_word"], g["end_word"])
        ann[r["id"]][r["annotator"]] = d
    return items, ann


def pred_of(u):
    return tuple(u["pred"]) if u.get("pred") else None


def jac(a, b):
    A, B = set(range(a[0], a[1] + 1)), set(range(b[0], b[1] + 1))
    return len(A & B) / len(A | B)


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def alpha_nominal(units):
    """units: list of lists of values (one per annotator who coded the unit)."""
    o = collections.Counter();
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[(vals[a], vals[b])] += 1 / (m - 1)
    n = sum(o.values())
    nc = collections.Counter()
    for (a, _), v in o.items():
        nc[a] += v
    do = sum(v for (a, b), v in o.items() if a != b) / n
    de = sum(nc[a] * nc[b] for a in nc for b in nc if a != b) / (n * (n - 1))
    return 1 - do / de


def prf(tp, npred, ngold):
    p = tp / npred if npred else 0; r = tp / ngold if ngold else 0
    return 100 * p, 100 * r, (200 * p * r / (p + r) if p + r else 0)


def model_counts(items, ann, who, kinds=("token", "stack"), only=None):
    """per sentence: [tp_exact, tp_soft, npred, ngold, units, unchanged]"""
    per = {}
    for iid, by in ann.items():
        if who not in by:
            continue
        if only and not only(iid):
            continue
        c = [0, 0.0, 0, 0, 0, 0]
        for u in items[iid]["units"]:
            if u["kind"] not in kinds or u["uid"] not in by[who]:
                continue
            g, p = by[who][u["uid"]], pred_of(u)
            c[4] += 1; c[5] += (g == p)
            c[2] += p is not None; c[3] += g is not None
            if g and p:
                c[0] += g == p; c[1] += jac(g, p)
        per[iid] = c
    return per


def summarize(per, boot=1000):
    t = [sum(c[k] for c in per.values()) for k in range(6)]
    P, R, F = prf(t[0], t[2], t[3]); sP, sR, sF = prf(t[1], t[2], t[3])
    rng = random.Random(0); keys = list(per); fs = []
    for _ in range(boot):
        s = [per[keys[rng.randrange(len(keys))]] for _ in keys]
        fs.append(prf(sum(c[0] for c in s), sum(c[2] for c in s), sum(c[3] for c in s))[2])
    fs.sort()
    return dict(sentences=len(per), units=t[4], pred_spans=t[2], gold_spans=t[3], P=P, R=R, F1=F,
                F1_ci=(fs[int(.025 * boot)], fs[int(.975 * boot) - 1]), soft_P=sP, soft_R=sR, soft_F1=sF,
                unchanged=100 * t[5] / t[4] if t[4] else None)


def agreement(items, ann):
    annotators = sorted({a for by in ann.values() for a in by})
    kinds = {iid: {u["uid"]: u["kind"] for u in it["units"]} for iid, it in items.items()}
    pairs = {}
    for a, b in itertools.combinations(annotators, 2):
        res = {}
        for kind in ("all", "token", "stack"):
            st, ex, tp, na, nb, js, n = [], 0, 0, 0, 0, [], 0
            for iid, by in ann.items():
                if a not in by or b not in by:
                    continue
                for uid, x in by[a].items():
                    if uid not in by[b] or (kind != "all" and kinds[iid].get(uid) != kind):
                        continue
                    y = by[b][uid]; n += 1
                    st.append((x is not None, y is not None)); ex += x == y
                    na += x is not None; nb += y is not None
                    if x and y:
                        tp += x == y; js.append(jac(x, y))
            if not n:
                continue
            res[kind] = dict(units=n, sentences=sum(1 for by in ann.values() if a in by and b in by),
                             status_agreement=100 * sum(p == q for p, q in st) / n, kappa=kappa(st),
                             exact_agreement=100 * ex / n, span_F1=200 * tp / (na + nb),
                             jaccard_both_aligned=sum(js) / len(js), aligned_a=100 * na / n, aligned_b=100 * nb / n)
        pairs[f"{a} x {b}"] = res
    units = []
    for iid, by in ann.items():
        for u in items[iid]["units"]:
            vals = [str(by[a][u["uid"]]) for a in by if u["uid"] in by[a]]
            units.append(vals)
    return dict(pairs=pairs, alpha_nominal_all=alpha_nominal(units),
                alpha_nominal_full=alpha_nominal([[str(ann[i][a][u["uid"]]) for a in FULL if u["uid"] in ann[i].get(a, {})]
                                                   for i in ann for u in items[i]["units"]]))


# ------------------------------------------------------------------ link level vs statistical aligners
def tok_en(s):                                   # = stat_baselines.tok_en
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", s)]


def links_eval(items, ann):
    base = {d["id"]: d for d in map(json.loads, open(BASELINES, encoding="utf-8"))}
    for iid in ann:
        assert base[iid]["english"] == items[iid]["english"], iid
    lines = {iid: (iid, base[iid]["english"]) for iid in ann}
    systems = {t: {iid: {tuple(x) for x in base[iid][t]} for iid in ann} for t in ("fast_align", "mgiza", "eflomal")}
    res = {}
    for who in sorted({a for by in ann.values() for a in by}):
        tot = collections.defaultdict(lambda: [0, 0, 0])
        for iid, by in ann.items():
            if who not in by or iid not in lines:
                continue
            it = items[iid]; n = lines[iid][0]
            et = tok_en(it["english"]); word = [bool(re.match(r"\w", w)) for w, _, _ in et]
            def to_links(i, rng):          # annotator word range -> baseline-token links
                s, e = it["en_tokens"][rng[0]]["s"], it["en_tokens"][rng[1]]["e"]
                return {(i, j) for j, (_, a, b) in enumerate(et) if a < e and b > s and word[j]}
            gold, stud = set(), set()
            for u in it["units"]:
                if u["kind"] != "token" or u["uid"] not in by[who]:
                    continue
                i = u["indices"][0] - 1
                if by[who][u["uid"]]:
                    gold |= to_links(i, by[who][u["uid"]])
                if pred_of(u):
                    stud |= to_links(i, pred_of(u))
            scored = {u["indices"][0] - 1 for u in it["units"] if u["kind"] == "token" and u["uid"] in by[who]}
            cand = {"student": stud, **{t: {(i, j) for i, j in L[n] if i in scored and j < len(word) and word[j]}
                                        for t, L in systems.items()}}
            for name, L in cand.items():
                tot[name][0] += len(L & gold); tot[name][1] += len(L); tot[name][2] += len(gold)
        res[who] = {name: dict(zip(("P", "R", "F1"), prf(*v))) for name, v in tot.items()}
    return res


def genre_of(items):
    g = json.load(open(GENRE, encoding="utf-8"))
    return {iid: (g[iid]["genre"], g[iid]["period"]) for iid in items}


def content_only(items, ann):
    lem = {iid: {u["uid"]: u["lemma"] for u in it["units"] if u["kind"] == "token"} for iid, it in items.items()}
    return {iid: {a: {uid: v for uid, v in d.items() if lem[iid].get(uid) not in FUNCTION} for a, d in by.items()}
            for iid, by in ann.items()}


if __name__ == "__main__":
    items, ann = load()
    res = {"gold": GOLD, "sentences_per_annotator": {a: sum(a in by for by in ann.values()) for a in sorted({a for by in ann.values() for a in by})},
           "units_by_kind": dict(collections.Counter(u["kind"] for iid, it in items.items() for u in it["units"]
                                                     if u["uid"] in ann[iid].get(FULL[0], {}))),   # sub-stacks are not units
           "agreement": agreement(items, ann)}
    model = {}
    for who in res["sentences_per_annotator"]:
        model[who] = {k: summarize(model_counts(items, ann, who, kinds)) for k, kinds in
                      (("all", ("token", "stack")), ("token", ("token",)), ("stack", ("stack",)))}
    # consensus of the full annotators: units on which all agree exactly, and majority decisions (>= 2 of 3)
    cons = {iid: {"C": {uid: v for uid, v in by[FULL[0]].items() if all(uid in by[a] and by[a][uid] == v for a in FULL[1:])}}
            for iid, by in ann.items() if all(a in by for a in FULL)}
    def majority(by):
        out = {}
        for uid in by[FULL[0]]:
            vals = [by[a][uid] for a in FULL if uid in by[a]]
            v, c = collections.Counter(vals).most_common(1)[0]
            if c * 2 > len(FULL):
                out[uid] = v
        return out
    maj = {iid: {"M": majority(by)} for iid, by in ann.items() if all(a in by for a in FULL)}
    model["consensus"] = {k: summarize(model_counts(items, cons, "C", kinds)) for k, kinds in
                          (("all", ("token", "stack")), ("token", ("token",)), ("stack", ("stack",)))}
    model["majority"] = {k: summarize(model_counts(items, maj, "M", kinds)) for k, kinds in
                         (("all", ("token", "stack")), ("token", ("token",)), ("stack", ("stack",)))}
    model["consensus_units"] = sum(len(d["C"]) for d in cons.values())
    model["majority_units"] = sum(len(d["M"]) for d in maj.values())
    # where the full annotators differ: status vs extent, direction, function words
    lem = {iid: {u["uid"]: u["lemma"] for u in it["units"] if u["kind"] == "token"} for iid, it in items.items()}
    dis = {}
    for a, b in itertools.combinations(FULL, 2):
        c = collections.Counter()
        for iid, by in ann.items():
            for uid, x in by[a].items():
                if uid not in by[b] or x == by[b][uid]:
                    continue
                y = by[b][uid]; c["differ"] += 1
                if (x is None) != (y is None):
                    c["status"] += 1; c[f"only_{b if x is None else a}"] += 1
                    c["status_function"] += lem[iid].get(uid) in FUNCTION
                else:
                    c["extent"] += 1
        dis[f"{a} x {b}"] = dict(c)
    res["disagreement"] = dis
    res["aligned_rate"] = {a: 100 * sum(v is not None for by in ann.values() for v in by[a].values())
                              / sum(len(by[a]) for by in ann.values()) for a in FULL}
    g = genre_of(items)
    for dim, ix in (("genre", 0), ("period", 1)):
        model[f"by_{dim}"] = {}
        for v in sorted({x[ix] for x in g.values()}):
            sel = lambda iid, v=v: g[iid][ix] == v
            model[f"by_{dim}"][v] = {who: summarize(model_counts(items, ann, who, only=sel)) for who in FULL}
    res["model"] = model
    cann = content_only(items, ann)
    ag = agreement(items, cann)
    res["content_words"] = {"agreement_pairs": ag["pairs"], "alpha_nominal_all": ag["alpha_nominal_all"],
                            "model": {who: summarize(model_counts(items, cann, who)) for who in res["sentences_per_annotator"]}}
    res["links"] = links_eval(items, ann)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
