"""Link F1 of student outputs against the expert gold alignments (Table baselines, last column).

Links (Sanskrit word, English \\w+ token) are taken from the numbered tags of each output and scored against each
annotator's word units, as in gold_eval.links_eval; the reported figure is the mean over annotators A, B and C.
An output that does not reproduce the English verbatim is, by default, located span by span through text search,
as in the held-out link F1 of the statistical baselines; --no-fallback scores such an output as empty.

Usage: python3 score_students.py [STUDENT_DIR] [--no-fallback]   (default: data/gold/students)
"""
import collections, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_eval as G

args = [a for a in sys.argv[1:] if not a.startswith("--")]
S = args[0] if args else os.path.join(G.DATA, "students")
FALLBACK = "--no-fallback" not in sys.argv
TAGRE = re.compile(r"<(\d+(?:,\d+)*)>|</(\d+(?:,\d+)*)>")


def spans(out, target):
    """number -> [(start, end)] character spans in target; None if not verbatim and no fallback."""
    res = collections.defaultdict(list); i = 0; buf = ""; open_ = {}
    for m in TAGRE.finditer(out):
        buf += out[i:m.start()]; i = m.end()
        if m.group(1): open_[m.group(1)] = len(buf)
        elif m.group(2) in open_:
            s = open_.pop(m.group(2))
            for k in m.group(2).split(","): res[int(k)].append((s, len(buf)))
    buf += out[i:]
    if buf == target: return res
    if not FALLBACK: return None
    res = collections.defaultdict(list)
    for m in re.finditer(r"<(\d+(?:,\d+)*)>(.*?)</\1>", out, re.DOTALL):
        q = target.find(m.group(2)) if m.group(2) else -1
        if q >= 0:
            for k in m.group(1).split(","): res[int(k)].append((q, q + len(m.group(2))))
    return res


items, ann = G.load()
for path in sorted(glob.glob(os.path.join(S, "*.ndjson"))):
    R = {d["id"]: d for d in map(json.loads, open(path, encoding="utf-8"))}
    nonverb = sum(1 for iid in ann if R[iid]["output"] and re.sub(r"</?[\d,]+>", "", R[iid]["output"]) != items[iid]["english"])
    per = {}
    for who in G.FULL:
        tp = npred = ngold = 0
        for iid, by in ann.items():
            it = items[iid]; et = G.tok_en(it["english"]); word = [bool(re.match(r"\w", w)) for w, _, _ in et]
            def to_links(i, s, e): return {(i, j) for j, (_, a, b) in enumerate(et) if a < e and b > s and word[j]}
            def rng(r): return it["en_tokens"][r[0]]["s"], it["en_tokens"][r[1]]["e"]
            sp = spans(R[iid]["output"], it["english"]) or {}
            gold, stud = set(), set()
            for u in it["units"]:
                if u["kind"] != "token" or u["uid"] not in by[who]: continue
                i = u["indices"][0] - 1
                if by[who][u["uid"]]: gold |= to_links(i, *rng(by[who][u["uid"]]))
                for s, e in sp.get(i + 1, []): stud |= to_links(i, s, e)
            tp += len(stud & gold); npred += len(stud); ngold += len(gold)
        per[who] = 200 * tp / (npred + ngold) if npred + ngold else 0
    print(f"{os.path.basename(path)[:-7]:28s} non-verbatim {nonverb:2d}  " + "  ".join(f"{w} {v:.1f}" for w, v in per.items())
          + f"  mean {sum(per.values()) / len(per):.1f}")
