#!/usr/bin/env python3
"""Shared helpers for the reference-dictionary agreement evaluation.

- byte-range parallel streaming of large ndjson files (CPU only, no GPU)
- Sanskrit key normalisation (IAST) and nominative->stem candidate generation
- frequency bands, metric computation for one headword, aggregation
- mismatch-sample formatting

Everything here is deterministic; sampling uses random.Random(42) on a sorted pool.
"""
import json, os, re, random, unicodedata, collections
from multiprocessing import Pool

MINED_DIR = os.environ.get("EVAL_MINED_DIR", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/final-output")
OUT_DIR = "/home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020"
NPROC = int(os.environ.get("EVAL_NPROC", "48"))
SEED = 42

BANDS = [("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19), ("20-99", 20, 99),
         ("100-999", 100, 999), (">=1000", 1000, 10 ** 12)]


def band_of(n):
    for name, lo, hi in BANDS:
        if lo <= n <= hi:
            return name
    return None


# ---------------------------------------------------------------- file streaming
def _chunks(path, nchunks):
    size = os.path.getsize(path)
    step = max(1, size // nchunks)
    bounds = [0]
    with open(path, "rb") as f:
        for i in range(1, nchunks):
            f.seek(i * step)
            f.readline()
            p = f.tell()
            if p > bounds[-1] and p < size:
                bounds.append(p)
    bounds.append(size)
    return [(path, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def iter_lines(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line


def parallel_scan(path, worker, nproc=None, nchunks=None):
    """worker((path,start,end)) -> picklable result; returns list of results in file order."""
    nproc = nproc or NPROC
    chunks = _chunks(path, nchunks or nproc * 4)
    with Pool(nproc) as pool:
        return pool.map(worker, chunks, chunksize=1)


# ---------------------------------------------------------------- Sanskrit keys
_ANUSVARA_CLASS = [("kg", "ṅ"), ("cj", "ñ"), ("ṭḍ", "ṇ"), ("td", "n"), ("pb", "m")]
_ANU_RE = re.compile(r"[ṃṁ](?=([kgcjṭḍtdpb]))")


def _anu(m):
    c = m.group(1)
    for cls, nasal in _ANUSVARA_CLASS:
        if c in cls:
            return nasal
    return "ṃ"


def sa_key(s):
    """Normalised IAST comparison key: NFC, lowercase, anusvara before a stop -> homorganic
    nasal (saṃgha == saṅgha), ṁ -> ṃ, strip surrounding whitespace/punctuation."""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = s.replace("ṁ", "ṃ").replace("ń", "ṅ")
    s = s.strip(" .,;:!?'\"()[]|/0123456789")
    s = _ANU_RE.sub(_anu, s)
    return s


def is_single_word(hw):
    return bool(hw) and " " not in hw and "-" not in hw and "_" not in hw


def stem_candidates(form):
    """Reference dictionaries (Negi, Apte, partly Hirakawa) cite Sanskrit in the nominative.
    Return an ORDERED list of (candidate_key, rule) for a normalised form. The first is the
    form itself. A candidate only ever takes effect if it equals a mined headword."""
    f = form
    out = [(f, "exact")]
    if len(f) < 3:
        return out
    if f.endswith("ḥ"):
        b = f[:-1]
        out.append((b, "strip-ḥ"))                       # dharmaḥ->dharma, agniḥ->agni, vāyuḥ->vāyu
        if b.endswith(("a", "i", "u")):
            out.append((b + "s", "ḥ->s"))                 # manaḥ->manas, cakṣuḥ->cakṣus
        if b.endswith("ā"):
            out.append((b[:-1] + "a", "āḥ->a"))           # plural nominatives devāḥ->deva
    if f.endswith(("am", "aṃ")):
        out.append((f[:-1], "strip-m"))                   # phalam->phala
    elif f.endswith(("m", "ṃ")) and len(f) > 3:
        out.append((f[:-1], "strip-m"))                   # accusatives / -īm etc. (rare)
    if f.endswith("ā"):
        out.append((f[:-1] + "an", "ā->an"))              # rājā->rājan, ātmā->ātman
        out.append((f[:-1] + "ṛ", "ā->ṛ"))                # kartā->kartṛ
    if f.endswith("ī"):
        out.append((f[:-1] + "in", "ī->in"))              # yogī->yogin
    if f.endswith("a"):
        out.append((f + "n", "a->an"))                    # karma->karman, nāma->nāman
    if f.endswith(("vān", "mān")):
        out.append((f[:-2] + "at", "vān->vat"))           # bhagavān->bhagavat
    seen, res = set(), []
    for c, r in out:
        if c and c not in seen:
            seen.add(c)
            res.append((c, r))
    return res


# ---------------------------------------------------------------- metrics
def metrics_from_matchsets(counts, matchsets, nref):
    """counts[i]   = attestation count of mined sense i (file order = rank order)
    matchsets[i] = set of reference-equivalent ids matched by sense i (empty = no match)
    nref         = number of reference equivalents of the headword.
    Returns the raw per-headword quantities that Agg accumulates."""
    hit = [bool(m) for m in matchsets]
    tot = sum(counts)
    mc = sum(c for c, h in zip(counts, hit) if h)
    rec_all = set().union(*matchsets) if matchsets else set()
    rec_10 = set().union(*matchsets[:10]) if matchsets else set()
    first = next((i + 1 for i, h in enumerate(hit) if h), None)
    return {"p1": int(any(hit[:1])), "p3": int(any(hit[:3])), "p5": int(any(hit[:5])),
            "mc": mc, "tot": tot, "nref": nref, "rec_all": len(rec_all), "rec_10": len(rec_10),
            "nsenses": len(counts), "first_hit_rank": first}


class Agg:
    """Accumulates per-headword records per band (+ 'all')."""

    def __init__(self):
        self.d = collections.defaultdict(lambda: collections.defaultdict(float))

    def add(self, n, m):
        for b in ("all", band_of(n)):
            a = self.d[b]
            a["headwords"] += 1
            a["p1_hits"] += m["p1"]; a["p3_hits"] += m["p3"]; a["p5_hits"] += m["p5"]
            a["matched_count"] += m["mc"]; a["total_count"] += m["tot"]
            a["macro_prec_sum"] += (m["mc"] / m["tot"]) if m["tot"] else 0.0
            if m["nref"]:                  # nref == 0: reference has no enumerable equivalents (sa-en prose)
                a["ref_equivalents"] += m["nref"]
                a["ref_recalled_all"] += m["rec_all"]; a["ref_recalled_top10"] += m["rec_10"]
                a["macro_recall_all_sum"] += m["rec_all"] / m["nref"]
                a["macro_recall_top10_sum"] += m["rec_10"] / m["nref"]
            a["senses"] += m["nsenses"]

    def result(self):
        out = {}
        for b in ["all"] + [x[0] for x in BANDS]:
            a = self.d.get(b)
            if not a or not a["headwords"]:
                out[b] = {"headwords": 0}
                continue
            H = a["headwords"]
            out[b] = {
                "headwords": int(H),
                "P@1": a["p1_hits"] / H, "P@3": a["p3_hits"] / H, "P@5": a["p5_hits"] / H,
                "attw_precision_micro": a["matched_count"] / a["total_count"] if a["total_count"] else None,
                "attw_precision_macro": a["macro_prec_sum"] / H,
                "recall_all_macro": a["macro_recall_all_sum"] / H if a["ref_equivalents"] else None,
                "recall_top10_macro": a["macro_recall_top10_sum"] / H if a["ref_equivalents"] else None,
                "recall_all_micro": a["ref_recalled_all"] / a["ref_equivalents"] if a["ref_equivalents"] else None,
                "recall_top10_micro": a["ref_recalled_top10"] / a["ref_equivalents"] if a["ref_equivalents"] else None,
                "counts": {"p1_hits": int(a["p1_hits"]), "p3_hits": int(a["p3_hits"]),
                           "p5_hits": int(a["p5_hits"]), "matched_count": int(a["matched_count"]),
                           "total_count": int(a["total_count"]),
                           "ref_equivalents": int(a["ref_equivalents"]),
                           "ref_recalled_all": int(a["ref_recalled_all"]),
                           "ref_recalled_top10": int(a["ref_recalled_top10"]),
                           "mined_senses": int(a["senses"])},
            }
        return out


def band_counts(ns):
    c = collections.Counter(band_of(n) for n in ns)
    out = {b[0]: c.get(b[0], 0) for b in BANDS}
    out["all"] = len(ns)
    return out


# ---------------------------------------------------------------- reporting helpers
def highlight(ex):
    if not ex:
        return "(no example)"
    t = ex.get("target", "")
    s, e = ex.get("start"), ex.get("end")
    if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(t):
        lo, hi = max(0, s - 110), min(len(t), e + 110)
        return ("…" if lo else "") + t[lo:s] + "⟦" + t[s:e] + "⟧" + t[e:hi] + ("…" if hi < len(t) else "")
    return t[:220]


def sample_pool(pool, k=40):
    pool = sorted(pool, key=lambda r: r["sa"])
    rnd = random.Random(SEED)
    return pool if len(pool) <= k else rnd.sample(pool, k)


def fmt_pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def md_table(res, title):
    """res: band -> metrics (from Agg.result())."""
    L = [f"**{title}**", "",
         "| band (n) | #hw | P@1 | P@3 | P@5 | att-w prec (micro) | att-w prec (macro) | recall@all (macro) | recall@top10 (macro) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in [x[0] for x in BANDS] + ["all"]:
        m = res.get(b, {})
        if not m.get("headwords"):
            L.append(f"| {b} | 0 | – | – | – | – | – | – | – |")
            continue
        L.append(f"| {b} | {m['headwords']} | {fmt_pct(m['P@1'])} | {fmt_pct(m['P@3'])} | {fmt_pct(m['P@5'])} | "
                 f"{fmt_pct(m['attw_precision_micro'])} | {fmt_pct(m['attw_precision_macro'])} | "
                 f"{fmt_pct(m['recall_all_macro'])} | {fmt_pct(m['recall_top10_macro'])} |")
    return "\n".join(L) + "\n"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- generic evaluation driver
CFG = {}     # set by the direction script BEFORE parallel_scan (workers are forked and inherit it)


def eval_worker(chunk):
    """CFG: sense_field, index (key -> refs), evaluate(sa, key, n, senses, refs, ex0) -> record."""
    field, index, evaluate = CFG["sense_field"], CFG["index"], CFG["evaluate"]
    light, recs = [], []
    nlines = 0
    for line in iter_lines(*chunk):
        nlines += 1
        d = json.loads(line)
        sa = d["sa"]
        if not is_single_word(sa):
            continue
        key = sa_key(sa)
        n = d["n"]
        in_ref = key in index
        light.append((key, n, in_ref))
        if not in_ref:
            continue
        senses = [(s[field], s["count"]) for s in d["senses"]]
        ex0 = d["senses"][0]["ex"][0] if d["senses"] and d["senses"][0].get("ex") else None
        recs.append(evaluate(sa, key, n, senses, index[key], ex0))
    return light, recs, nlines


STATS = {}


def run_eval(mined_file):
    parts = parallel_scan(os.path.join(MINED_DIR, mined_file), eval_worker)
    STATS["mined_file_lines"] = sum(p[2] for p in parts)
    light = [x for p in parts for x in p[0]]
    recs = [x for p in parts for x in p[1]]
    return light, recs


def aggregate(recs, variants):
    out = {}
    for v in variants:
        a = Agg()
        for r in recs:
            if v in r["m"]:
                a.add(r["n"], r["m"][v])
        out[v] = a.result()
    return out


def sanity_print(recs, variant, k=10, fmt=None):
    rnd = random.Random(SEED)
    pool = sorted(recs, key=lambda r: r["sa"])
    for label, want in (("MATCHED top-1", 1), ("UNMATCHED top-1", 0)):
        sel = [r for r in pool if variant in r["m"] and r["m"][variant]["p1"] == want]
        print(f"--- {k} random {label} [{variant}] (of {len(sel)})")
        for r in rnd.sample(sel, min(k, len(sel))):
            print("   ", fmt(r) if fmt else (r["sa"], r["n"], r["top"][:1], r["refs_show"][:6]))


def shuffled_index(index, light, stratum=None):
    """Chance-agreement control: every E key receives the reference entry of ANOTHER E key of the same
    frequency band (and same `stratum(key)` if given). Deterministic: keys sorted, random.Random(42).shuffle,
    then each key takes the entry of its successor in the shuffled cycle (so never its own, if the stratum has
    >1 key)."""
    groups = collections.defaultdict(list)
    seen = set()
    for key, n, e in light:
        if e and key not in seen:
            seen.add(key)
            groups[(band_of(n), stratum(key) if stratum else None)].append(key)
    rnd = random.Random(SEED)
    out = {}
    for g in sorted(groups, key=str):
        ks = sorted(groups[g])
        rnd.shuffle(ks)
        for i, k in enumerate(ks):
            out[k] = index[ks[(i + 1) % len(ks)]]
    return out
