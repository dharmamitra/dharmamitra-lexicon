#!/usr/bin/env python3
"""Sanskrit grammatical enrichment of the Mitrasamgraha sa-en sentence pairs.

Uses EXACTLY the dharmanexus byt5 CTranslate2 analyzer (analyze_worker.py:
clean_input -> "SLM " mode -> full_analysis [{unsandhied, lemma, tag}]), i.e.
the same full_analysis schema the sa-bo / sa-zh substrate carries, so the
downstream seed / mining code can treat sa-en like the other Sanskrit-source
directions.

Calibration (scratch, 60 sents/bin): the analyzer is reliable up to ~200 chars
of cleaned input and truncates/garbles beyond that (28/60 bad at 200-280 chars,
44/60 at 280-360), so every sentence is CHUNKED at danda / punctuation, then at
spaces, into pieces <= CHUNK chars; the per-chunk analyses are concatenated.

  python enrich_saen.py --candidates 2400        # per-file-capped sample of train
                                                  #   -> sa-en/seed_candidates.ndjson
  CUDA_VISIBLE_DEVICES=0 python enrich_saen.py --shard 0 --num-shards 2
                                                  #   -> sa-en/enriched/<file>.ndjson

Row schema: id, file, split, license, line, root_string (Sanskrit as in the
TSV), par_string (English), n_chunks, full_analysis, unsandhied, lemmatized.
"""
import argparse, glob, json, os, random, re, sys, time

PIPE = os.path.expanduser("~/code/dharmanexus-sanskrit-processing")
sys.path.insert(0, PIPE)
import analyze_worker as W  # clean_input, read_skt_tags, make_postprocess, DEFAULT_MODEL

DATA = os.environ.get("SAEN_DATA", "/home/sebastian/code/mitrasamgraha-dataset/data")
OUT = os.environ.get("SAEN_OUT", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-en")
# SAEN_FLAT=1: DATA holds plain <file>.tsv (no split/license dirs), e.g. the sa-en-mdc
# substrate (mitra-data-collection skt-en/aligned minus exclusions). Rows get split="mdc".
FLAT = os.environ.get("SAEN_FLAT") == "1"
CHUNK = 150
SEED = 42


def list_files():
    files = []
    if FLAT:
        for fp in sorted(glob.glob(os.path.join(DATA, "*.tsv"))):
            files.append(dict(path=fp, rel="mdc/" + os.path.basename(fp), split="mdc", license=""))
        return files
    for split in ("train", "val", "test"):
        for fp in sorted(glob.glob(os.path.join(DATA, split, "**", "*.tsv"), recursive=True)):
            rel = os.path.relpath(fp, DATA)
            parts = rel.split(os.sep)
            lic = parts[1] if len(parts) == 3 else ""
            files.append(dict(path=fp, rel=rel, split=split, license=lic))
    return files


def read_pairs(f):
    with open(f["path"], encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            sa, en = p[0].strip(), p[1].strip()
            if not sa or not en:
                continue
            yield {"id": f"{f['rel']}:{i}", "file": f["rel"], "split": f["split"],
                   "license": f["license"], "line": i, "root_string": sa, "par_string": en}


_PUNCT = re.compile(r"\s*(?://|\|\||[/|।॥])\s*")


def chunk_sanskrit(sa, limit=CHUNK):
    """Clean (analyzer input rules) then cut into <= limit-char pieces. Cut points:
    danda-like punctuation first, then spaces (never before an avagraha token)."""
    pieces = []
    for seg in _PUNCT.split(sa):
        c = W.clean_input(seg)
        if not c:
            continue
        if len(c) <= limit:
            pieces.append(c)
            continue
        cur = []
        for w in c.split():
            cand = " ".join(cur + [w])
            if cur and len(cand) > limit and not w.startswith("'"):
                pieces.append(" ".join(cur)); cur = [w]
            else:
                cur.append(w)
        if cur:
            pieces.append(" ".join(cur))
    # re-merge adjacent short pieces (punctuation splits can be tiny)
    merged = []
    for p in pieces:
        if merged and len(merged[-1]) + 1 + len(p) <= limit:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


class Analyzer:
    def __init__(self, batch=256):
        import ctranslate2, transformers
        self.post = W.make_postprocess(W.read_skt_tags(os.path.join(PIPE, "data", "sanskrit_tags.tsv")))
        self.tok = transformers.AutoTokenizer.from_pretrained("chronbmm/sanskrit5-multitask")
        self.tr = ctranslate2.Translator(W.DEFAULT_MODEL, device="cuda", device_index=[0],
                                         intra_threads=4, compute_type="int8_float16")
        self.batch = batch

    def analyze(self, chunks):
        """chunks: list[str] (cleaned, <= CHUNK chars) -> list[full_analysis]."""
        uniq = sorted(set(chunks), key=len)
        out = {}
        for i in range(0, len(uniq), self.batch):
            b = uniq[i:i + self.batch]
            enc = self.tok(["SLM " + t for t in b], truncation=True, max_length=512)
            toks = [self.tok.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]
            res = self.tr.translate_batch(toks, max_decoding_length=768, beam_size=1,
                                          max_batch_size=self.batch)
            for t, r in zip(b, res):
                out[t] = self.post(self.tok.decode(self.tok.convert_tokens_to_ids(r.hypotheses[0])))
        return [out[c] for c in chunks]


def enrich_rows(an, rows):
    flat, spans = [], []
    for r in rows:
        ch = chunk_sanskrit(r["root_string"])
        spans.append((len(flat), len(ch))); flat.extend(ch)
    fas = an.analyze(flat) if flat else []
    for r, (s, n) in zip(rows, spans):
        fa = [t for k in range(s, s + n) for t in fas[k]]
        r["n_chunks"] = n
        r["full_analysis"] = fa
        r["unsandhied"] = " ".join(t["unsandhied"] for t in fa)
        r["lemmatized"] = " ".join(t["lemma"] for t in fa)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=0,
                    help="sample N train pairs (capped per file) for Gemini seed mining")
    ap.add_argument("--per-file-cap", type=int, default=80)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--rows-per-batch", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = list_files()
    log = lambda m: print(f"[enrich shard {args.shard}] {m}", file=sys.stderr, flush=True)

    if args.candidates:
        rng = random.Random(SEED)
        pool = []
        for f in files:
            if f["split"] != "train":
                continue
            rows = [r for r in read_pairs(f)
                    if 30 <= len(r["root_string"]) <= 260 and 4 <= len(r["par_string"].split()) <= 60]
            rng.shuffle(rows)
            pool.extend(rows[:args.per_file_cap])
        rng.shuffle(pool)
        pool = pool[:args.candidates]
        log(f"{len(pool)} candidate rows from {len(files)} files")
        an = Analyzer()
        t0 = time.time()
        with open(os.path.join(OUT, "seed_candidates.ndjson"), "w", encoding="utf-8") as fh:
            for i in range(0, len(pool), args.rows_per_batch):
                for r in enrich_rows(an, pool[i:i + args.rows_per_batch]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"DONE candidates in {time.time()-t0:.0f}s -> {OUT}/seed_candidates.ndjson")
        return

    odir = os.path.join(OUT, "enriched"); os.makedirs(odir, exist_ok=True)
    mine = files[args.shard::args.num_shards]
    an = Analyzer()
    t0, nrows = time.time(), 0
    for f in mine:
        op = os.path.join(odir, f["rel"].replace(os.sep, "__") + ".ndjson")
        if os.path.exists(op):
            log(f"skip (done) {f['rel']}"); continue
        rows = list(read_pairs(f))
        tmp = op + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for i in range(0, len(rows), args.rows_per_batch):
                for r in enrich_rows(an, rows[i:i + args.rows_per_batch]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                nrows += min(args.rows_per_batch, len(rows) - i)
                if nrows % 10000 < args.rows_per_batch:
                    log(f"{nrows} rows, {nrows/(time.time()-t0):.0f} rows/s ({f['rel']})")
        os.replace(tmp, op)
        log(f"wrote {len(rows)} rows -> {os.path.basename(op)}")
    log(f"DONE {nrows} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
