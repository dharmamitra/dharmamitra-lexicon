#!/usr/bin/env python3
"""v2 mining for the Tibetan-target directions (sa-bo, zh-bo, en-bo) with the
qwen span-projection model + vLLM. Companion to mine_sazh.py (same output
schema and QC signals); prompts match the v2 training format exactly, with all
Tibetan converted Wylie -> Unicode at the prompt boundary (pipeline convention).

  python mine_direction.py --direction sa-bo --limit 300 --measure
  CUDA_VISIBLE_DEVICES=0 python mine_direction.py --direction sa-bo --part 0 --nparts 2
"""
import argparse, collections, glob, json, os, re, sys, time

V1 = "/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026"
CKPT = os.environ.get("DICTSPAN_CKPT", "/mnt/data/qwen35-9b-dictspan-mm")
OUT_BASE = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"

INSTRUCTION = ("Insert the numbered tags into the target line, marking the "
               "word(s) that translate each numbered source word. Copy the "
               "target verbatim and omit numbers without a clear equivalent.")
TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)
norm = lambda s: " ".join((s or "").split())

DIRS = {
    # ---- v3 regeneration (2026-10-07): MITRA-parallel v2 substrates built by prep_v2_substrate.py ----
    "sa-bo-v2": dict(sp="/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4/sa-bo", glob="substrate.ndjson",
                     pairids=None, kind="sanskrit", src_field="root", tgt_field="par"),
    "sa-zh-v2": dict(sp="/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4/sa-zh", glob="substrate.ndjson",
                     pairids=None, kind="sanskrit", src_field="root", tgt_field="par", target="chinese"),
    "sa-bo": dict(sp="sa-bo/sentence-pairs-enriched", glob="sa-bo_matches_*.ndjson",
                  pairids=None, kind="sanskrit", src_field="root", tgt_field="par"),
    "zh-bo": dict(sp="zh-bo/sentence-pairs", glob="bo-zh_matches_*.filtered.ndjson",
                  pairids="final-output/zh-bo.productive_pairids.txt",
                  kind="chinese", src_field="par", tgt_field="root"),
    "en-bo": dict(sp="en-bo", glob="84000_en-bo.ndjson",
                  pairids=None, kind="english", src_field="root", tgt_field="par"),
    # MonlamAI TM corpus (~/data/monlamai-mitra-mt) MINUS the slices mined elsewhere
    # (Lotsawa House -> en-bo-lh, 84000 -> en-bo, Tsadra -> en-bo-dnx, >=50% duplicates).
    # Substrate + per-text metadata: build/prep_monlam_substrate.py. Same row shape as the
    # 84000 substrate (Tibetan stored as Wylie, converted to Unicode here as usual).
    "en-bo-monlam": dict(sp="en-bo", glob="monlam_en-bo.ndjson",
                         pairids=None, kind="english", src_field="root", tgt_field="par"),
    # dharmanexus-english (EGS + Tsadra) x Tibetan, aligned in mitra-multilingual-matching.
    # segtext format: root_segtext = list of English strings, par_segtext = list of
    # Wylie Tibetan segments (joined before the standard unicode conversion).
    "en-bo-dnx": dict(sp="/mnt2/code/mitra-multilingual-matching/multilingual-2026",
                      glob=["en-bo_egs_matches_*.filtered.ndjson",
                            "en-bo_tsadra_matches_*.filtered.ndjson"], pairids=None,
                      kind="english", src_field="root", tgt_field="par", segtext=True),
    # Lotsawa House (lotsawahouse.org) x Tibetan, same segtext shape as en-bo-dnx.
    # Kept a separate direction so its evidence keeps the per-segment lotsawahouse.org
    # links, which are MANDATORY downstream (CC BY-NC 4.0 attribution).
    "en-bo-lh": dict(sp="/mnt2/code/mitra-multilingual-matching/multilingual-2026",
                     glob="en-bo_lh_matches_*.filtered.ndjson", pairids=None,
                     kind="english", src_field="root", tgt_field="par", segtext=True),
    # Mitrasamgraha Sanskrit x English (mitrasamgraha-dataset TSVs), Sanskrit side
    # enriched by our byt5 grammatical analyzer (enrich_saen.py: full_analysis).
    # ENGLISH target: no Wylie conversion, no verse heuristic. Needs the sa-en-
    # capable checkpoint: DICTSPAN_CKPT=/mnt/data/qwen35-9b-dictspan-saen-mm.
    "sa-en": dict(sp="/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-en/enriched",
                  glob="*.ndjson", pairids=None, kind="sanskrit", src_field="root",
                  tgt_field="par", target="english"),
    # mitra-data-collection skt-en/aligned MINUS Mitrasamgraha overlap, Japanese-derived English
    # and junk (build/prep_saen_mdc.py); enriched with enrich_saen.py (SAEN_FLAT=1). Same
    # English-target recipe + checkpoint as sa-en.
    "sa-en-mdc": dict(sp="/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-en-mdc/enriched",
                      glob="*.ndjson", pairids=None, kind="sanskrit", src_field="root",
                      tgt_field="par", target="english"),
}


def is_verse_bo(wylie):
    segs = [s for s in re.split(r"[/|]+", wylie) if s.strip()]
    if len(segs) < 4:
        return False
    counts = [len(s.split()) for s in segs]
    ln, n = collections.Counter(counts).most_common(1)[0]
    return ln in (7, 9, 11) and n / len(segs) >= 0.6


def is_verse_zh(zh):
    """verse heuristic of mine_sazh.py"""
    if "　　" in zh:
        return True
    segs = [x for x in re.split(r"[，、。！？；\s]+", zh) if x]
    if len(segs) >= 4:
        c = collections.Counter(len(x) for x in segs)
        ln, n = c.most_common(1)[0]
        if ln in (5, 7) and n / len(segs) >= 0.6:
            return True
    return False


def gen_inputs(direction, limit=0, input_file=None):
    import pyewts
    conv = pyewts.pyewts()
    cfg = DIRS[direction]
    pids = None
    if cfg["pairids"]:
        pids = set(open(os.path.join(V1, cfg["pairids"]), encoding="utf-8").read().split("\n"))
        pids.discard("")
    nseen = 0
    root = cfg["sp"] if os.path.isabs(cfg["sp"]) else os.path.join(V1, cfg["sp"])
    pats = cfg["glob"] if isinstance(cfg["glob"], list) else [cfg["glob"]]
    files = sorted({f for pat in pats for f in glob.glob(os.path.join(root, pat))})
    if input_file:            # explicit substrate file (e.g. a re-aligned subset) instead of the glob
        files = [input_file]
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            d = json.loads(line)
            if pids is not None and d.get("id") not in pids:
                continue
            if cfg.get("segtext"):
                d[f"{cfg['tgt_field']}_string"] = " ".join(d.get(f"{cfg['tgt_field']}_segtext") or [])
                d[f"{cfg['src_field']}_string"] = " ".join(d.get(f"{cfg['src_field']}_segtext") or [])
            wylie = (d.get(f"{cfg['tgt_field']}_string") or "").strip()
            if not wylie:
                continue
            english_target = cfg.get("target") == "english"
            if cfg["kind"] == "sanskrit":
                fa = d.get("full_analysis") or []
                toks = [(t.get("unsandhied", ""), t.get("lemma", "")) for t in fa if t.get("unsandhied")]
                if not toks:
                    continue
                skt = " ".join(f"<{i}>{s}</{i}>" for i, (s, l) in enumerate(toks, 1))
                lem = " ".join(f"<{i}>{l}</{i}>" for i, (s, l) in enumerate(toks, 1))
                src_block = f"SANSKRIT: {skt}\nLEMMAS: {lem}"
            elif cfg["kind"] == "chinese":
                words = (d.get("par_zh_segmented") or "").split()
                if not words:
                    continue
                toks = [(w, w) for w in words]
                src_block = "CHINESE: " + " ".join(f"<{i}>{w}</{i}>" for i, w in enumerate(words, 1))
            else:  # english (v2 build_unit junk filter)
                en = (d.get(f"{cfg['src_field']}_string") or "").strip()
                if not en or len(wylie) < 15 or len(en.split()) < 3:
                    continue
                letters = sum(c.isalpha() for c in wylie)
                if letters < 0.6 * len(wylie.replace(" ", "")):
                    continue
                words = en.split()
                toks = [(w, w) for w in words]
                src_block = "ENGLISH: " + " ".join(f"<{i}>{w}</{i}>" for i, w in enumerate(words, 1))
            if cfg.get("target") == "chinese":       # sa-zh: exactly the mine_sazh.py / training block
                zh = (d.get("par_string") or "").strip()
                seg = (d.get("par_zh_segmented") or "").strip()
                if not zh or not seg:
                    continue
                prompt = (f"{INSTRUCTION}\n\n{src_block}\nCHINESE: {zh}\nCHINESE_SEGMENTED: {seg}\nOUTPUT:\n")
                if len(prompt) > 6000:
                    continue
                yield {"id": d.get("id"), "toks": toks, "target": zh, "verse": is_verse_zh(zh),
                       "prompt": prompt, "src_segnr": d.get("root_segnr"), "tgt_segnr": d.get("par_segnr"),
                       "src_original": (d.get("root_string") or "").strip()}
                nseen += 1
                if limit and nseen >= limit:
                    return
                continue
            if english_target:
                bo = norm(wylie)   # plain English target
                if len(bo.split()) < 3 or sum(c.isalpha() for c in bo) < 0.6 * len(bo.replace(" ", "")):
                    continue
                prompt = f"{INSTRUCTION}\n\n{src_block}\nENGLISH: {bo}\nOUTPUT:\n"
            else:
                bo = norm(conv.toUnicode(wylie))
                prompt = f"{INSTRUCTION}\n\n{src_block}\nTIBETAN: {bo}\nOUTPUT:\n"
            if len(prompt) > 5000:   # unicode bo tokenizes denser; keep inside 2048-token ctx
                continue
            yield {"id": d.get("id"), "toks": toks, "target": bo,
                   "verse": False if english_target else is_verse_bo(wylie),
                   "prompt": prompt, "src_segnr": d.get(f"{cfg['src_field']}_segnr"),
                   "tgt_segnr": d.get(f"{cfg['tgt_field']}_segnr"),
                   "src_original": (d.get(f"{cfg['src_field']}_string") or "").strip()}
            nseen += 1
            if limit and nseen >= limit:
                return


def score_row(it, o):
    text = o.outputs[0].text.strip()
    lps = [list(lp.values())[0].logprob for lp in (o.outputs[0].logprobs or [])]
    conf = round(sum(lps) / len(lps), 3) if lps else -9.0
    grounded, nun, nums = {}, 0, []
    for m in TAG.finditer(text):
        n = int(m.group(1)); span = norm(m.group(2)).strip("་"); nums.append(n)
        if 1 <= n <= len(it["toks"]) and span:
            if span in it["target"]:
                grounded[n] = span
            else:
                nun += 1
    detag = norm(re.sub(r"</?\d+>", "", text))
    verbatim = detag == norm(it["target"])
    ung_frac = nun / max(len(nums), 1)
    numbering_bad = len(nums) != len(set(nums)) or any(n < 1 or n > len(it["toks"]) for n in nums)
    cov = len(grounded) / max(len(it["toks"]), 1)
    flag = bool(it["verse"] or conf < -0.45 or ung_frac > 0.25 or numbering_bad
                or cov < 0.30 or not verbatim)
    return {"id": it["id"], "target": it["target"], "src_original": it["src_original"],
            "src_segnr": it["src_segnr"], "tgt_segnr": it["tgt_segnr"],
            "toks": it["toks"], "output": text, "conf": conf,
            "ung_frac": round(ung_frac, 2), "coverage": round(cov, 2),
            "verse": it["verse"], "numbering_bad": numbering_bad,
            "verbatim": verbatim, "flagged": flag,
            "aligns": {str(n): grounded[n] for n in sorted(grounded)}}


def _print_qc(direction, rows):
    n = max(len(rows), 1)
    print(f"[QC {direction}] verbatim-copy: {100*sum(r['verbatim'] for r in rows)/n:.1f}%  "
          f"numbering-bad: {100*sum(r['numbering_bad'] for r in rows)/n:.1f}%  "
          f"flagged: {100*sum(r['flagged'] for r in rows)/n:.1f}%  "
          f"spans/sentence: {sum(len(r['aligns']) for r in rows)/n:.2f}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=sorted(DIRS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print N built prompts, no GPU")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--input", default=None,
                    help="substrate ndjson to mine instead of the direction's glob (same row format)")
    args = ap.parse_args()
    out = args.out or os.path.join(OUT_BASE, args.direction, "mined.ndjson")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    items = [it for i, it in enumerate(gen_inputs(args.direction, args.limit, args.input))
             if i % args.nparts == args.part]
    print(f"[{args.direction} part {args.part}/{args.nparts}] {len(items)} sentences", file=sys.stderr)
    if args.dry:
        for it in items[:2]:
            print("----\n" + it["prompt"])
        return
    if not items:   # empty delta (v4 round 2): nothing to mine, leave an empty part file
        out_path = out if args.nparts == 1 else f"{out}.part{args.part}"
        open(out_path, "a").close()
        print(f"[{args.direction} part {args.part}] DONE 0 -> {out_path} (empty input)")
        return

    # exact token-length gate: one over-long prompt aborts the whole llm.generate,
    # so pre-tokenize (parallel fast tokenizer), drop the tail, and hand vLLM the
    # token ids directly. 1900 leaves room for the tagged output within 2048... the
    # generation is cut at the ctx limit for near-cap prompts, which QC then flags.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok([it["prompt"] for it in items]).input_ids
    keep = [i for i, t in enumerate(ids) if len(t) <= 1900]
    if len(keep) < len(items):
        print(f"[{args.direction}] dropped {len(items) - len(keep)} prompts > 1900 tokens",
              file=sys.stderr)
    items = [items[i] for i in keep]
    ids = [ids[i] for i in keep]

    out_path = out if args.nparts == 1 else f"{out}.part{args.part}"
    # checkpointing: generate in chunks, append rows after each chunk. Item order
    # is deterministic, so resume = skip as many items as lines already written.
    done = 0
    if not args.measure and os.path.exists(out_path):
        done = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"[{args.direction} part {args.part}] resuming past {done} already-written rows",
              file=sys.stderr)
    items, ids = items[done:], ids[done:]

    from vllm import LLM, SamplingParams, TokensPrompt
    # MAX_NUM_SEQS: cap the scheduler batch (default 1024 warms the sampler with 1024 dummy
    # requests, which OOMs on a card with < ~35 GB free even when the KV cache fits).
    llm = LLM(model=CKPT, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
              max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "1024")))
    sp = SamplingParams(temperature=0.0, max_tokens=512, logprobs=1, stop=["\n"])

    CHUNK = 5000
    t0 = time.time()
    outs = []
    fh = None if args.measure else open(out_path, "a", encoding="utf-8")
    for c0 in range(0, len(ids), CHUNK):
        chunk_outs = llm.generate([TokensPrompt(prompt_token_ids=t) for t in ids[c0:c0 + CHUNK]],
                                  sp, use_tqdm=True)
        outs.extend(chunk_outs)
        if fh is not None:
            for it, o in zip(items[c0:c0 + CHUNK], chunk_outs):
                fh.write(json.dumps(score_row(it, o), ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{args.direction} part {args.part}] checkpoint: "
                  f"{done + c0 + len(chunk_outs)} rows written", file=sys.stderr)
    if fh is not None:
        fh.close()
    rate = len(items) / max(time.time() - t0, 1e-9)
    print(f"[{args.direction}] {len(items)} sentences at {rate:.1f} sent/s", file=sys.stderr)

    if args.measure:
        rows = [score_row(it, o) for it, o in zip(items, outs)]
        _print_qc(args.direction, rows)
        for r in rows[:3]:
            print(json.dumps({"TIBETAN": r["target"], "OUTPUT": r["output"],
                              "aligns": r["aligns"], "conf": r["conf"]}, ensure_ascii=False))
        return

    # QC over the FULL part file (includes rows written before a resume)
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    _print_qc(args.direction, rows)
    nflag = sum(r["flagged"] for r in rows)
    print(f"[{args.direction} part {args.part}] DONE {len(rows)} -> {out_path} "
          f"({nflag} flagged, {100*nflag/max(len(rows),1):.0f}%)")


if __name__ == "__main__":
    main()
