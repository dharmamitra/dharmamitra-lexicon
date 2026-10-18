#!/usr/bin/env python3
"""Strict-prompt instruction-seed generation for the qwen dictionary-extraction
finetune (v2 of the gemma-2-mitra seed mining).

Differences vs mine_spans.py (v1), addressing over-generation of bad matches:
  - STRICT prompts: tag only on high confidence; omitting is the expected
    default; explicit per-language stopword lists that must never be latched.
  - gemini-3.6-flash with a high thinking budget (16384).
  - 2,000 instructions per language pair (sa-bo, sa-zh, bo-zh).
  - Same data plumbing as v1: Sanskrit grammar enrichment (full_analysis
    unsandhied+lemma), Chinese kundoku segmentation (par_zh_segmented) as
    tagging aid, verbatim span-grounding gate, reservoir sampling.

  python generate_seeds.py --direction sa-bo --smoke 3     # live examples
  python generate_seeds.py --direction sa-bo               # full 2k
  python generate_seeds.py --all                           # all three pairs
"""
import argparse, glob, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

V1 = "/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026"
sys.path.insert(0, V1)
from mine_spans import (DIRS, API_KEY as _V1_KEY, TAG, build_unit, parse_output,
                        reconstruct, length_filter, reservoir)

API_KEY = os.environ.get("GEMINI_API_KEY", _V1_KEY)

BASE = Path(V1)   # v1's BASE points at a stale ~/code path; data lives on /mnt2

# en-bo direction (not in v1 DIRS): English query -> Tibetan target
DIRS = dict(DIRS)
DIRS["en-bo"] = dict(data="en-bo", glob="84000_en-bo.ndjson",
                     q_field="root", q_kind="english", t_field="par", t_kind="tibetan")

OUT_BASE = Path("/mnt2/code/mitra-dictionaries/qwen-dictionaries")
MODEL_ID = "gemini-3.6-flash"
THINKING_BUDGET = 16384          # high thinking budget per user spec
SEED = 42
N_SEEDS = 2000                   # instructions per language pair
N_SAMPLE = int(os.environ.get("N_SAMPLE", "2800"))
WORKERS = int(os.environ.get("GEMINI_WORKERS", "40"))

# ---- stopword lists (never latch to these) ----
SA_STOP = ("ca, vā, na, hi, tu, eva, api, iti, atha, khalu, punar, sma, ha, "
           "cet, tathā, yathā, tatra, tadā, ataḥ, kim, cit, cid, api, tat, "
           "sa, te, ye, yad, idam, etad, kaścit")
BO_STOP = ("ni, kyi, gyi, gi, kyis, gyis, gis, yi, yis, 'i, du, tu, su, ru, "
           "la, las, nas, dang, pa, pa'i, ba, ba'i, po, mo, par, bar, pas, "
           "bas, ste, te, de, cing, zhing, shing, ces, zhes, yin, yod, med, "
           "do, 'o, so, to, ro, go, ngo, lo, 'am, 'ang, yang, kyang, zhig, "
           "cig, rnams, dag, can, nyid, tsam, na, no, der, dir, 'di")
ZH_STOP = ("之 其 者 而 也 矣 於 于 以 為 是 故 所 若 乃 即 亦 復 已 耳 焉 "
           "哉 乎 與 及 則 但 更 皆 悉 諸 等 中 時 當 應 得 有 無 不")
EN_STOP = ("the, a, an, of, to, in, and, or, is, are, was, were, be, been, "
           "that, this, these, those, it, its, by, with, for, as, on, at, "
           "from, into, then, thus, so, such, which, who, whom, what, when, "
           "will, shall, would, should, may, might, have, has, had, do, does, not")

STRICT_RULES = """CONFIDENCE POLICY (the most important rules):
A) Tag a number ONLY if you are highly confident the span is the actual translation of that word. If you are uncertain, OMIT the number entirely. It is EXPECTED and CORRECT that many numbers remain untagged - precision matters far more than coverage here.
B) NEVER tag a span that consists only of the function words / grammatical particles listed below, and NEVER attach them to a content span unless they are genuinely part of the translation:
   - Sanskrit function words (never treat as content, never use as anchor): {sa_stop}
   - Tibetan function syllables (never tag alone, never latch onto): {bo_stop}
   - Chinese function characters (never tag alone, never latch onto): {zh_stop}
C) If the sentence pair looks like a loose paraphrase rather than a translation, tag only the words whose correspondence is unmistakable (names, technical terms, numerals) and omit the rest."""

P_SA_BO = """You are an expert Sanskrit-Tibetan philologist doing high-precision word alignment.
You are given a Sanskrit sentence with each word in a numbered tag <1>..</1> (SANSKRIT), the same words as lemmas with the SAME numbers (LEMMAS), and the Tibetan translation in Wylie (TIBETAN).
TASK: Reproduce the TIBETAN line EXACTLY and insert the numbered tags around the Tibetan word(s) translating each numbered Sanskrit word.
""" + STRICT_RULES + """
FORMAT RULES: 1) Copy the Tibetan verbatim - never translate/reorder/add/delete it; insert ONLY tags. 2) <n> wraps the Tibetan translating Sanskrit word n. 3) A span may cover several consecutive Tibetan words. 4) Tags will be out of numerical order; never move Tibetan to make them sequential. 5) Each Tibetan word in at most one tag; no overlap/nesting. 6) Omit numbers with no confident Tibetan equivalent; never emit <n></n>. 7) Balanced tags. 8) Output ONLY the tagged Tibetan line.
EXAMPLE
SANSKRIT: <1>cittaṃ</1> <2>kāśyapa</2> <3>na</3> <4>labhyate</4>
LEMMAS: <1>citta</1> <2>kāśyapa</2> <3>na</3> <4>labh</4>
TIBETAN: 'od srungs sems ni mi rnyed do
OUTPUT: <2>'od srungs</2> <1>sems</1> ni <3>mi</3> <4>rnyed do</4>
(Note: <3>mi</3> is kept because negation na->mi is an unmistakable correspondence; ni and do stay untagged as function syllables.)
NOW DO THIS ONE
SANSKRIT: {sanskrit_tagged}
LEMMAS: {lemmas_tagged}
TIBETAN: {target}
OUTPUT:"""

P_SA_ZH = """You are an expert Sanskrit-Chinese philologist doing high-precision word alignment.
You are given a Sanskrit sentence with each word in a numbered tag <1>..</1> (SANSKRIT), the same words as lemmas with the SAME numbers (LEMMAS), the Chinese translation as written (CHINESE), and the same Chinese with word boundaries marked by spaces as an aid only (CHINESE_SEGMENTED).
TASK: Reproduce the CHINESE line EXACTLY and insert the numbered tags around the Chinese character(s) that translate each numbered Sanskrit word.
""" + STRICT_RULES + """
FORMAT RULES: 1) Copy the original CHINESE verbatim; insert ONLY tags. Use CHINESE_SEGMENTED only as a boundary hint. 2) <n> wraps the Chinese translating Sanskrit word n. 3) A span may cover one or several consecutive characters. 4) Tags will be out of numerical order. 5) Each character in at most one tag; no overlap/nesting. 6) Punctuation stays untagged. 7) Omit numbers with no confident Chinese equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged Chinese line.
EXAMPLE
SANSKRIT: <1>buddhaḥ</1> <2>dharmaṃ</2> <3>deśayati</3>
LEMMAS: <1>buddha</1> <2>dharma</2> <3>diś</3>
CHINESE: 佛為眾生說法。
CHINESE_SEGMENTED: 佛 為 眾生 說 法
OUTPUT: <1>佛</1>為眾生<3>說</3><2>法</2>。
(Note: 為 and 眾生 stay untagged - 為 is a function character and 眾生 has no Sanskrit source here.)
NOW DO THIS ONE
SANSKRIT: {sanskrit_tagged}
LEMMAS: {lemmas_tagged}
CHINESE: {target}
CHINESE_SEGMENTED: {aid}
OUTPUT:"""

P_ZH_BO = """You are an expert Chinese-Tibetan philologist doing high-precision word alignment.
You are given a Chinese sentence with each word in a numbered tag <1>..</1> (CHINESE), and the Tibetan translation in Wylie (TIBETAN).
TASK: Reproduce the TIBETAN line EXACTLY and insert the numbered tags around the Tibetan word(s) translating each numbered Chinese word.
""" + STRICT_RULES + """
FORMAT RULES: 1) Copy the Tibetan verbatim; insert ONLY tags. 2) <n> wraps the Tibetan translating Chinese word n. 3) A span may cover several consecutive Tibetan words. 4) Tags will be out of numerical order. 5) Each Tibetan word in at most one tag; no overlap/nesting. 6) Omit numbers with no confident Tibetan equivalent; never emit <n></n>. 7) Balanced tags. 8) Output ONLY the tagged Tibetan line.
EXAMPLE
CHINESE: <1>佛</1> <2>說</2> <3>法</3>
TIBETAN: sangs rgyas kyis chos bshad do
OUTPUT: <1>sangs rgyas</1> kyis <3>chos</3> <2>bshad</2> do
NOW DO THIS ONE
CHINESE: {chinese_tagged}
TIBETAN: {target}
OUTPUT:"""

STRICT_PROMPTS = {("sanskrit", "tibetan"): P_SA_BO,
                  ("sanskrit", "chinese"): P_SA_ZH,
                  ("chinese", "tibetan"): P_ZH_BO}
PAIR_DIRECTIONS = ["sa-bo", "sa-zh", "zh-bo"]   # sa-bo, sa-zh, bo-zh pairs


_v1_build_unit = build_unit
def build_unit(d, cfg):
    if cfg["q_kind"] != "english":
        return _v1_build_unit(d, cfg)
    target = (d.get(f"{cfg['t_field']}_string") or "").strip()
    en = (d.get(f"{cfg['q_field']}_string") or "").strip()
    # junk filter: target must be real tibetan wylie, english must be real prose
    if not target or not en or len(target) < 15 or len(en.split()) < 3:
        return None
    letters = sum(c.isalpha() for c in target)
    if letters < 0.6 * len(target.replace(" ", "")):
        return None
    words = en.split()
    toks = [(w, w) for w in words]
    return {"toks": toks, "q": {"english_tagged": " ".join(
                f"<{i}>{w}</{i}>" for i, w in enumerate(words, 1))},
            "target": target, "aid": "", "id": d.get("id"),
            "src_segnr": d.get(f"{cfg['q_field']}_segnr"),
            "tgt_segnr": d.get(f"{cfg['t_field']}_segnr")}


P_EN_BO = """You are an expert translator-philologist doing high-precision English-Tibetan word alignment on 84000 translations.
You are given an English sentence with each word in a numbered tag <1>..</1> (ENGLISH), and the Tibetan source in Wylie (TIBETAN).
TASK: Reproduce the TIBETAN line EXACTLY and insert the numbered tags around the Tibetan word(s) corresponding to each numbered English word.
""" + STRICT_RULES + """
   - English function words (never tag, never use as anchor): {en_stop}
FORMAT RULES: 1) Copy the Tibetan verbatim; insert ONLY tags. 2) <n> wraps the Tibetan corresponding to English word n; a multi-word English concept should have its tags on the single head word only. 3) A span may cover several consecutive Tibetan words. 4) Tags will be out of numerical order. 5) Each Tibetan word in at most one tag; no overlap/nesting. 6) Omit numbers with no confident Tibetan equivalent; never emit <n></n>. 7) Balanced tags. 8) Output ONLY the tagged Tibetan line.
EXAMPLE
ENGLISH: <1>the</1> <2>Buddha</2> <3>taught</3> <4>the</4> <5>Dharma</5>
TIBETAN: sangs rgyas kyis chos bstan to
OUTPUT: <2>sangs rgyas</2> kyis <5>chos</5> <3>bstan</3> to
(Note: both instances of "the" stay untagged as function words.)
NOW DO THIS ONE
ENGLISH: {english_tagged}
TIBETAN: {target}
OUTPUT:"""

STRICT_PROMPTS[("english", "tibetan")] = P_EN_BO
PAIR_DIRECTIONS.append("en-bo")


def fmt_strict(u, cfg):
    p = STRICT_PROMPTS[(cfg["q_kind"], cfg["t_kind"])]
    return p.format(target=u["target"], aid=u["aid"], sa_stop=SA_STOP,
                    bo_stop=BO_STOP, zh_stop=ZH_STOP, en_stop=EN_STOP, **u["q"])


def instruction_row(u, cfg, tagged, direction):
    """lme DictionaryMining format: input = instruction block, target = tagged line."""
    p = STRICT_PROMPTS[(cfg["q_kind"], cfg["t_kind"])]
    # the finetune input is the full prompt minus the trailing OUTPUT cue value
    input_text = p.format(target=u["target"], aid=u["aid"], sa_stop=SA_STOP,
                          bo_stop=BO_STOP, zh_stop=ZH_STOP, en_stop=EN_STOP, **u["q"])
    return {"input_text": input_text, "target_text": tagged, "lang": direction}


def run_direction(direction, smoke=0):
    cfg = DIRS[direction]
    out_dir = OUT_BASE / direction
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    files = sorted(glob.glob(str(BASE / cfg["data"] / cfg["glob"])))
    id_set = None
    if cfg.get("pairids"):
        id_set = set(open(BASE / cfg["pairids"], encoding="utf-8").read().split("\n"))
        id_set.discard("")
    n_want = smoke if smoke else N_SAMPLE
    rows, total = reservoir(files, n_want, rng, id_set)
    rows = length_filter(rows, cfg["q_field"], cfg["t_field"])
    units = [u for u in (build_unit(d, cfg) for d in rows) if u]
    print(f"[{direction}] sampled {len(rows)} of {total}, {len(units)} usable units",
          file=sys.stderr)

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=API_KEY)
    gcfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET))

    lock = threading.Lock()
    done_keys = set()
    kept0 = 0
    rawp, seedp = out_dir / "gemini_raw.ndjson", out_dir / "seeds.ndjson"
    if rawp.exists():
        for l in open(rawp, encoding="utf-8"):
            try:
                d = json.loads(l)
                done_keys.add((str(d.get("src_segnr")), str(d.get("tgt_segnr")), d.get("target")))
            except Exception:
                pass
    if seedp.exists():
        kept0 = sum(1 for _ in open(seedp, encoding="utf-8"))
    units = [u for u in units
             if (str(u["src_segnr"]), str(u["tgt_segnr"]), u["target"]) not in done_keys]
    print(f"[{direction}] resume: {len(done_keys)} already done, {kept0} seeds kept, "
          f"{len(units)} units to run", file=sys.stderr)
    st = {"done": 0, "kept": kept0, "t0": time.time()}
    raw_fh = open(rawp, "a", encoding="utf-8")
    seed_fh = open(seedp, "a", encoding="utf-8")

    def work(u):
        prompt = fmt_strict(u, cfg)
        out, last_err = None, None
        for a in range(8):
            try:
                out = (client.models.generate_content(
                    model=MODEL_ID, contents=prompt, config=gcfg).text or "").strip()
                break
            except Exception as e:
                last_err = str(e)[:120]
                wait = 15 * (a + 1) if ('429' in last_err or 'RESOURCE_EXHAUSTED' in last_err) else 2 ** a
                if a == 7:
                    with lock:
                        st['errs'] = st.get('errs', 0) + 1
                        if st['errs'] % 25 == 1:
                            print(f"[err x{st['errs']}] {last_err}", file=sys.stderr)
                    return None
                time.sleep(wait)
        if out is None:
            return None
        grounded, faithful = parse_output(out, len(u["toks"]), u["target"])
        tagged = reconstruct(u["target"], grounded)
        return u, out, grounded, faithful, tagged

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, u) for u in units]
        for f in as_completed(futs):
            r = f.result()
            with lock:
                st["done"] += 1
                if st["done"] % 200 == 0:
                    el = time.time() - st["t0"]
                    print(f"[{direction}] {st['done']}/{len(units)} kept={st['kept']} "
                          f"({el:.0f}s)", file=sys.stderr)
                if r is None:
                    continue
                u, out, grounded, faithful, tagged = r
                raw_fh.write(json.dumps({
                    "id": u["id"], "src_segnr": u["src_segnr"], "tgt_segnr": u["tgt_segnr"],
                    "target": u["target"], "raw": out, "grounded": grounded,
                    "faithful": faithful, "n_toks": len(u["toks"]),
                    "n_tagged": len(grounded)}, ensure_ascii=False) + "\n")
                # keep: faithful reproduction, at least one grounded span,
                # and not tagging everything (strictness sanity: <=85% of tokens)
                if (faithful and grounded and st["kept"] < N_SEEDS
                        and len(grounded) <= 0.90 * len(u["toks"])):
                    seed_fh.write(json.dumps(
                        instruction_row(u, cfg, tagged, direction),
                        ensure_ascii=False) + "\n")
                    st["kept"] += 1
                if smoke:
                    print(f"\n--- {direction} example\nTARGET: {u['target'][:150]}\n"
                          f"RAW:    {out[:150]}\nTAGGED: {tagged[:150]}\n"
                          f"tagged {len(grounded)}/{len(u['toks'])} tokens, faithful={faithful}")
    raw_fh.close(); seed_fh.close()
    print(f"[{direction}] DONE: kept {st['kept']} seeds "
          f"-> {out_dir}/seeds.ndjson", file=sys.stderr)
    return st["kept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=PAIR_DIRECTIONS + ["en-bo"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    dirs = PAIR_DIRECTIONS if args.all else [args.direction]
    if not dirs or dirs == [None]:
        ap.error("need --direction or --all")
    for d in dirs:
        run_direction(d, smoke=args.smoke)


if __name__ == "__main__":
    main()
