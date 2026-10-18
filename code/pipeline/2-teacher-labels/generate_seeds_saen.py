#!/usr/bin/env python3
"""Strict-prompt instruction-seed generation for the sa-en (Sanskrit -> English)
direction of the qwen dictionary span-projection finetune, on the Mitrasamgraha
Sanskrit-English sentence pairs.

Same design as generate_seeds.py (sa-bo / sa-zh / zh-bo / en-bo), with:
  - gemini-3.8-flash, thinking_level HIGH
  - Sanskrit side from OUR grammatical analyzer (enrich_saen.py -> full_analysis:
    unsandhied word separation + lemmas), presented as SANSKRIT / LEMMAS lines
  - English target: tags inserted around the English word(s) translating each
    numbered Sanskrit word; strict confidence policy + Sanskrit/English stopword
    no-latch lists
  - gates: verbatim reproduction, span grounding, >=2 grounded spans, <=90% of
    tokens tagged; stops at N_SEEDS kept.

  python generate_seeds_saen.py --smoke 3
  python generate_seeds_saen.py            # -> sa-en/seeds.ndjson (1,000)
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

V1 = "/mnt2/code/mitra-dictionaries/mitra-dictionaries-2026"
sys.path.insert(0, V1)
from mine_spans import API_KEY as _V1_KEY, TAG, parse_output, reconstruct
from generate_seeds import SA_STOP, EN_STOP

def _env_key():
    """GEMINI_API_KEY from the environment, else ~/code/mitra-evaluation/.env
    (the v1 hardcoded key's prepaid credits are depleted as of 2026-09-06)."""
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    envp = os.path.expanduser("~/code/mitra-evaluation/.env")
    if os.path.exists(envp):
        for line in open(envp):
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return _V1_KEY

API_KEY = _env_key()
OUT_DIR = Path("/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-en")
CANDIDATES = OUT_DIR / "seed_candidates.ndjson"
MODEL_ID = "gemini-3.8-flash"
THINKING_LEVEL = "HIGH"
SEED = 42
N_SEEDS = int(os.environ.get("N_SEEDS", "1000"))
WAVE = 400
WORKERS = int(os.environ.get("GEMINI_WORKERS", "40"))
DIRECTION = "sa-en"

STRICT_RULES = """CONFIDENCE POLICY (the most important rules):
A) Tag a number ONLY if you are highly confident the span is the actual translation of that word. If you are uncertain, OMIT the number entirely. It is EXPECTED and CORRECT that many numbers remain untagged - precision matters far more than coverage here.
B) NEVER tag a span that consists only of function words, and NEVER attach function words to a content span unless they are genuinely part of the translation (e.g. a phrasal verb or a fixed expression):
   - Sanskrit function words (never treat as content, never use as anchor): {sa_stop}
   - English function words (never tag alone, never latch onto): {en_stop}
C) If the English is a loose or literary paraphrase rather than a close translation, tag only the words whose correspondence is unmistakable (names, technical terms, numerals, clear content words) and omit the rest.
D) Sanskrit compound members are numbered separately (a trailing hyphen marks a prior member). Tag each member's own English equivalent only if it is clearly separable in the English; if the compound is rendered as one English unit, tag the whole English unit ONCE under the head (last) member and omit the other members."""

P_SA_EN = """You are an expert Sanskrit-English philologist doing high-precision word alignment.
You are given a Sanskrit sentence with each word in a numbered tag <1>..</1> (SANSKRIT, sandhi resolved and words separated by a grammatical analyzer), the same words as dictionary lemmas with the SAME numbers (LEMMAS), and the published English translation (ENGLISH).
TASK: Reproduce the ENGLISH line EXACTLY and insert the numbered tags around the English word(s) translating each numbered Sanskrit word.
""" + STRICT_RULES + """
FORMAT RULES: 1) Copy the English verbatim - never translate/reorder/add/delete it, keep its punctuation and capitalization; insert ONLY tags. 2) <n> wraps the English translating Sanskrit word n. 3) A span may cover several consecutive English words (e.g. a verb with its particle, an adjective-noun rendering of one Sanskrit word). 4) Tags will be out of numerical order; never move English to make them sequential. 5) Each English word in at most one tag; no overlap/nesting. 6) Punctuation stays outside the tags. 7) Omit numbers with no confident English equivalent; never emit <n></n>. 8) Balanced tags. 9) Output ONLY the tagged English line.
EXAMPLE
SANSKRIT: <1>rājā</1> <2>tu</2> <3>tam</3> <4>brāhmaṇam</4> <5>dṛṣṭvā</5> <6>prahṛṣṭaḥ</6> <7>abhavat</7>
LEMMAS: <1>rājan</1> <2>tu</2> <3>tad</3> <4>brāhmaṇa</4> <5>dṛś</5> <6>prahṛṣ</6> <7>bhū</7>
ENGLISH: But the king, seeing that Brahmin, was filled with joy.
OUTPUT: But the <1>king</1>, <5>seeing</5> that <4>Brahmin</4>, <6>was filled with joy</6>.
(Note: tu, tam and abhavat stay untagged - "But" and "that" are function words, and the copula is absorbed into the periphrastic "was filled with joy", which is tagged once under prahṛṣṭaḥ.)
NOW DO THIS ONE
SANSKRIT: {sanskrit_tagged}
LEMMAS: {lemmas_tagged}
ENGLISH: {target}
OUTPUT:"""


def build_unit(d):
    fa = d.get("full_analysis") or []
    toks = [(t.get("unsandhied", ""), t.get("lemma", "")) for t in fa if t.get("unsandhied")]
    target = " ".join((d.get("par_string") or "").split())
    if len(toks) < 3 or len(target.split()) < 4:
        return None
    if sum(c.isalpha() for c in target) < 0.6 * len(target.replace(" ", "")):
        return None
    return {"toks": toks,
            "q": {"sanskrit_tagged": " ".join(f"<{i}>{s}</{i}>" for i, (s, l) in enumerate(toks, 1)),
                  "lemmas_tagged": " ".join(f"<{i}>{l}</{i}>" for i, (s, l) in enumerate(toks, 1))},
            "target": target, "id": d["id"], "file": d["file"], "license": d.get("license", ""),
            "root_string": d["root_string"]}


def fmt(u):
    return P_SA_EN.format(target=u["target"], sa_stop=SA_STOP, en_stop=EN_STOP, **u["q"])


def positional_tagged(out, target, grounded):
    """Use the model's own tag positions when the detagged output equals the
    target modulo whitespace (safer than first-occurrence placement for English,
    where words repeat). Returns None if anything is off -> caller falls back."""
    s = re.sub(r"\s+", " ", out).strip()
    s = re.sub(r"<(\d+)>\s+", r"<\1>", s)
    s = re.sub(r"\s+</(\d+)>", r"</\1>", s)
    if re.sub(r"</?\d+>", "", s) != target:
        return None
    seen = {}
    for m in TAG.finditer(s):
        n, span = int(m.group(1)), m.group(2)
        if n in seen or n not in grounded or grounded[n] != span:
            return None
        seen[n] = span
    if set(seen) != set(grounded) or re.search(r"<\d+>[^<]*<\d+>", s):
        return None
    return s


def instruction_row(u, tagged):
    return {"input_text": fmt(u), "target_text": tagged, "lang": DIRECTION,
            "id": u["id"], "file": u["file"], "license": u["license"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(SEED)
    cands = [json.loads(l) for l in open(CANDIDATES, encoding="utf-8")]
    units = [u for u in (build_unit(d) for d in cands) if u]
    rng.shuffle(units)
    print(f"[{DIRECTION}] {len(units)} usable units of {len(cands)} candidates", file=sys.stderr)

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=API_KEY)
    gcfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL))

    rawp, seedp = OUT_DIR / "gemini_raw.ndjson", OUT_DIR / "seeds.ndjson"
    done_ids, kept0 = set(), 0
    if not args.smoke:
        if rawp.exists():
            for l in open(rawp, encoding="utf-8"):
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass
        if seedp.exists():
            kept0 = sum(1 for _ in open(seedp, encoding="utf-8"))
        units = [u for u in units if u["id"] not in done_ids]
        print(f"[{DIRECTION}] resume: {len(done_ids)} done, {kept0} seeds kept", file=sys.stderr)
    if args.smoke:
        units = units[:args.smoke]
    lock = threading.Lock()
    st = {"done": 0, "kept": kept0, "errs": 0, "faithful": 0, "t0": time.time()}
    raw_fh = open(rawp, "a", encoding="utf-8") if not args.smoke else None
    seed_fh = open(seedp, "a", encoding="utf-8") if not args.smoke else None

    def work(u):
        prompt = fmt(u)
        out, last_err = None, None
        for a in range(8):
            try:
                out = (client.models.generate_content(
                    model=MODEL_ID, contents=prompt, config=gcfg).text or "").strip()
                break
            except Exception as e:
                last_err = str(e)[:160]
                wait = 15 * (a + 1) if ("429" in last_err or "RESOURCE_EXHAUSTED" in last_err) else 2 ** a
                if a == 7:
                    with lock:
                        st["errs"] += 1
                        if st["errs"] % 10 == 1:
                            print(f"[err x{st['errs']}] {last_err}", file=sys.stderr)
                    return None
                time.sleep(wait)
        if out is None:
            return None
        out = out.splitlines()[-1].strip() if "\n" in out.strip() else out   # tolerate a stray preamble
        grounded, faithful = parse_output(out, len(u["toks"]), u["target"])
        tagged = positional_tagged(out, u["target"], grounded) or reconstruct(u["target"], grounded)
        return u, out, grounded, faithful, tagged

    def keep(u, grounded, faithful):
        return (faithful and len(grounded) >= 2
                and len(grounded) <= 0.90 * len(u["toks"]))

    pos = 0
    while pos < len(units) and (args.smoke or st["kept"] < N_SEEDS):
        wave = units[pos:pos + WAVE]; pos += len(wave)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for f in as_completed([ex.submit(work, u) for u in wave]):
                r = f.result()
                with lock:
                    st["done"] += 1
                    if r is None:
                        continue
                    u, out, grounded, faithful, tagged = r
                    st["faithful"] += int(faithful)
                    if args.smoke:
                        print(f"\n--- {u['id']}\nSANSKRIT: {u['root_string'][:200]}\nTARGET: {u['target'][:200]}\n"
                              f"RAW:    {out[:300]}\nTAGGED: {tagged[:300]}\n"
                              f"tagged {len(grounded)}/{len(u['toks'])} tokens, faithful={faithful}, keep={keep(u, grounded, faithful)}")
                        for n in sorted(grounded):
                            print(f"   {n:2} {u['toks'][n-1][0]:18} {u['toks'][n-1][1]:14} -> {grounded[n]}")
                        continue
                    raw_fh.write(json.dumps({
                        "id": u["id"], "file": u["file"], "target": u["target"], "raw": out,
                        "grounded": grounded, "faithful": faithful, "n_toks": len(u["toks"]),
                        "n_tagged": len(grounded)}, ensure_ascii=False) + "\n")
                    if keep(u, grounded, faithful) and st["kept"] < N_SEEDS:
                        seed_fh.write(json.dumps(instruction_row(u, tagged), ensure_ascii=False) + "\n")
                        st["kept"] += 1
                    if st["done"] % 50 == 0:
                        raw_fh.flush(); seed_fh.flush()
                        print(f"[{DIRECTION}] {st['done']} calls kept={st['kept']} faithful={st['faithful']} "
                              f"errs={st['errs']} ({time.time()-st['t0']:.0f}s)", file=sys.stderr)
    if raw_fh:
        raw_fh.close(); seed_fh.close()
    print(f"[{DIRECTION}] DONE calls={st['done']} kept={st['kept']} faithful={st['faithful']} "
          f"errs={st['errs']} -> {seedp}", file=sys.stderr)


if __name__ == "__main__":
    main()
