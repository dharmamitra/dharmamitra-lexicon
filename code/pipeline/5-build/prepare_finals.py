#!/usr/bin/env python3
"""Stage 0 of the v2 dictionary build: turn the qwen mines (mined.ndjson.part*)
into per-direction hybrid/final.ndjson in the exact shape the proven v1 build
scripts expect.

- sa-zh: pass-through concat (Chinese target, no script conversion).
- sa-bo / zh-bo / en-bo: the v1 build stage is Wylie-internal (shad '/', '_'
  spaces, Wylie particle sets), so targets + align spans are converted
  Unicode -> Wylie with pyewts here, at the build boundary. The webapp evidence
  stage converts back and ships BOTH forms (dual wylie/unicode support).
  Every converted span is re-grounded in the converted target; failures dropped
  and counted (expected ~0: this unicode came from wylie originally).

Parallel: multiprocessing over rows (128-core box).
"""
import json, os, re, sys
from multiprocessing import Pool

SRC = "/mnt2/code/mitra-dictionaries/qwen-dictionaries-v4"
DIRS = ["sa-zh", "sa-bo", "zh-bo", "en-bo", "en-bo-dnx", "en-bo-lh", "en-bo-monlam"]
WORKERS = int(os.environ.get("WORKERS", "64"))

_conv = None
def _get_conv():
    global _conv
    if _conv is None:
        import pyewts
        _conv = pyewts.pyewts()
    return _conv


# ---- cleanup: substrate artifacts must not reach examples or headwords ----
FOLIO = re.compile(r"@[#0-9/ab]*\*?")           # wylie folio markers: @325ba, @#/, @##/, @/
WS = re.compile(r"[ \t]{2,}")
BO_EDGE = re.compile(r"^[\s_/]+|[\s_]+$")        # leading space/underscore/shad + trailing space/underscore
ZH_PAIRS = [("「", "」"), ("『", "』"), ("（", "）"), ("〔", "〕"), ("【", "】")]
ZH_LEAD = "」』）。，、；：？！·—…○ 　"        # strip at sentence start
ZH_TRAIL = "「『（，、；：·—…○ 　"             # strip at sentence end (keep 。！？ and balanced closers)


def clean_bo(s):
    s = FOLIO.sub(" ", s)
    s = WS.sub(" ", s)
    s = BO_EDGE.sub("", s)          # trailing shad // is legitimate — keep it
    return s.strip()


def _scrub_unbalanced(s):
    # drop unmatched bracket characters (stack per pair type)
    drop = set()
    for op, cl in ZH_PAIRS:
        stack = []
        for i, ch in enumerate(s):
            if ch == op:
                stack.append(i)
            elif ch == cl:
                if stack:
                    stack.pop()
                else:
                    drop.add(i)
        drop.update(stack)
    return "".join(ch for i, ch in enumerate(s) if i not in drop)


def clean_zh(s):
    s = _scrub_unbalanced(s)
    while s and s[0] in ZH_LEAD:
        s = s[1:]
    while s and s[-1] in ZH_TRAIL:
        s = s[:-1]
    return s.strip()


def clean_zh_span(sp):
    sp = _scrub_unbalanced(sp).strip()
    return sp.strip("」』）「『（。，、；：？！·—…○ 　")


# ---- integrity gate (2026-09-10) ---------------------------------------------
# mine_direction.py flags rows for: verse, conf < -0.45, ung_frac > 0.25,
# numbering_bad, coverage < 0.30, not verbatim. Nothing downstream ever read the
# flag, so broken outputs reached the dictionary. Low coverage alone is NOT a
# defect (short sentences legitimately align few words) and is still kept; every
# other reason means the model output itself is unreliable -> row dropped.
def integrity_bad(r):
    return bool(r.get("verse") or r.get("numbering_bad") or (not r.get("verbatim", True))
                or (r.get("conf", 0) is not None and r.get("conf", 0) < -0.45)
                or (r.get("ung_frac", 0) or 0) > 0.25)


# sa-zh keeps verse rows (decision 2026-10-07): verse makes up a fifth of the v2 sa-zh corpus and aligns almost as well
# as prose (agreement with Hirakawa on word-level spans, exact / relaxed: verse 30.3 / 81.2%, prose 34.2 / 84.6%).
def integrity_bad_keep_verse(r):
    # v4 (2026-10-16): gate OFF -- MITRA-parallel v3 pairs are post-corrected + verified, every mined row is used
    return False


def convert_row(line):
    """bo-target directions: unicode -> wylie + folio scrub + span re-grounding."""
    r = json.loads(line)
    if integrity_bad_keep_verse(r):   # sa-bo keeps verse as well (decision 2026-10-13)
        return None, -1
    conv = _get_conv()
    wy_target = clean_bo(conv.toWylie(r["target"]).strip())
    if not wy_target:
        return None, 0
    if r.get("src_original") and re.search(r"[一-鿿]", r["src_original"] or ""):
        r["src_original"] = clean_zh(r["src_original"])   # zh-bo: chinese source side
    aligns, dropped = {}, 0
    for n, span in r["aligns"].items():
        wsp = clean_bo(conv.toWylie(span).strip()).strip("/").strip()
        if wsp and wsp in wy_target:
            aligns[n] = wsp
        else:
            dropped += 1
    r["target"] = wy_target
    r["aligns"] = aligns
    r["source"] = "qwen-v2"
    return json.dumps(r, ensure_ascii=False), dropped


def clean_zh_row(line):
    """sa-zh: chinese-target boundary/bracket cleanup + span re-grounding."""
    r = json.loads(line)
    if integrity_bad_keep_verse(r):
        return None, -1
    tgt = clean_zh(r["target"])
    if not tgt:
        return None, 0
    aligns, dropped = {}, 0
    for n, span in r["aligns"].items():
        sp = clean_zh_span(span)
        if sp and sp in tgt:
            aligns[n] = sp
        else:
            dropped += 1
    r["target"] = tgt
    r["aligns"] = aligns
    r["source"] = "qwen-v2"
    return json.dumps(r, ensure_ascii=False), dropped


def build(direction):
    out_dir = os.path.join(SRC, direction, "hybrid")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "final.ndjson")
    parts = [os.path.join(SRC, direction, f"mined.ndjson.part{i}") for i in (0, 1)]
    fn = clean_zh_row if direction == "sa-zh" else convert_row   # sa-zh target = chinese
    n = dropped = gated = 0

    # sa-zh: pair-level lexical filter (lex50, lexfilter_sazh.py) — records listed in lex_drop_ids.txt are skipped
    drop_path = os.path.join(SRC, direction, "lex_drop_ids.txt")
    lexdrop = set(open(drop_path).read().split()) if os.path.exists(drop_path) else set()
    nlex = [0]

    def lines():
        for p in parts:
            for line in open(p, encoding="utf-8"):
                if lexdrop and json.loads(line)["id"] in lexdrop:
                    nlex[0] += 1
                    continue
                yield line

    with open(out_path, "w", encoding="utf-8") as out, Pool(WORKERS) as pool:
        for res, drp in pool.imap(fn, lines(), chunksize=500):
            if drp < 0:
                gated += 1
                continue
            dropped += drp
            if res is not None:
                out.write(res + "\n")
                n += 1
    if lexdrop:
        print(f"[{direction}] {nlex[0]} rows dropped by the lexical filter ({os.path.basename(drop_path)})")
    print(f"[{direction}] final.ndjson: {n} rows, {gated} rows dropped by the integrity gate, "
          f"{dropped} spans dropped at re-grounding")


if __name__ == "__main__":
    for d in (sys.argv[1:] or DIRS):
        build(d)
