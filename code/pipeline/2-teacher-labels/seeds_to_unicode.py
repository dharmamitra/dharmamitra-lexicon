#!/usr/bin/env python3
"""Post-hoc: rebuild sa-bo / zh-bo seeds with Tibetan in Unicode script.

Reads gemini_raw.ndjson (Wylie targets + grounded spans, as mined), converts
target and spans to Tibetan script with pyewts, re-grounds (a converted span
must still be a substring of the converted target; failures are dropped -
precision-safe), and emits seeds_unicode.ndjson with fully unicode prompts
(unicode example, unicode stopword list, no Wylie framing).

Units are rebuilt with the same deterministic sampling as generate_seeds, and
matched to raw rows by (src_segnr, tgt_segnr, wylie_target).
"""
import json, sys, re
from pathlib import Path

sys.path.insert(0, "/mnt2/code/mitra-dictionaries/qwen-dictionaries")
import generate_seeds as G
import pyewts

conv = pyewts.pyewts()
U = lambda w: conv.toUnicode(w)

BO_STOP_U = ", ".join(U(x.strip()) for x in G.BO_STOP.split(","))

P_SA_BO_U = G.P_SA_BO.replace("the Tibetan translation in Wylie (TIBETAN)",
                              "the Tibetan translation (TIBETAN)")
P_ZH_BO_U = G.P_ZH_BO.replace("the Tibetan translation in Wylie (TIBETAN)",
                              "the Tibetan translation (TIBETAN)")
# convert the few-shot example lines
for _w, _u in [
    ("'od srungs sems ni mi rnyed do", U("'od srungs sems ni mi rnyed do")),
    ("<2>'od srungs</2> <1>sems</1> ni <3>mi</3> <4>rnyed do</4>",
     f"<2>{U(chr(39)+'od srungs')}</2> <1>{U('sems')}</1> {U('ni')} <3>{U('mi')}</3> <4>{U('rnyed do')}</4>"),
    ("(Note: <3>mi</3> is kept", f"(Note: <3>{U('mi')}</3> is kept"),
    ("ni and do stay untagged", f"{U('ni')} and {U('do')} stay untagged"),
]:
    P_SA_BO_U = P_SA_BO_U.replace(_w, _u)
for _w, _u in [
    ("sangs rgyas kyis chos bshad do", U("sangs rgyas kyis chos bshad do")),
    ("<1>sangs rgyas</1> kyis <3>chos</3> <2>bshad</2> do",
     f"<1>{U('sangs rgyas')}</1> {U('kyis')} <3>{U('chos')}</3> <2>{U('bshad')}</2> {U('do')}"),
]:
    P_ZH_BO_U = P_ZH_BO_U.replace(_w, _u)

P_EN_BO_U = G.P_EN_BO.replace("the Tibetan source in Wylie (TIBETAN)",
                              "the Tibetan source (TIBETAN)")
for _w, _u in [
    ("sangs rgyas kyis chos bstan to", U("sangs rgyas kyis chos bstan to")),
    ("<2>sangs rgyas</2> kyis <5>chos</5> <3>bstan</3> to",
     f"<2>{U('sangs rgyas')}</2> {U('kyis')} <5>{U('chos')}</5> <3>{U('bstan')}</3> {U('to')}"),
]:
    P_EN_BO_U = P_EN_BO_U.replace(_w, _u)

PROMPT_U = {"sa-bo": P_SA_BO_U, "zh-bo": P_ZH_BO_U, "en-bo": P_EN_BO_U}


def run(direction):
    cfg = G.DIRS[direction]
    out_dir = G.OUT_BASE / direction
    import glob, random
    rng = random.Random(G.SEED)
    files = sorted(glob.glob(str(G.BASE / cfg["data"] / cfg["glob"])))
    id_set = None
    if cfg.get("pairids"):
        id_set = set(open(G.BASE / cfg["pairids"], encoding="utf-8").read().split("\n"))
        id_set.discard("")
    # union of all sampling sizes ever used for this direction, so raw rows
    # from every generation run find their unit (reservoir output depends on k)
    import os as _os
    sizes = [int(x) for x in _os.environ.get("N_SAMPLES", str(G.N_SAMPLE)).split(",")]
    units = {}
    for k in sizes:
        import random as _random
        rows, _ = G.reservoir(files, k, _random.Random(G.SEED), id_set)
        rows = G.length_filter(rows, cfg["q_field"], cfg["t_field"])
        for d in rows:
            u = G.build_unit(d, cfg)
            if u:
                units.setdefault((str(u["src_segnr"]), str(u["tgt_segnr"]), u["target"]), u)

    kept, dropped_reground = 0, 0
    with open(out_dir / "seeds_unicode.ndjson", "w", encoding="utf-8") as out:
        for line in open(out_dir / "gemini_raw.ndjson", encoding="utf-8"):
            if kept >= G.N_SEEDS:
                break
            r = json.loads(line)
            if not (r["faithful"] and r["grounded"]):
                continue
            if len(r["grounded"]) > 0.90 * r["n_toks"]:
                continue
            key = (str(r["src_segnr"]), str(r["tgt_segnr"]), r["target"])
            u = units.get(key)
            if u is None:
                continue
            t_uni = U(r["target"])
            spans_uni = {}
            ok = True
            for n, span in r["grounded"].items():
                su = U(span).strip("་")
                if su and su in t_uni:
                    spans_uni[int(n)] = su
                else:
                    ok = False
                    break
            if not ok or not spans_uni:
                dropped_reground += 1
                continue
            tagged = G.reconstruct(t_uni, spans_uni)
            p = PROMPT_U[direction]
            input_text = p.format(target=t_uni, aid=u["aid"],
                                  sa_stop=G.SA_STOP, bo_stop=BO_STOP_U,
                                  zh_stop=G.ZH_STOP, en_stop=G.EN_STOP, **u["q"])
            out.write(json.dumps({"input_text": input_text, "target_text": tagged,
                                  "lang": direction}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[{direction}] unicode seeds: {kept} kept, {dropped_reground} dropped at re-grounding")


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["sa-bo", "zh-bo", "en-bo"]):
        run(d)
