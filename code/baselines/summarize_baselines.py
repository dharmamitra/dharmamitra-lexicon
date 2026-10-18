#!/usr/bin/env python3
"""Collect the baseline results into one summary (markdown + LaTeX rows).

  link-level agreement with the held-out teacher labels: work/<dir>.scores.json
  reference-lexicon P@1 by band:  notes/eval/parts/<dir>.json (MITRA-dict as published),
                                  notes/eval/parts/<dir>.<tool>.json (raw token dictionaries)
"""
import json, os, sys
from pathlib import Path

WORK = Path("/mnt2/code/mitra-dictionaries/qwen-dictionaries/baselines/work")
PARTS = Path("/home/sebastian/papers/mitra-dict-2026/notes/eval/parts")
DIRS = ["sa-bo", "sa-zh", "sa-en"]
VARIANT = {"sa-bo": "strict", "sa-zh": "strict", "sa-en": "mw_apte"}
BANDS = ["1", "2-4", "5-19", "20-99", "100-999", ">=1000"]
TOOLS = ["fast_align", "mgiza", "eflomal", "llm"]
NAME = {"fast_align": "fast_align", "mgiza": "GIZA++ (mgiza, IBM4)", "eflomal": "eflomal",
        "llm": "student links, raw token dictionary", "student (four-dir)": "student (4-dir ckpt)",
        "student (five-dir)": "student (5-dir ckpt)"}


def pooled(m, bands):
    n = sum(m[b]["headwords"] for b in bands if b in m and m[b].get("headwords"))
    c = sum(m[b]["headwords"] * m[b]["P@1"] for b in bands if b in m and m[b].get("headwords"))
    return 100 * c / n if n else float("nan"), n


def main():
    L = ["# Statistical baselines vs. the LLM aligner", ""]
    L += ["## 1. Link-level agreement with held-out teacher labels", "",
          "Links = (Sanskrit word, target token) pairs; target tokens: Tibetan syllables, Chinese characters, English words; "
          "punctuation excluded. 'on tagged' = restricted to source words the teacher tagged (the teacher leaves function "
          "words untagged by design, so the baselines' overall precision is bounded by that policy).", ""]
    L += ["| direction | system | P | R | F1 | 95% CI F1 | P on tagged | F1 on tagged |", "|---|---|---:|---:|---:|---|---:|---:|"]
    tex_link = {}
    for d in DIRS:
        p = WORK / f"{d}.scores.json"
        if not p.exists():
            continue
        S = json.load(open(p))
        for name, m in S.items():
            L.append(f"| {d} | {NAME.get(name, name)} | {m['precision']:.1f} | {m['recall']:.1f} | {m['f1']:.1f} | "
                     f"[{m['f1_ci'][0]:.1f}, {m['f1_ci'][1]:.1f}] | {m['precision_on_tagged_source']:.1f} | {m['f1_on_tagged_source']:.1f} |")
            tex_link[(d, name)] = m
    L += ["", "## 2. Reference-lexicon agreement (P@1, strict; sa-en: MW72 ∪ Apte) by headword frequency band", "",
          "Baseline dictionaries: lemma -> longest contiguous run of linked target tokens, counted over the full corpus, no "
          "filtering, single-word (token) entries only. 'MITRA-dict' = the released dictionary (phrase entries, integrity gate, "
          "noise filters) as in the paper; 'student links, raw' = the student's token links through the same raw builder as the "
          "baselines (apples to apples).", ""]
    L += ["| direction | dictionary | " + " | ".join(BANDS) + " | n>=20 pooled | \\|E\\| |", "|---|---|" + "---:|" * (len(BANDS) + 2)]
    tex_dict = {}
    for d in DIRS:
        rows = [("MITRA-dict (released)", PARTS / f"{d}.json")] + [(NAME[t], PARTS / f"{d}.{t}.json") for t in TOOLS]
        for label, p in rows:
            if not p.exists():
                continue
            m = json.load(open(p))["variants"][VARIANT[d]]
            cells = [f"{100*m[b]['P@1']:.1f} ({m[b]['headwords']})" if m.get(b, {}).get("headwords") else "–" for b in BANDS]
            p20, n20 = pooled(m, ["20-99", "100-999", ">=1000"])
            L.append(f"| {d} | {label} | " + " | ".join(cells) + f" | {p20:.1f} ({n20}) | {m['all']['headwords']} |")
            tex_dict[(d, label)] = (m, p20)
    out = WORK / "summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
