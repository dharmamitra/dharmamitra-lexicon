"""Numbers for tab:baselines / tab:baselines-refdict from the v2 baseline run (2026-10-13).
usage: baseline_numbers.py DIR...   (DIR = sa-bo / sa-zh; work dir from BASELINE_WORK)"""
import json, os, sys
W = os.environ.get("BASELINE_WORK", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/baselines/work-v2")
P = os.environ.get("BASELINE_PARTS", "/home/sebastian/papers/mitra-dict-2026/notes/eval-lfix-20261018/parts")
B = ["1", "2-4", "5-19", "20-99", "100-999", ">=1000"]
def pooled(m, bands):
    n = sum(m[b]["headwords"] for b in bands if m.get(b) and m[b].get("headwords"))
    return 100 * sum(m[b]["headwords"] * m[b]["P@1"] for b in bands if m.get(b) and m[b].get("headwords")) / n
for d in sys.argv[1:]:
    S = json.load(open(f"{W}/{d}.scores.json"))
    for name, m in S.items():
        print(f"{d} link {name:24} P {m['precision']:.1f} R {m['recall']:.1f} F1 {m['f1']:.1f}  P on tagged {m['precision_on_tagged_source']:.1f}")
    for t in ("fast_align", "mgiza", "eflomal", "llm"):
        st = json.load(open(f"{P}/{d}.{t}.json"))["variants"]["strict"]
        print(f"{d} refdict {t:10} n>=20 {pooled(st, B[3:]):.1f}  n<=4 {pooled(st, B[:2]):.1f}")
