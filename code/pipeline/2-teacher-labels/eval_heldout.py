#!/usr/bin/env python3
"""Student-teacher agreement on held-out teacher labels.

Runs a span-projection checkpoint (vLLM, greedy, exactly the training prompt
format: input_text + "\nOUTPUT:\n") over a jsonl of {input_text, target_text, lang}
rows, stores the predictions, and scores span-level precision / recall / F1 per
direction with bootstrap confidence intervals.

A predicted span counts as correct if the same tag number wraps the same span
text as in the teacher label (the metric of eval_spans.py, i.e. of the 86.9 figure).
A relaxed variant additionally credits partial character overlap (Jaccard >= 0.5)
of the two spans for the same number.

  python eval_heldout.py --ckpt /mnt/data/qwen35-9b-dictspan-mm \
      --data ~/data/qwen-dictionaries-heldout500/test.jsonl --tag four-dir
  python eval_heldout.py --score-only preds.jsonl      # re-score without GPU
"""
import argparse, collections, json, os, random, re, sys

TAG = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)
norm = lambda s: " ".join((s or "").split())


def parse_spans(s):
    d = {}
    for m in TAG.finditer(s):
        n = m.group(1)
        if n not in d:
            d[n] = norm(m.group(2))
    return d


def jaccard(a, b):
    """character-multiset overlap (whitespace ignored), symmetric."""
    ca, cb = collections.Counter(a.replace(" ", "")), collections.Counter(b.replace(" ", ""))
    inter = sum((ca & cb).values()); union = sum((ca | cb).values())
    return inter / union if union else 0.0


def counts(pred, gold, relaxed=False):
    gp, pp = parse_spans(gold), parse_spans(pred)
    cor = 0
    for n, span in gp.items():
        p = pp.get(n)
        if p is None:
            continue
        if p == span or (relaxed and jaccard(p, span) >= 0.5):
            cor += 1
    return cor, len(gp), len(pp)


def prf(c, g, p):
    r = c / g if g else 0.0
    pr = c / p if p else 0.0
    f = 2 * r * pr / (r + pr) if r + pr else 0.0
    return 100 * pr, 100 * r, 100 * f


def score(rows, relaxed=False, boot=1000, seed=0):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["lang"]].append(counts(r["pred"], r["target_text"], relaxed))
    out = {}
    rng = random.Random(seed)
    for lang, cs in sorted(by.items()):
        tot = [sum(x[i] for x in cs) for i in range(3)]
        p, r, f = prf(*tot)
        bs = []
        for _ in range(boot):
            samp = [cs[rng.randrange(len(cs))] for _ in cs]
            bs.append(prf(*[sum(x[i] for x in samp) for i in range(3)]))
        lo = [sorted(b[i] for b in bs)[int(0.025 * boot)] for i in range(3)]
        hi = [sorted(b[i] for b in bs)[int(0.975 * boot) - 1] for i in range(3)]
        out[lang] = dict(n=len(cs), gold=tot[1], pred=tot[2], correct=tot[0],
                         precision=p, recall=r, f1=f,
                         ci=dict(precision=(lo[0], hi[0]), recall=(lo[1], hi[1]), f1=(lo[2], hi[2])))
    return out


def wellformed(rows):
    """verbatim reproduction and ungrounded-span rate of the student output."""
    by = collections.defaultdict(lambda: [0, 0, 0, 0])   # n, verbatim, spans, ungrounded
    for r in rows:
        target = r["input_text"].rsplit("\n", 1)[-1]
        # target line = last data line of the block for sa-bo/sa-en; for sa-zh it is the
        # CHINESE: line (CHINESE_SEGMENTED follows). Recover from the gold instead:
        target = norm(re.sub(r"</?\d+>", "", r["target_text"]))
        detag = norm(re.sub(r"</?\d+>", "", r["pred"]))
        b = by[r["lang"]]
        b[0] += 1; b[1] += int(detag == target)
        for m in TAG.finditer(r["pred"]):
            b[2] += 1; b[3] += int(norm(m.group(2)) not in target)
    return {l: dict(n=b[0], verbatim=100 * b[1] / b[0], spans=b[2],
                    ungrounded=100 * b[3] / max(b[2], 1)) for l, b in by.items()}


def report(rows, tag):
    print(f"\n=== {tag}: student-teacher agreement (n={len(rows)}) ===")
    print(f"{'dir':7} {'n':>4} {'gold':>6} {'pred':>6} {'P':>6} {'R':>6} {'F1':>6}   95% CI F1        | relaxed F1")
    strict, relaxed = score(rows), score(rows, relaxed=True)
    for lang in strict:
        s, x = strict[lang], relaxed[lang]
        print(f"{lang:7} {s['n']:4d} {s['gold']:6d} {s['pred']:6d} {s['precision']:6.1f} {s['recall']:6.1f} "
              f"{s['f1']:6.1f}   [{s['ci']['f1'][0]:5.1f}, {s['ci']['f1'][1]:5.1f}]   | {x['f1']:6.1f}")
    wf = wellformed(rows)
    print("well-formedness:", {l: f"verbatim {v['verbatim']:.1f}%, ungrounded {v['ungrounded']:.2f}%"
                               for l, v in wf.items()})
    return dict(tag=tag, strict=strict, relaxed=relaxed, wellformed=wf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="predictions jsonl (default: next to data)")
    ap.add_argument("--score-only", default=None, help="existing predictions jsonl")
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()

    if a.score_only:
        rows = [json.loads(l) for l in open(a.score_only, encoding="utf-8")]
        res = report(rows, a.tag or os.path.basename(a.score_only))
    else:
        rows = [json.loads(l) for l in open(os.path.expanduser(a.data), encoding="utf-8")]
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048)
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop=["\n"])
        outs = llm.generate([r["input_text"] + "\nOUTPUT:\n" for r in rows], sp, use_tqdm=True)
        for r, o in zip(rows, outs):
            r["pred"] = o.outputs[0].text.strip()
            r["ckpt"] = a.ckpt
        out = a.out or os.path.expanduser(a.data).replace(".jsonl", f".preds-{a.tag or 'ckpt'}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("predictions ->", out)
        res = report(rows, a.tag or a.ckpt)
        with open(out.replace(".jsonl", ".scores.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
