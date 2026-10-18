#!/usr/bin/env python3
"""Assemble the lme SFT dataset from the four seed sets with a MINIMAL prompt.

The strict generation prompt (confidence policy, stopword lists) did its job at
mining time; the finetune learns the behavior from the outputs, so the training
input keeps only a one-line instruction + the data block.

Out: ~/data/qwen-dictionaries-spanmine/{train,val,test}.jsonl
     ({input_text, target_text, lang}; directions shuffled together)
"""
import json, os, random

SRC = "/mnt2/code/mitra-dictionaries/qwen-dictionaries"
OUT = os.path.expanduser("~/data/qwen-dictionaries-spanmine")
SEEDS = {
    "sa-bo": f"{SRC}/sa-bo/seeds_unicode.ndjson",
    "sa-zh": f"{SRC}/sa-zh/seeds.ndjson",
    "zh-bo": f"{SRC}/zh-bo/seeds_unicode.ndjson",
    "en-bo": f"{SRC}/en-bo/seeds_unicode.ndjson",
}
INSTRUCTION = ("Insert the numbered tags into the target line, marking the "
               "word(s) that translate each numbered source word. Copy the "
               "target verbatim and omit numbers without a clear equivalent.")
N_VAL, N_TEST = 50, 20   # per direction

os.makedirs(OUT, exist_ok=True)
rng = random.Random(42)
splits = {"train": [], "val": [], "test": []}
for lang, path in SEEDS.items():
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        # data block = everything after the mining prompt's cue, minus OUTPUT:
        block = r["input_text"].split("NOW DO THIS ONE\n", 1)[1]
        block = block.rsplit("OUTPUT:", 1)[0].strip()
        rows.append({"input_text": f"{INSTRUCTION}\n\n{block}",
                     "target_text": r["target_text"], "lang": lang})
    rng.shuffle(rows)
    splits["test"] += rows[:N_TEST]
    splits["val"] += rows[N_TEST:N_TEST + N_VAL]
    splits["train"] += rows[N_TEST + N_VAL:]
    print(f"{lang}: {len(rows)} total")

for name, rows in splits.items():
    rng.shuffle(rows)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")
print("->", OUT)
