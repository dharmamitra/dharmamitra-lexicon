#!/usr/bin/env python3
"""Assemble the lme SFT dataset for the sa-en-capable span-projection model:
the four existing strict-mined directions (sa-bo, sa-zh, zh-bo, en-bo) plus the
new Mitrasamgraha sa-en seeds (gemini-3.8-flash HIGH thinking), with the same
MINIMAL prompt as build_sft_dataset.py.

sa-en is oversampled (SAEN_OVERSAMPLE, default 2) so the new direction is not
drowned by the ~7.8k existing rows in the single training epoch.

Out: ~/data/qwen-dictionaries-spanmine-saen/{train,val,test}.jsonl
     ({input_text, target_text, lang}; directions shuffled together)
"""
import json, os, random

SRC = "/mnt2/code/mitra-dictionaries/qwen-dictionaries"
OUT = os.path.expanduser("~/data/qwen-dictionaries-spanmine-saen")
SEEDS = {
    "sa-bo": f"{SRC}/sa-bo/seeds_unicode.ndjson",
    "sa-zh": f"{SRC}/sa-zh/seeds.ndjson",
    "zh-bo": f"{SRC}/zh-bo/seeds_unicode.ndjson",
    "en-bo": f"{SRC}/en-bo/seeds_unicode.ndjson",
    "sa-en": f"{SRC}/sa-en/seeds.ndjson",
}
INSTRUCTION = ("Insert the numbered tags into the target line, marking the "
               "word(s) that translate each numbered source word. Copy the "
               "target verbatim and omit numbers without a clear equivalent.")
N_VAL, N_TEST = 50, 20   # per direction
OVERSAMPLE = {"sa-en": int(os.environ.get("SAEN_OVERSAMPLE", "2"))}

os.makedirs(OUT, exist_ok=True)
rng = random.Random(42)
splits = {"train": [], "val": [], "test": []}
for lang, path in SEEDS.items():
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        block = r["input_text"].split("NOW DO THIS ONE\n", 1)[1]
        block = block.rsplit("OUTPUT:", 1)[0].strip()
        rows.append({"input_text": f"{INSTRUCTION}\n\n{block}",
                     "target_text": r["target_text"], "lang": lang})
    rng.shuffle(rows)
    splits["test"] += rows[:N_TEST]
    splits["val"] += rows[N_TEST:N_TEST + N_VAL]
    train = rows[N_TEST + N_VAL:]
    splits["train"] += train * OVERSAMPLE.get(lang, 1)
    print(f"{lang}: {len(rows)} total, {len(train)} train x{OVERSAMPLE.get(lang, 1)}")

for name, rows in splits.items():
    rng.shuffle(rows)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")
print("->", OUT)
