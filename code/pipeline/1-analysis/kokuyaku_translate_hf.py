#!/usr/bin/env python3
"""Translate each Chinese `par_string` -> Japanese with the kokuyaku-finetuned
Qwen3.5-0.8B model (best_chrf), via HF transformers batched generation.

vLLM 0.19 has no text-only Qwen3_5ForCausalLM (only the multimodal variant), so
this uses transformers directly -- the same stack the model was trained/eval'd
with. One process per GPU. Prompt format matches training:
  user: "次の漢文（古典中国語）を日本語に訓読・翻訳してください。\n\n{src}"
  apply_chat_template(add_generation_prompt=True, enable_thinking=False)

`par_japanese` is inserted right after `par_string`. Resumable: records already
carrying `par_japanese` are skipped. Records are length-sorted within each file
to minimize padding waste.
"""
import argparse, json, os, time

MODEL_DEFAULT = "/home/sebastian/code/lme/results/TranslationKokuyakuQwen35Experiment/2e-05/best_chrf"
INSTRUCTION = "次の漢文（古典中国語）を日本語に訓読・翻訳してください。\n\n{src}"


def insert_after(d, anchor, key, value):
    out = {}
    for k, v in d.items():
        out[k] = v
        if k == anchor:
            out[key] = value
    if anchor not in d:
        out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    tag = lambda m: print(f"[gpu {gpu}] {m}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    tag(f"loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tag("model ready")

    def build_prompt(src):
        messages = [{"role": "user", "content": INSTRUCTION.format(src=src)}]
        try:
            return tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)

    for fpath in args.files:
        records = []
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        need_idx = [i for i, r in enumerate(records)
                    if (r.get("par_string") or "").strip() and not r.get("par_japanese")]
        if not need_idx:
            tag(f"{os.path.basename(fpath)}: nothing to do ({len(records)} recs)")
            continue
        tag(f"{os.path.basename(fpath)}: {len(need_idx)}/{len(records)} to translate")

        # build prompts + token lengths, then sort by length to cut padding
        prompts = {i: build_prompt(records[i]["par_string"]) for i in need_idx}
        lens = {i: len(tok(prompts[i], add_special_tokens=False)["input_ids"]) for i in need_idx}
        order = sorted(need_idx, key=lambda i: lens[i])

        f_start = time.time()
        done = 0
        for b in range(0, len(order), args.batch_size):
            idxs = order[b:b + args.batch_size]
            batch_prompts = [prompts[i] for i in idxs]
            enc = tok(batch_prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_input_tokens,
                      add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for i, t in zip(idxs, texts):
                records[i]["par_japanese"] = t.strip()
            done += len(idxs)
            el = time.time() - f_start
            tag(f"  {os.path.basename(fpath)}: {done}/{len(order)} ({done/el:.1f} sent/s)")

        out_records = [insert_after(r, "par_string", "par_japanese", r.pop("par_japanese"))
                       if "par_japanese" in r else r for r in records]
        tmp = fpath + ".tmp"
        with open(tmp, "w") as fh:
            for r in out_records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, fpath)
        tag(f"{os.path.basename(fpath)}: wrote {len(out_records)} records "
            f"({len(need_idx)} translated) in {time.time()-f_start:.0f}s")

    tag("DONE")


if __name__ == "__main__":
    main()
