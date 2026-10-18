#!/usr/bin/env python3
"""Wrap a finetuned Qwen3.5 TEXT checkpoint (Qwen3_5ForCausalLM, as saved by the
lme SFT) into the multimodal Qwen3_5ForConditionalGeneration layout that vLLM
0.19 requires (it registers only the multimodal arch). Mirrors how
/mnt/data/qwen35-9b-dictspan-mm was built: the language weights come from the
finetune, the (unused) vision tower + preprocessor configs are taken from the
TEMPLATE checkpoint.

  python convert_to_mm.py --src ~/code/lme/results/<exp>/2e-05/final --out /mnt/data/qwen35-9b-dictspan-saen-mm
"""
import argparse, json, os, shutil

TEMPLATE = "/mnt/data/qwen35-9b-dictspan-mm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=TEMPLATE)
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(args.out, exist_ok=True)
    tensors = {}
    with safe_open(os.path.join(args.src, "model.safetensors"), "pt") as f:
        src_keys = set(f.keys())
        for k in src_keys:
            tensors[k] = f.get_tensor(k)
    n_lang = len(tensors)
    with safe_open(os.path.join(args.template, "model.safetensors"), "pt") as f:
        tpl_keys = set(f.keys())
        for k in tpl_keys:
            if k not in src_keys:
                assert "visual" in k, f"unexpected non-visual template-only key {k}"
                tensors[k] = f.get_tensor(k)
            else:
                assert f.get_slice(k).get_shape() == list(tensors[k].shape), f"shape mismatch {k}"
    missing = [k for k in src_keys if k not in tpl_keys]
    assert not missing, f"src keys absent from template: {missing[:5]}"
    print(f"language tensors: {n_lang}, visual tensors: {len(tensors) - n_lang}")
    save_file(tensors, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    # configs: template's mm config with the text_config refreshed from the src config
    cfg = json.load(open(os.path.join(args.template, "config.json")))
    src_cfg = json.load(open(os.path.join(args.src, "config.json")))
    for k, v in src_cfg.items():
        if k in cfg["text_config"] and k not in ("architectures", "model_type"):
            cfg["text_config"][k] = v
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy(os.path.join(args.template, fn), args.out)
    for fn in ("generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        p = os.path.join(args.src, fn)
        if os.path.exists(p):
            shutil.copy(p, args.out)
    print("->", args.out)


if __name__ == "__main__":
    main()
