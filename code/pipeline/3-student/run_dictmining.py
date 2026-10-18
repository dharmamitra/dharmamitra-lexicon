"""
Launch ONE dictionary-mining finetune experiment (one model, one GPU).

Pick the model + GPU via env so two copies can run in parallel:

  GPU0 (Qwen 0.8B):
    CUDA_VISIBLE_DEVICES=0 DICTMINING_EXPERIMENT=DictMiningQwen35Experiment \
    DICTMINING_BATCH=16 python run_dictmining.py

  GPU1 (Gemma 9.5B):
    CUDA_VISIBLE_DEVICES=1 DICTMINING_EXPERIMENT=DictMiningGemmaExperiment \
    DICTMINING_BATCH=2 python run_dictmining.py
"""
import os

# Single-GPU per process; keep NCCL P2P off (matches run_kokuyaku).
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lme.experiments.dictionary_mining as M


def main() -> None:
    exp_name = os.environ.get("DICTMINING_EXPERIMENT", "DictMiningQwen35Experiment")
    batch_size = int(os.environ.get("DICTMINING_BATCH", "16"))
    learning_rates = [float(x) for x in os.environ.get("DICTMINING_LR", "2e-5").split(",")]

    experiment = getattr(M, exp_name)()
    print(f"[run] {exp_name}  batch={batch_size}  lrs={learning_rates}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    experiment.run(batch_size, learning_rates)


if __name__ == "__main__":
    main()
