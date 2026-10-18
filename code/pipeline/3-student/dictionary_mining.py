"""
Dictionary-term mining SFT: given a source passage + segmentation, an aligned
target (Tibetan, or Sanskrit with segmentation/lemmas) and one query TERM,
extract the target word/phrase that translates TERM.

Two models trained in parallel on separate GPUs (see run_dictmining.py):
  - DictMiningQwen35Experiment  (Qwen/Qwen3.5-0.8B, fused Adam)      -> GPU 0
  - DictMiningGemmaExperiment   (gemma2-mitra-base 9.5B, 8-bit Adam) -> GPU 1

Both are BASE models; gemma2-mitra-base has no chat template, so we use a single
plain-text prompt format that works for either model.
"""
import os

from transformers import PreTrainedTokenizerBase

from lme.data_processors import DictionaryMiningProcessor, DictionaryMiningSaboProcessor, SpanProjectionProcessor, SpanProjectionEnBoProcessor, AlignmentDenoisingProcessor, MitraKnnSftProcessor, QwenDictSpanProcessor, QwenDictSaenSpanProcessor, TibSegTerseProcessor, ZhSegTerseProcessor, JointSegTerseProcessor, SegPostcorrectionProcessor, SegPostcorrectionV2Processor, SegPostcorrectionV3Processor
from lme.model_mixins import Qwen35Mixin, GemmaMitraMixin, Gemma4E4BMixin, Gemma4E2BMixin, Qwen35Stage29BMixin
from lme.training_argument_mixins import DictMiningFinetuneArgsMixin
from lme.training_pipelines import FinetuneDictMiningSFTExperimentBase


class DictionaryMining:
    """Task mixin: data source + plain-text prompt + eval caps."""
    MAX_INPUT_LENGTH = 512
    DATA_PROCESSOR_CLS = DictionaryMiningProcessor
    EVAL_GEN_CAP = 500
    EVAL_MAX_NEW_TOKENS = 48

    INSTRUCTION = (
        "You are a Buddhist bilingual lexicographer. You are given a source passage (SRC) "
        "with its word segmentation (SRC_SEGMENTED), its aligned translation (TGT) -- "
        "Tibetan, or Sanskrit also given as TGT_SEGMENTED / TGT_LEMMAS -- and a single query "
        "TERM taken from the source. Identify the exact word or phrase in the target that "
        "translates TERM. For a Tibetan target, output that Tibetan span. For a Sanskrit "
        "target, output '<lemma> | <surface form>'. If the target contains no translation of "
        "TERM, output NONE."
    )

    def build_prompt(self, tokenizer: PreTrainedTokenizerBase, source: str) -> str:
        # Plain text (no chat template): identical for train and eval, so the
        # prompt/completion boundary is consistent for both base models. The
        # trailing newline keeps the boundary on a clean token (a bare "ANSWER:"
        # colon merges with a following ASCII letter, breaking completion-only
        # loss masking).
        return f"{self.INSTRUCTION}\n\n{source}\n\nANSWER:\n"


class DictMiningQwen35Experiment(
    Qwen35Mixin, DictMiningFinetuneArgsMixin, DictionaryMining, FinetuneDictMiningSFTExperimentBase
):
    OPTIM = "adamw_torch_fused"
    TARGET_BATCH = 256
    EVAL_BATCH = 32


class DictMiningGemmaExperiment(
    GemmaMitraMixin, DictMiningFinetuneArgsMixin, DictionaryMining, FinetuneDictMiningSFTExperimentBase
):
    # Full finetune of 9.5B: 8-bit Adam keeps optimizer states small enough for
    # one GPU. Smaller effective batch so optimizer steps (hence time-based
    # benchmark points) come often enough on the larger model.
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 64
    EVAL_BATCH = 16


class DictMiningGemmaSaboExperiment(DictMiningGemmaExperiment):
    # sa->bo model: identical recipe, separate output dir so it doesn't clobber
    # the zh-> model's checkpoint. Uses its own data processor (hence its own
    # dataset_cache dir) pointed at the sa->bo term-finetune data.
    DATA_PROCESSOR_CLS = DictionaryMiningSaboProcessor


class SpanProjection:
    """Whole-sentence span-projection task: given a per-direction instruction +
    numbered source + target sentence (all already in input_text), output the
    target sentence with the numbered tags inserted. Plain-text base-model prompt;
    the instruction is baked into input_text, so build_prompt only adds OUTPUT:."""
    # token dist (gemma tok): p50=319 p99=1025 p99.9=1491 max=3225; 512 truncates
    # 17% / 1024 truncates 1% / 2048 truncates ~0%. SFT pads to longest-in-batch,
    # so the higher cap is nearly free (only long sentences pay).
    MAX_INPUT_LENGTH = 2048
    DATA_PROCESSOR_CLS = SpanProjectionProcessor
    # target outputs are short (p99=271 tok); 448 covers them. Untrained/early model
    # won't emit EOS so it generates to this cap every example -> keep eval cheap.
    EVAL_GEN_CAP = 300
    EVAL_MAX_NEW_TOKENS = 448

    def build_prompt(self, tokenizer: PreTrainedTokenizerBase, source: str) -> str:
        return f"{source}\nOUTPUT:\n"


class SpanProjectionGemmaExperiment(
    GemmaMitraMixin, DictMiningFinetuneArgsMixin, SpanProjection, FinetuneDictMiningSFTExperimentBase
):
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 64
    EVAL_BATCH = 8
    MAX_INPUT_LENGTH = 2048   # set here: GemmaMitraMixin(512) precedes SpanProjection in MRO


class SpanProjectionEnBoGemmaExperiment(SpanProjectionGemmaExperiment):
    # English->Tibetan span projection on the Tsadra en-bo data (gemini-3.5-flash
    # mined). Identical recipe to SpanProjectionGemmaExperiment (full finetune of
    # gemma2-mitra-base, 8-bit Adam); only the data source differs, so it gets its
    # own output dir + dataset cache via the en-bo processor.
    DATA_PROCESSOR_CLS = SpanProjectionEnBoProcessor


class SpanProjectionGemma4Experiment(
    Gemma4E4BMixin, DictMiningFinetuneArgsMixin, SpanProjection, FinetuneDictMiningSFTExperimentBase
):
    # Same span-projection task/data/evaluator, on google/gemma-4-E4B instead of
    # gemma2-mitra-base. Separate output dir + dataset cache (own class).
    # Adafactor, NOT adamw_bnb_8bit: the bitsandbytes 8-bit kernel throws
    # "invalid configuration argument" on a gemma-4 multimodal tensor, and full
    # AdamW would OOM (fp32 states on ~7B params). Adafactor sidesteps both.
    OPTIM = "adafactor"
    TARGET_BATCH = 64
    EVAL_BATCH = 8
    MAX_INPUT_LENGTH = 2048   # set here: Gemma4E4BMixin(512) precedes SpanProjection in MRO


class SpanProjectionGemma4E2BExperiment(
    Gemma4E2BMixin, DictMiningFinetuneArgsMixin, SpanProjection, FinetuneDictMiningSFTExperimentBase
):
    # smaller E2B variant; same recipe as the E4B span experiment.
    OPTIM = "adafactor"
    TARGET_BATCH = 64
    EVAL_BATCH = 8
    MAX_INPUT_LENGTH = 2048


class AlignmentDenoising:
    """Sentence-alignment post-correction (sa-bo + sa-zh + bo-zh; no pali).
    Given an instruction + two position-numbered sentence lists (a drift-prone
    automatic alignment), output the corrected alignment chain "SRC_i,TGT_j;..."
    or NONE when the two sides are unrelated. The full instruction + both lists
    are baked into input_text, so build_prompt only appends OUTPUT:. Plain-text
    base-model prompt (gemma2-mitra-base has no chat template)."""
    # 10-sentence bilingual windows. Cap 2048 (median full-len ~1054 tok, p99 ~1847);
    # the ~0.5% of samples exceeding 2048 are CULLED at data-build time (not truncated,
    # which would chop the OUTPUT:/target off the end and corrupt the label).
    MAX_INPUT_LENGTH = 2048
    DATA_PROCESSOR_CLS = AlignmentDenoisingProcessor
    EVAL_GEN_CAP = 500
    # Target chains are short (train: p50=66, p99=106, max=156 tok). Cap at 128
    # (covers p99.9) and stop at the first newline so an EOS-less early model
    # halts at end-of-chain instead of hallucinating lines to the cap on every
    # example -- the chains are single-line, so '\n' is a safe stop.
    EVAL_MAX_NEW_TOKENS = 128
    EVAL_STOP = "\n"

    def build_prompt(self, tokenizer: PreTrainedTokenizerBase, source: str) -> str:
        return f"{source}\nOUTPUT:\n"


class AlignmentDenoisingGemmaExperiment(
    GemmaMitraMixin, DictMiningFinetuneArgsMixin, AlignmentDenoising, FinetuneDictMiningSFTExperimentBase
):
    # Same recipe as SpanProjectionGemmaExperiment (full finetune of
    # gemma2-mitra-base 9.5B, 8-bit Adam), separate data + output dir + cache.
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 64
    EVAL_BATCH = 8
    MAX_INPUT_LENGTH = 2048   # set here: GemmaMitraMixin(512) precedes AlignmentDenoising in MRO


class AlignmentDenoisingGemma4E2BExperiment(
    Gemma4E2BMixin, DictMiningFinetuneArgsMixin, AlignmentDenoising, FinetuneDictMiningSFTExperimentBase
):
    # Same alignment-denoising task/data on google/gemma-4-E2B instead of
    # gemma2-mitra-base. adafactor (NOT adamw_bnb_8bit): the bitsandbytes 8-bit
    # kernel errors on a gemma-4 multimodal tensor, and full AdamW would OOM.
    # Shares AlignmentDenoisingProcessor data; separate output dir (own class).
    OPTIM = "adafactor"
    TARGET_BATCH = 64
    EVAL_BATCH = 8
    MAX_INPUT_LENGTH = 2048   # set here: Gemma4E2BMixin(512) precedes AlignmentDenoising in MRO


class MitraKnnTranslation:
    """Whole-sentence translation SFT on the gold-anchored knn-RAG mined data.

    input_text is ALREADY the full production gemma4 knn-translate prompt (with
    the example's own gold pair stripped from the retrieved references), so
    build_prompt only adds a trailing newline for a clean prompt/completion
    boundary (the prompt ends in a ':' — a bare colon merges with the first
    completion token, breaking completion-only loss masking). target_text is the
    Gemini "improve on the gold" translation.
    """
    DATA_PROCESSOR_CLS = MitraKnnSftProcessor
    # The data is CULLED at build time to <= MITRAKNN_MAX_INPUT_LENGTH tokens
    # (build_sft_data.py --cap), so nothing is truncated. Default 2048 matches
    # the working span/alignment gemma experiments and fits a full 9.5B finetune
    # on one GPU; raise the cap (and rebuild the data with a matching --cap) only
    # with FSDP / more memory. Set on the EXPERIMENT class too since
    # GemmaMitraMixin(512) precedes this mixin in the MRO.
    MAX_INPUT_LENGTH = int(os.environ.get("MITRAKNN_MAX_INPUT_LENGTH", "2048"))
    EVAL_GEN_CAP = 300          # full val set
    EVAL_MAX_NEW_TOKENS = 256   # full sentence translations, not short terms

    def build_prompt(self, tokenizer: PreTrainedTokenizerBase, source: str) -> str:
        return f"{source}\n"


class KnnTranslationGemmaMitraExperiment(
    GemmaMitraMixin, DictMiningFinetuneArgsMixin, MitraKnnTranslation, FinetuneDictMiningSFTExperimentBase
):
    # Full finetune of gemma2-mitra-base (9.5B base model) with 8-bit Adam, same
    # recipe as the other Gemma experiments. Long context (knn refs) -> small
    # per-device batch; effective batch held at 64 via grad accumulation.
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 64
    EVAL_BATCH = 1   # eval generation on long prompts is memory-heavy; keep at 1
    # MRO: GemmaMitraMixin(512) precedes MitraKnnTranslation, so set the caps on
    # the experiment class itself to win. Env-tunable; rebuild data with a
    # matching build_sft_data.py --cap if you change it.
    MAX_INPUT_LENGTH = int(os.environ.get("MITRAKNN_MAX_INPUT_LENGTH", "2048"))
    EVAL_GEN_CAP = 300
    EVAL_MAX_NEW_TOKENS = 256


class SpanProjectionQwenDictExperiment(
    Qwen35Stage29BMixin, DictMiningFinetuneArgsMixin, SpanProjection, FinetuneDictMiningSFTExperimentBase
):
    """v2 dictionary span-projection: 9B stage2 (ck6145) full SFT on the four
    strict-mined directions simultaneously (sa-bo, sa-zh, zh-bo, en-bo; Tibetan
    in unicode; minimal instruction). Run one epoch: DICTMINING_NUM_EPOCHS=1.
    8-bit Adam so the 9B full finetune fits a single GPU."""
    DATA_PROCESSOR_CLS = QwenDictSpanProcessor
    OPTIM = "adamw_bnb_8bit"
    # per-process target: the accum math is not world-size aware, so divide by
    # WORLD_SIZE (torchrun sets it) to keep the effective batch at 64 under DDP
    TARGET_BATCH = 64 // int(os.environ.get("WORLD_SIZE", "1"))
    # eval loss materializes full 248k-vocab logits (no fused CE) -> batch 8
    # OOMs next to resident training state; 2 fits
    EVAL_BATCH = 2
    MAX_INPUT_LENGTH = 2048   # set here: mixin default precedes SpanProjection in MRO


class SpanProjectionQwenDictSaenExperiment(SpanProjectionQwenDictExperiment):
    """Same recipe as SpanProjectionQwenDictExperiment (9B stage2 ck6145 full SFT,
    8-bit Adam, one epoch), on the five-direction data that adds Mitrasamgraha
    sa-en (Sanskrit -> English, English target tagged). Own class => own output
    dir + dataset cache; the four-direction checkpoint is left untouched."""
    DATA_PROCESSOR_CLS = QwenDictSaenSpanProcessor


class TibSegTerse:
    """Tibetan learner annotation: Wylie block -> terse multi-line annotation
    (sentences per line, words with /POS, [ ] Sanskrit-unit brackets). Plain-text
    base-model prompt; the instruction is baked into input_text."""
    MAX_INPUT_LENGTH = 2048
    DATA_PROCESSOR_CLS = TibSegTerseProcessor
    EVAL_GEN_CAP = 100
    # targets are <= 1024 chars of ASCII Wylie (~450 Qwen tokens); 768 is safe
    EVAL_MAX_NEW_TOKENS = 768

    def build_prompt(self, tokenizer, source: str) -> str:
        return f"{source}\nOUTPUT:\n"


class TibSegTerseQwenExperiment(
    Qwen35Stage29BMixin, DictMiningFinetuneArgsMixin, TibSegTerse, FinetuneDictMiningSFTExperimentBase
):
    """9B stage2 (ck6145) full SFT on the 5k terse Tibetan annotation set, 8-bit
    Adam, single GPU. Small data -> smaller effective batch (32) and 2 epochs:
    DICTMINING_NUM_EPOCHS=2 ./run_tibseg.sh"""
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 32 // int(os.environ.get("WORLD_SIZE", "1"))
    EVAL_BATCH = 2
    # instruction (~130 tok) + Wylie (<=700 chars) + target (<=1024 chars) < 1000 tokens;
    # a smaller cap keeps the 248k-vocab logits small on the squeezed GPU1 (TIBSEG_MAX_LEN)
    MAX_INPUT_LENGTH = int(os.environ.get("TIBSEG_MAX_LEN", "2048"))


class ZhSegTerse:
    """Buddhist Chinese annotation: raw unpunctuated window -> terse multi-line
    annotation (sentences per line, WORD/POS, MARK/p punctuation, [ ]=T compound
    units). Plain-text base-model prompt; the instruction is baked into input_text.
    Chinese counterpart of TibSegTerse."""
    MAX_INPUT_LENGTH = 2048
    DATA_PROCESSOR_CLS = ZhSegTerseProcessor
    EVAL_GEN_CAP = 100
    # targets: <= 200 CJK chars -> ~150 tokens "w/X " (~4 tokens each) => ~700 tokens; 1024 is safe
    EVAL_MAX_NEW_TOKENS = 1024

    def build_prompt(self, tokenizer, source: str) -> str:
        return f"{source}\nOUTPUT:\n"


class ZhSegTerseQwenExperiment(
    Qwen35Stage29BMixin, DictMiningFinetuneArgsMixin, ZhSegTerse, FinetuneDictMiningSFTExperimentBase
):
    """9B stage2 (ck6145) full SFT on the ~1k Gemini-gold Chinese annotation set,
    8-bit Adam, single GPU. Same recipe as TibSegTerseQwenExperiment:
    DICTMINING_NUM_EPOCHS=2 ./run_zhseg.sh"""
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 32 // int(os.environ.get("WORLD_SIZE", "1"))
    EVAL_BATCH = 2
    MAX_INPUT_LENGTH = 2048


class JointSegTerseQwenExperiment(ZhSegTerseQwenExperiment):
    """Joint Chinese + Tibetan terse annotation: same recipe as ZhSegTerseQwenExperiment
    on the merged set (Chinese windows + sentences x2, Tibetan sentences + paragraphs).
    val/test are zh/bo interleaved, so EVAL_GEN_CAP=100 evaluates 50 of each.
    DICTMINING_NUM_EPOCHS=2 ./run_jointseg.sh"""
    DATA_PROCESSOR_CLS = JointSegTerseProcessor
    EVAL_GEN_CAP = 200          # final test phase: all 100 zh + 100 bo (run script caps in-training eval to 100)
    EVAL_MAX_NEW_TOKENS = 1024


# ---- 2026-10-07: base-model ablation for the MITRA-dict paper. Identical recipe and data as
# SpanProjectionQwenDictSaenExperiment (five-direction span projection, one epoch, full finetune, 8-bit Adam,
# effective batch 64, lr 2e-5); only the base model differs. Own class names -> own output dirs.
class SpanProjectionDictSaenVanillaQwen9BExperiment(SpanProjectionQwenDictSaenExperiment):
    """Off-the-shelf Qwen3.5-9B-Base (no Buddhist continued pretraining)."""
    MODEL_NAME = os.environ.get("ABL_VANILLA_QWEN9B", "Qwen/Qwen3.5-9B-Base")


class SpanProjectionDictSaenMitraQwen2BExperiment(SpanProjectionQwenDictSaenExperiment):
    """MITRA continued-pretraining Qwen3.5-2B (stage 2 base)."""
    MODEL_NAME = os.environ.get("ABL_MITRA_QWEN2B", "/mnt/data/mitra-qwen35-2b-base-stage2")


class SpanProjectionDictSaenGemmaMitraExperiment(
    GemmaMitraMixin, DictMiningFinetuneArgsMixin, SpanProjection, FinetuneDictMiningSFTExperimentBase
):
    """gemma2-mitra-base (9.5B, the previous MITRA generation), same task, data and recipe."""
    DATA_PROCESSOR_CLS = QwenDictSaenSpanProcessor
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 64 // int(os.environ.get("WORLD_SIZE", "1"))
    EVAL_BATCH = 2
    MAX_INPUT_LENGTH = 2048


# ---- 2026-10-07: Wylie control. Same data with every Tibetan line in Wylie (ablation/to_wylie.py), same recipe.
class QwenDictSaenWylieSpanProcessor(QwenDictSaenSpanProcessor):
    BASE = os.path.expanduser("~/data/qwen-dictionaries-spanmine-saen-wylie")


class SpanProjectionDictSaenWylieMitraQwen9BExperiment(SpanProjectionQwenDictSaenExperiment):
    """MITRA Qwen3.5-9B (stage 2), Tibetan in Wylie."""
    DATA_PROCESSOR_CLS = QwenDictSaenWylieSpanProcessor


class SpanProjectionDictSaenWylieGemmaMitraExperiment(SpanProjectionDictSaenGemmaMitraExperiment):
    """gemma2-mitra-base, Tibetan in Wylie."""
    DATA_PROCESSOR_CLS = QwenDictSaenWylieSpanProcessor


class SpanProjectionDictSaenMitraQwen2B3epExperiment(SpanProjectionDictSaenMitraQwen2BExperiment):
    """Compute-matched control: MITRA Qwen3.5-2B for 3 epochs (~ the wall time of one 9B epoch: 13 vs 39 min)."""


class SpanProjectionQwenDictSaen3epExperiment(SpanProjectionQwenDictSaenExperiment):
    """Control: the MITRA Qwen3.5-9B student (five-direction data, same recipe) trained for 3 epochs."""


class SegPostcorrection:
    """Alignment post-correction: v2 vecalign beads (+ the dropped nexus segments as one-sided
    beads, {a1}/{b1}-prefixed) -> corrected prefix-only bead chain "a1 b1 # a2 b2 b3 # a3".
    The instruction is baked into input_text."""
    MAX_INPUT_LENGTH = 4096       # data culled at build time to <= 4000 tokens (prompt + chain)
    DATA_PROCESSOR_CLS = SegPostcorrectionProcessor
    EVAL_GEN_CAP = 200
    # chains: <= ~45+45 segment ids at ~3 tokens each + separators -> < 400 tokens
    EVAL_MAX_NEW_TOKENS = 512
    EVAL_STOP = "\n"             # chains are single-line

    def build_prompt(self, tokenizer, source: str) -> str:
        return f"{source}\nOUTPUT:\n"


class SegPostcorrectionQwenExperiment(
    Qwen35Stage29BMixin, DictMiningFinetuneArgsMixin, SegPostcorrection, FinetuneDictMiningSFTExperimentBase
):
    """9B stage2 (ck6145) full SFT on ~1k gemini-mined post-correction windows, 8-bit Adam,
    single GPU / single process (no NCCL on this rig). ./run_segpostcorr.sh"""
    OPTIM = "adamw_bnb_8bit"
    TARGET_BATCH = 32
    EVAL_BATCH = 1
    MAX_INPUT_LENGTH = 4096


class SegPostcorrectionV2QwenExperiment(SegPostcorrectionQwenExperiment):
    """Same recipe on the v2 post-correction data (~4.3k windows). ./run_segpostcorr.sh with
    DICTMINING_EXPERIMENT=SegPostcorrectionV2QwenExperiment (see run_segpostcorr_v2.sh)."""
    DATA_PROCESSOR_CLS = SegPostcorrectionV2Processor
    EVAL_GEN_CAP = 120


class SegPostcorrectionV3QwenExperiment(SegPostcorrectionQwenExperiment):
    """Same recipe on the v3 post-correction data: v2 + Tibetan-English (bo-en) windows (~5.7k train).
    ./run_segpostcorr_v3.sh"""
    DATA_PROCESSOR_CLS = SegPostcorrectionV3Processor
    EVAL_GEN_CAP = 140
