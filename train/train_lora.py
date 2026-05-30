#!/usr/bin/env python3
"""QLoRA fine-tune of Qwen2.5-7B-Instruct on the account intelligence
gold dataset.

Stack: Unsloth (4-bit base + custom kernels) + TRL SFTTrainer.
Hardware: DGX Spark, 64 GB VRAM, CUDA.
Sequence length: 16384 (covers p95 ~12k + headroom).
Effective batch: 16 (batch_size=1 * grad_accum=16).
    Conservative starting point at seq_len=16k. If VRAM headroom
    confirmed > 20 GB after the first 50 steps, bump per_device_batch
    to 2 and grad_accum to 8 to keep effective batch=16.

Expected runtime for 1 epoch on ~520 examples @ seq_len 16384:
    ~2–3 hours on a single 64 GB H100-class device.

Run via train/run.sh (tmux + log file) so it survives SSH disconnect.

Outputs (per run, under train/runs/<timestamp>/):
    checkpoint-NNN/                  # LoRA adapter shards
    checkpoint-NNN/eval_outputs/     # per-eval-ID predicted JSON
    eval_NNN.txt                     # eval.py report against held-out 50
    train.log                        # full stdout+stderr (via run.sh)
    vram.log                         # peak VRAM per 50 steps
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Unsloth must be imported before transformers/peft to apply its patches.
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from transformers import DataCollatorForLanguageModeling  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"


# ---- Hyperparameters --------------------------------------------------------
# Locked per request — adjust here, not via CLI, so the choice is
# versioned alongside the run.

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
# Restored to 16384 for sole-tenant runs (vLLM stopped). Covers p95~12k
# and max~13.4k token records without truncation. Drop to 8192 if you
# need to coexist with vLLM or other GPU workloads.
MAX_SEQ_LENGTH = 16384

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0       # Unsloth-recommended for speed
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE = 1e-4     # was 2e-4; dropped after round 1 over-memorized prompts
PER_DEVICE_BATCH = 1     # conservative at seq_len=16k; can bump to 2 if VRAM allows
GRAD_ACCUM_STEPS = 16    # keep effective batch = 16
NUM_EPOCHS = 1
WARMUP_STEPS = 5         # ~15% of total ~33 steps
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# Save + eval cadence — sized for ~33-step runs
SAVE_STEPS = 10
EVAL_STEPS = 10
LOGGING_STEPS = 1
VRAM_LOG_INTERVAL = 5    # log peak VRAM every N steps

# Completion-only masking uses Qwen2.5 special tokens directly inside the
# CompletionOnlyCollator (no string template needed).


# ---- Completion-only collator ----------------------------------------------

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Mask everything before the assistant turn with -100 so loss is only
    computed on the assistant content.

    Uses Qwen2.5's special token IDs directly (<|im_start|> + role name)
    rather than string-encoding the template, which proved brittle: the
    BPE tokenization of a standalone template string can differ from how
    those tokens appear mid-sequence after apply_chat_template.

    Verifies on init that the marking strategy actually works against a
    sample chat sequence — fails loudly rather than silently masking
    everything (which round 2 did, producing train_loss=0).
    """

    def __init__(self, tokenizer, ignore_index: int = -100):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.ignore_index = ignore_index

        # Resolve Qwen2.5 special tokens
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if self.im_start_id is None or self.im_start_id == tokenizer.unk_token_id:
            raise RuntimeError(
                "Tokenizer has no <|im_start|> special token — this collator "
                "only supports Qwen2.5-style chat templates."
            )
        # "assistant" tokenizes to one or more BPE pieces. Capture them
        # so we can verify the role name follows <|im_start|>.
        self.assistant_role_ids = tokenizer.encode("assistant", add_special_tokens=False)

        # Self-test on a synthetic sequence. If masking would mask EVERYTHING
        # or NOTHING, refuse to construct.
        sample_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "ASSISTANT_CONTENT_MARKER"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        cutoff = self._find_assistant_cutoff(sample_ids)
        if cutoff is None:
            raise RuntimeError(
                "Self-test FAILED: could not find assistant role marker in "
                f"sample tokenization. im_start_id={self.im_start_id}, "
                f"assistant_role_ids={self.assistant_role_ids}, "
                f"sample_ids tail={sample_ids[-30:]}"
            )
        if cutoff >= len(sample_ids):
            raise RuntimeError(
                f"Self-test FAILED: cutoff {cutoff} is at or past end of "
                f"sample (len={len(sample_ids)}) — no assistant content to "
                "compute loss on."
            )
        print(
            f"[collator] self-test OK: cutoff={cutoff}/{len(sample_ids)} "
            f"(mask {cutoff} prefix tokens, train on {len(sample_ids)-cutoff} suffix)"
        )

    def _find_assistant_cutoff(self, ids):
        """Return the index right after `<|im_start|>assistant\\n` for the
        LAST assistant turn. Returns None if not found."""
        n_role = len(self.assistant_role_ids)
        last_cutoff = None
        for pos in range(len(ids) - n_role - 1):
            if ids[pos] != self.im_start_id:
                continue
            if ids[pos + 1 : pos + 1 + n_role] != self.assistant_role_ids:
                continue
            # Skip the role tokens; mask through the newline that follows.
            cutoff = pos + 1 + n_role
            # Advance past a single \n token if present (Qwen2.5 newline = id 198).
            if cutoff < len(ids):
                newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
                if newline_ids and ids[cutoff] == newline_ids[0]:
                    cutoff += 1
            last_cutoff = cutoff
        return last_cutoff

    _debug_dumped = False

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # One-shot debug on the very first batch — proves the collator
        # actually ran AND shows what's in the labels before we mask.
        if not CompletionOnlyCollator._debug_dumped:
            CompletionOnlyCollator._debug_dumped = True
            print("\n=== [collator DEBUG] first batch ===")
            print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
            print(f"labels shape:    {tuple(batch['labels'].shape)}")
            ids0 = batch["input_ids"][0].cpu().tolist()
            lbl0 = batch["labels"][0].cpu().tolist()
            already_masked = sum(1 for x in lbl0 if x == -100)
            print(f"first example length: {len(ids0)}")
            print(f"labels already -100 before our masking: {already_masked}")
            im_start_positions = [i for i, t in enumerate(ids0) if t == self.im_start_id]
            print(f"<|im_start|> positions in first example: {im_start_positions}")
            for pos in im_start_positions:
                role_tokens = ids0[pos + 1 : pos + 6]
                role_decoded = self.tokenizer.decode(role_tokens, skip_special_tokens=False)
                print(f"  pos {pos}: next tokens={role_tokens} → {role_decoded!r}")
            cutoff = self._find_assistant_cutoff(ids0)
            print(f"computed cutoff for first example: {cutoff}")
            print("=== end debug ===\n")

        for i in range(batch["labels"].size(0)):
            labels = batch["labels"][i].cpu().tolist()
            cutoff = self._find_assistant_cutoff(labels)
            if cutoff is None:
                batch["labels"][i, :] = self.ignore_index
            else:
                batch["labels"][i, :cutoff] = self.ignore_index
        return batch


# ---- VRAM logging callback --------------------------------------------------

class VramLogCallback(TrainerCallback):
    """Logs peak VRAM (allocated + reserved) every VRAM_LOG_INTERVAL steps
    so we can see headroom before committing a full epoch.

    Writes vram.log next to the run directory:
        step=50  loss=2.341  peak_alloc=42.1GB  peak_reserved=48.7GB  free=15.3GB
    """

    def __init__(self, run_dir: Path, total_vram_gb: float | None = None):
        self.run_dir = run_dir
        self.log_path = run_dir / "vram.log"
        self.log_path.write_text("# VRAM peak usage per training step\n")
        if total_vram_gb is None and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.total_vram_gb = total_vram_gb or 0.0
        self._last_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # SFTTrainer emits {"loss": ..., ...} on logging_steps. Capture it
        # so we can write loss alongside VRAM at the next interval tick.
        if logs and "loss" in logs:
            self._last_loss = logs["loss"]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % VRAM_LOG_INTERVAL != 0:
            return
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        free = max(self.total_vram_gb - reserved, 0.0)
        loss_str = f"{self._last_loss:.4f}" if self._last_loss is not None else "n/a"
        line = (
            f"step={state.global_step}  "
            f"loss={loss_str}  "
            f"peak_alloc={alloc:.1f}GB  "
            f"peak_reserved={reserved:.1f}GB  "
            f"free={free:.1f}GB\n"
        )
        with open(self.log_path, "a") as fp:
            fp.write(line)
        print(f"[vram] {line.strip()}")
        torch.cuda.reset_peak_memory_stats()


# ---- Post-checkpoint sanity-check hook -------------------------------------

# Number of eval examples to generate on each save — just enough to eyeball
# that the model isn't broken (empty output, prompt-regurgitation, repetition
# loops). NOT a quality eval. Full eval.py against the held-out 50 runs
# MANUALLY after training, never inline — round 3 wedged for 8+ hours when
# the old callback ran inference on all 50 at every checkpoint.
SANITY_N = 3


class SanityCheckCallback(TrainerCallback):
    """After each save, generate output for exactly SANITY_N eval examples
    and write them to <checkpoint>/sanity_outputs/. No JSON parsing, no
    scoring, no eval.py — purely a "is the model producing sane text?"
    smoke test that costs a few minutes, not hours.

    Example selection is the first SANITY_N records of eval.jsonl in file
    order — deterministic across checkpoints so outputs are comparable.
    """

    def __init__(self, run_dir: Path, eval_jsonl: Path, model, tokenizer):
        self.run_dir = run_dir
        self.eval_jsonl = eval_jsonl
        self.model = model
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        out_dir = ckpt_dir / "sanity_outputs"
        out_dir.mkdir(exist_ok=True)

        print(f"\n[sanity] generating {SANITY_N} samples for {ckpt_dir.name}...")
        FastLanguageModel.for_inference(self.model)

        n_done = 0
        with open(self.eval_jsonl) as f:
            for line in f:
                if n_done >= SANITY_N:
                    break
                rec = json.loads(line)
                example_id = rec.get("id") or f"line{n_done}"

                # Drop the assistant turn — model must predict it.
                prompt_msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
                input_text = self.tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=8192,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen = self.tokenizer.decode(
                    output[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                (out_dir / f"{example_id}.raw.txt").write_text(gen)
                n_done += 1

        FastLanguageModel.for_training(self.model)
        print(f"[sanity] {n_done} samples → {out_dir.relative_to(self.run_dir.parent)}")


# ---- Main -------------------------------------------------------------------

def preflight_gpu_check(min_free_gb: float = 60.0) -> bool:
    """Refuse to start if another process is hogging GPU memory. Round 4
    crashed the system because vLLM + training exceeded the unified-memory
    pool. Better to fail fast than burn hours setting up before OOM."""
    if not torch.cuda.is_available():
        print("WARN: CUDA not available — preflight skipped")
        return True
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    total_gb = total_bytes / 1e9
    print(f"[preflight] GPU memory: {free_gb:.1f} GB free of {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(
            f"ERROR: only {free_gb:.1f} GB GPU memory free, need {min_free_gb}+ GB. "
            "Another process is using the GPU. Stop it (e.g. vLLM) or pass "
            "--force to proceed anyway."
        )
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--eval-jsonl", type=Path, default=DATA_DIR / "eval.jsonl")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Subdirectory under train/runs/ (default: timestamp)")
    ap.add_argument("--force", action="store_true",
                    help="Skip GPU-memory preflight check")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Path to a checkpoint dir to resume from, e.g. "
                         "train/runs/round3-clean/checkpoint-10. Reloads "
                         "optimizer/scheduler/step state and continues.")
    args = ap.parse_args()

    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).exists():
        print(f"ERROR: resume checkpoint {args.resume_from_checkpoint} not found.",
              file=sys.stderr)
        return 1

    if not args.train_jsonl.exists():
        print(f"ERROR: {args.train_jsonl} not found. Run format_jsonl.py first.",
              file=sys.stderr)
        return 1

    if not args.force and not preflight_gpu_check():
        return 1

    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"max_seq_length={MAX_SEQ_LENGTH}, batch={PER_DEVICE_BATCH}, "
          f"grad_accum={GRAD_ACCUM_STEPS} (effective={PER_DEVICE_BATCH * GRAD_ACCUM_STEPS})")

    # ---- Load 4-bit base + attach LoRA ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,        # auto: bf16 on H100
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # ~30% memory savings
        random_state=42,
    )

    # ---- Datasets ----
    train_ds = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_ds = load_dataset("json", data_files=str(args.eval_jsonl), split="train")

    # Pre-apply the chat template to a "text" column. Unsloth's SFTTrainer
    # requires an explicit formatting target — auto-detect from "messages"
    # is unreliable across TRL/Unsloth versions.
    def to_text(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_text, remove_columns=eval_ds.column_names)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ---- Training args ----
    # Use SFTConfig (not TrainingArguments) so max_seq_length is actually
    # honored. In TRL 0.20+, max_seq_length passed directly to SFTTrainer
    # is ignored — it has to come through SFTConfig. Round 3 silently
    # truncated to 1024 because of this.
    targs = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=4,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        report_to="none",
        seed=42,
        optim="adamw_8bit",
        # SFT-specific (these were being silently ignored on SFTTrainer)
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    # Completion-only loss: mask everything up to and including the
    # assistant marker. Without this, training learns to regenerate the
    # system prompt instead of the brief (observed empirically in round 1
    # — model collapsed into prompt-regurgitation + repetition loops).
    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=targs,
    )

    trainer.add_callback(VramLogCallback(run_dir))
    trainer.add_callback(SanityCheckCallback(run_dir, args.eval_jsonl, model, tokenizer))

    # Resume support. NOTE: SFTConfig/TrainingArguments has a
    # `resume_from_checkpoint` field, but the HF Trainer does NOT read it —
    # the docs explicitly say it's "not directly used by Trainer." The only
    # path that actually reloads optimizer/scheduler/trainer state is
    # passing it to trainer.train(). So we route the CLI flag through here.
    print(f"Starting training...{' resuming from ' + args.resume_from_checkpoint if args.resume_from_checkpoint else ''}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    # Final save
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final adapter saved to {final_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
