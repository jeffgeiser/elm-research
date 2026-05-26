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
import subprocess
import sys
from pathlib import Path

# Unsloth must be imported before transformers/peft to apply its patches.
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import TrainingArguments, TrainerCallback  # noqa: E402
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  # noqa: E402


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"
EVAL_SCRIPT = ROOT / "eval.py"


# ---- Hyperparameters --------------------------------------------------------
# Locked per request — adjust here, not via CLI, so the choice is
# versioned alongside the run.

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 16384   # p95 ~12k, max ~13.4k; headroom for chat template tokens

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

# Response template — the chat-template marker that prefixes the assistant
# turn for Qwen2.5. Used by DataCollatorForCompletionOnlyLM to mask the
# system + user prefix from the loss. Without this, the model learns to
# regenerate the prompt instead of the brief.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


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


# ---- Post-checkpoint eval hook ---------------------------------------------

class PostCheckpointEvalCallback(TrainerCallback):
    """After each save, run inference on the held-out 50, save predictions
    by their canonical example-NNN.json names, then invoke eval.py to
    produce a report card alongside the checkpoint.

    ID binding is by the "id" field in each eval.jsonl record — NOT by
    line order. Robust to dataset reordering.
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
        out_dir = ckpt_dir / "eval_outputs"
        out_dir.mkdir(exist_ok=True)

        print(f"\n[eval] running inference on held-out 50 for {ckpt_dir.name}...")
        FastLanguageModel.for_inference(self.model)

        n_done = 0
        n_parse_ok = 0
        with open(self.eval_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                example_id = rec.get("id")
                if not example_id:
                    print(f"[eval] WARN: record missing 'id' field; skipping")
                    continue

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

                # Save raw and parsed
                (out_dir / f"{example_id}.raw.txt").write_text(gen)
                try:
                    parsed = json.loads(gen)
                    (out_dir / f"{example_id}.json").write_text(
                        json.dumps(parsed, indent=2)
                    )
                    n_parse_ok += 1
                except json.JSONDecodeError:
                    pass  # eval.py will flag missing files / parse failures
                n_done += 1

        FastLanguageModel.for_training(self.model)

        report_path = self.run_dir / f"eval_{state.global_step}.txt"
        with open(report_path, "w") as rep:
            subprocess.run(
                [sys.executable, str(EVAL_SCRIPT), "--model", str(out_dir)],
                stdout=rep, stderr=subprocess.STDOUT, check=False,
            )
        print(f"[eval] {n_done} predictions ({n_parse_ok} parsed) → {report_path.name}")


# ---- Main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--eval-jsonl", type=Path, default=DATA_DIR / "eval.jsonl")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Subdirectory under train/runs/ (default: timestamp)")
    args = ap.parse_args()

    if not args.train_jsonl.exists():
        print(f"ERROR: {args.train_jsonl} not found. Run format_jsonl.py first.",
              file=sys.stderr)
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
    targs = TrainingArguments(
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
    )

    # Completion-only loss: mask everything up to and including the
    # assistant marker. Without this, training learns to regenerate the
    # system prompt instead of the brief (observed empirically in round 1
    # — model collapsed into prompt-regurgitation + repetition loops).
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        data_collator=collator,
        packing=False,
        args=targs,
    )

    trainer.add_callback(VramLogCallback(run_dir))
    trainer.add_callback(PostCheckpointEvalCallback(run_dir, args.eval_jsonl, model, tokenizer))

    print("Starting training...")
    trainer.train()

    # Final save
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final adapter saved to {final_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
