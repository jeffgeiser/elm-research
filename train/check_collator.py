#!/usr/bin/env python3
"""Smoke test for the completion-only collator.

Verifies in seconds (no GPU needed) that the collator's masking strategy
actually works against a real chat-template-formatted record from
train.jsonl. Catches the round-2 bug (train_loss=0 because nothing
matched) before committing to a multi-hour run.

Run from the repo root:
    python train/check_collator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from train_lora import CompletionOnlyCollator, BASE_MODEL  # noqa: E402


def main() -> int:
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    # Read one real record from train.jsonl
    train_path = HERE / "data" / "train.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run format_jsonl.py first.")
        return 1
    with open(train_path) as f:
        rec = json.loads(f.readline())
    print(f"Sample record id={rec.get('id', '?')}")

    text = tokenizer.apply_chat_template(
        rec["messages"], tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Sequence length: {len(ids)} tokens")

    # Build a fake batch the way SFTTrainer would
    example = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids[:]}
    batch = collator.torch_call([example])

    labels = batch["labels"][0]
    masked = int((labels == -100).sum())
    kept = int((labels != -100).sum())
    print(f"Masked: {masked} / Kept: {kept}")
    print(f"Mask ratio: {100 * masked / len(labels):.1f}%")

    if kept == 0:
        print("FAIL: every label is masked — loss would be 0")
        return 1
    if masked == 0:
        print("FAIL: nothing masked — would train on the system prompt")
        return 1
    if kept < 100:
        print(f"WARN: only {kept} tokens contribute to loss — suspiciously small")

    # Show the first/last few un-masked tokens (the assistant content boundary)
    keep_positions = (labels != -100).nonzero(as_tuple=True)[0]
    if len(keep_positions) > 0:
        first_kept = keep_positions[0].item()
        last_kept = keep_positions[-1].item()
        boundary = tokenizer.decode(ids[max(0, first_kept - 5) : first_kept + 5])
        tail = tokenizer.decode(ids[last_kept - 5 : last_kept + 1])
        print(f"\nBoundary at position {first_kept}:")
        print(f"  ...{boundary!r}...")
        print(f"Tail at position {last_kept}:")
        print(f"  ...{tail!r}")

    print("\nPASS: collator masking works as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
