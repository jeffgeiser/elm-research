#!/usr/bin/env python3
"""Generate model predictions for the held-out eval set.

This is the manual, run-after-training counterpart to eval.py. The old
PostCheckpointEvalCallback used to generate predictions inline at every
checkpoint, which wedged training for 8+ hours. That callback is gone;
this script does the same generation ONCE, on demand, against a finished
adapter — then you score the output dir with eval.py.

Flow:
    # 1. Generate predictions from the trained adapter (run on the Spark)
    python infer_eval.py --adapter train/runs/round4/final \
        --out train/runs/round4/eval_outputs

    # 2. Score them
    python eval.py --model train/runs/round4/eval_outputs

Outputs, per eval record, into --out:
    <id>.raw.txt   # raw decoded generation (always written)
    <id>.json      # pretty-printed, only if the raw text parses as JSON
                   # (eval.py reads these; a missing one = parse failure,
                   #  which eval.py reports as a schema miss)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Unsloth must be imported before transformers/peft to apply its patches.
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402


HERE = Path(__file__).parent.resolve()
DEFAULT_EVAL_JSONL = HERE / "train" / "data" / "eval.jsonl"

_DECODER = json.JSONDecoder()


def extract_first_json(text: str) -> dict | None:
    """Parse the FIRST complete JSON object in `text`, tolerating leading
    prose / markdown fences and trailing junk.

    The trained model often appends a second `{"_meta": {...}}` object after
    the brief — a plain json.loads() then dies with 'Extra data'. raw_decode
    stops at the end of the first value and ignores whatever follows, which
    salvages those cases. Returns None if no object can be parsed.
    """
    if not text:
        return None
    # Drop a leading ```json / ``` fence if present, then seek the first '{'.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
    start = stripped.find("{")
    if start == -1:
        return None
    try:
        obj, _end = _DECODER.raw_decode(stripped[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to the trained LoRA adapter dir (e.g. "
                         "train/runs/round4/final). Unsloth loads the base "
                         "model named in its adapter_config automatically.")
    ap.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL,
                    help="Eval records with messages[] + id (default: "
                         "train/data/eval.jsonl)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir for predictions (default: "
                         "<adapter>/../eval_outputs)")
    ap.add_argument("--max-seq-length", type=int, default=8192,
                    help="Must match (or exceed) the training seq length so "
                         "long inputs aren't truncated differently than in "
                         "training. Round 4 trained at 8192.")
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="Generation budget for the JSON brief.")
    ap.add_argument("--repetition-penalty", type=float, default=1.15,
                    help="Penalize token repetition to break greedy "
                         "degeneration loops (the '0000...' / 'sf_activity...' "
                         "tails). 1.0 disables; 1.1-1.2 is a safe range.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only generate for the first N eval records — for a "
                         "quick schema-adherence read before committing to "
                         "the full set. eval.py will warn about the rest as "
                         "missing; that's expected on a limited run.")
    args = ap.parse_args()

    if not args.adapter.exists():
        print(f"ERROR: adapter dir {args.adapter} not found", file=sys.stderr)
        return 1
    if not args.eval_jsonl.exists():
        print(f"ERROR: eval file {args.eval_jsonl} not found", file=sys.stderr)
        return 1

    out_dir = args.out or (args.adapter.parent / "eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Adapter: {args.adapter}")
    print(f"Eval:    {args.eval_jsonl}")
    print(f"Out:     {out_dir}")

    # Unsloth detects the PEFT adapter_config and loads base + adapter.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    n_done = 0
    n_parse_ok = 0
    with open(args.eval_jsonl) as f:
        for line in f:
            if args.limit is not None and n_done >= args.limit:
                print(f"--limit {args.limit} reached; stopping early.")
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            example_id = rec.get("id")
            if not example_id:
                print("WARN: record missing 'id' field; skipping")
                continue

            # Drop the assistant turn — the model must predict it.
            prompt_msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
            input_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            (out_dir / f"{example_id}.raw.txt").write_text(gen)
            parsed = extract_first_json(gen)
            if parsed is not None:
                (out_dir / f"{example_id}.json").write_text(
                    json.dumps(parsed, indent=2)
                )
                n_parse_ok += 1
            # else: eval.py will flag the missing .json as a schema miss
            n_done += 1
            print(f"  [{n_done}] {example_id}: "
                  f"{'parsed' if (out_dir / f'{example_id}.json').exists() else 'RAW ONLY (parse failed)'}")

    print(f"\nDone: {n_done} predictions, {n_parse_ok} parsed as JSON → {out_dir}")
    print(f"Now score with:  python eval.py --model {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
