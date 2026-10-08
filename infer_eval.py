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

Base-model baseline (no adapter, no Unsloth — plain transformers on
CUDA or Apple MPS):
    python infer_eval.py --no-adapter --out train/runs/baseline/eval_outputs

Quantized GGUF export (llama.cpp's llama-completion must be on PATH; runs
on Apple Silicon):
    python infer_eval.py --gguf models/account-intelligence-7b-round10.Q4_K_M.gguf \
        --out train/runs/gguf_eval

Generation is schema-constrained with Outlines (see constrained.py): the
model can only emit tokens that keep the output a valid brief. Pass
--unconstrained to reproduce the old free-running generate() baseline.
The --gguf path applies the same cleaned schema through llama.cpp's
JSON-schema grammar instead of Outlines.

Outputs, per eval record, into --out:
    <id>.raw.txt   # raw decoded generation (always written)
    <id>.json      # pretty-printed, only if the raw text parses as JSON
                   # (eval.py reads these; a missing one = parse failure,
                   #  which eval.py reports as a schema miss)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from constrained import SCHEMA_PATH, build_generator, finalize, load_constraint_schema


HERE = Path(__file__).parent.resolve()
DEFAULT_EVAL_JSONL = HERE / "train" / "data" / "eval.jsonl"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

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


def load_torch_generate(args):
    """Load the adapter (Unsloth) or bare base model (transformers) and
    return generate(prompt_msgs) -> str."""
    if args.no_adapter:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "mps"
        print(f"Device:  {device}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # Load (and cast to fp16) on CPU, then move. device_map="mps" segfaults
        # in torch's MPS cast kernel when transformers loads shards in threads.
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.float16,
        ).to(device)
        model.eval()
    else:
        # Unsloth must be imported before transformers/peft to apply its patches.
        from unsloth import FastLanguageModel
        import torch

        # Unsloth detects the PEFT adapter_config and loads base + adapter.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.adapter),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

    generator = None
    if not args.unconstrained:
        print("Building Outlines schema index (~30s)...")
        generator = build_generator(model, tokenizer)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    def generate(prompt_msgs: list[dict]) -> str:
        input_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        with torch.no_grad():
            if generator is not None:
                # Outlines tokenizes, generates under the schema mask, and
                # returns only the newly generated text.
                return generator(input_text, **gen_kwargs)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, **gen_kwargs)
            return tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

    return generate


def load_gguf_generate(args):
    """Return generate(prompt_msgs) -> str backed by llama.cpp's
    llama-completion, one process per example."""
    if shutil.which("llama-completion") is None:
        sys.exit("ERROR: llama-completion not found on PATH (brew install llama.cpp)")

    workdir = Path(tempfile.mkdtemp(prefix="infer_eval_gguf_"))
    schema_file = workdir / "schema.json"
    schema_file.write_text(json.dumps(load_constraint_schema()))
    prompt_file = workdir / "prompt.txt"

    cmd = [
        "llama-completion", "-m", str(args.gguf), "-f", str(prompt_file),
        "-n", str(args.max_new_tokens),
        "-c", str(args.max_seq_length + args.max_new_tokens),
        "-ngl", "99", "-no-cnv", "--no-display-prompt",
        # Greedy, with the penalty window widened from llama.cpp's 64 tokens
        # to the whole context to match transformers' repetition_penalty.
        "--temp", "0", "--repeat-penalty", str(args.repetition_penalty),
        "--repeat-last-n", "-1",
    ]
    if not args.unconstrained:
        cmd += ["--json-schema-file", str(schema_file)]

    def generate(prompt_msgs: list[dict]) -> str:
        # Qwen ChatML, written out by hand so this path needs no tokenizer.
        # Matches tokenizer.apply_chat_template for system+user turns.
        prompt_file.write_text(
            "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                    for m in prompt_msgs)
            + "<|im_start|>assistant\n"
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"WARN: llama-completion exited {proc.returncode}: "
                  f"{proc.stderr[-300:]!r}", file=sys.stderr)
        # llama-completion appends this marker to stdout when the model stops.
        return proc.stdout.replace(" [end of text]", "").rstrip("\n")

    return generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", type=Path,
                     help="Path to the trained LoRA adapter dir (e.g. "
                          "train/runs/round4/final). Unsloth loads the base "
                          "model named in its adapter_config automatically.")
    src.add_argument("--no-adapter", action="store_true",
                     help="Run the untuned base model (--base-model) in fp16 "
                          "with plain transformers, no Unsloth — the "
                          "before-fine-tuning baseline. Requires --out.")
    src.add_argument("--gguf", type=Path,
                     help="Run a GGUF file through llama.cpp's llama-completion "
                          "(Metal on Apple Silicon), constrained by llama.cpp's "
                          "JSON-schema grammar. Requires --out.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="HF model used with --no-adapter "
                         f"(default: {DEFAULT_BASE_MODEL})")
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
    ap.add_argument("--unconstrained", action="store_true",
                    help="Skip Outlines and use plain model.generate() — the "
                         "pre-constrained-decoding baseline.")
    args = ap.parse_args()

    if args.adapter is None and args.out is None:
        print("ERROR: --no-adapter / --gguf require --out", file=sys.stderr)
        return 1
    if args.gguf is not None and not args.gguf.exists():
        print(f"ERROR: GGUF file {args.gguf} not found", file=sys.stderr)
        return 1
    if args.adapter is not None and not args.adapter.exists():
        print(f"ERROR: adapter dir {args.adapter} not found", file=sys.stderr)
        return 1
    if not args.eval_jsonl.exists():
        print(f"ERROR: eval file {args.eval_jsonl} not found", file=sys.stderr)
        return 1

    out_dir = args.out or (args.adapter.parent / "eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale predictions from prior runs — otherwise eval.py scores a
    # mix of runs (leftover {id}.json files pollute the report).
    stale = list(out_dir.glob("*.json")) + list(out_dir.glob("*.raw.txt"))
    for p in stale:
        p.unlink()
    print(f"Adapter: {args.adapter or (f'(none — GGUF {args.gguf})' if args.gguf else f'(none — base model {args.base_model})')}")
    print(f"Eval:    {args.eval_jsonl}")
    print(f"Out:     {out_dir}  (cleared {len(stale)} stale files)")

    generate = load_gguf_generate(args) if args.gguf else load_torch_generate(args)
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))

    n_done = 0
    n_parse_ok = 0
    n_valid = 0
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
            t0 = time.time()
            gen = generate(prompt_msgs)
            elapsed = time.time() - t0

            (out_dir / f"{example_id}.raw.txt").write_text(gen)
            # Constrained output can still fail to parse if it hit
            # --max-new-tokens mid-object (truncated prefix).
            parsed = extract_first_json(gen)
            valid = False
            if parsed is not None:
                if not args.unconstrained:
                    parsed = finalize(parsed)
                valid = validator.is_valid(parsed)
                n_valid += valid
                (out_dir / f"{example_id}.json").write_text(
                    json.dumps(parsed, indent=2)
                )
                n_parse_ok += 1
            # else: eval.py will flag the missing .json as a schema miss
            n_done += 1
            ok = parsed is not None
            # Inline diagnostic: size + tail makes the failure mode (truncation,
            # repetition loop, trailing junk) visible right in the run log.
            status = "✓ valid " if valid else ("~ parsed" if ok else "✗ FAILED")
            print(f"  [{n_done}] {example_id}: {len(gen):>6}B {elapsed:>4.0f}s "
                  f"{status} | tail: {gen[-90:]!r}", flush=True)

    print(f"\nDone: {n_done} predictions, {n_parse_ok} parsed as JSON, "
          f"{n_valid} schema-valid → {out_dir}")
    print(f"Now score with:  uv run python eval.py --model {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
