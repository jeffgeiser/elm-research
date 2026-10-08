#!/usr/bin/env bash
# Merge a trained LoRA adapter into the fp16 base model, convert to GGUF,
# and quantize to Q4_K_M (the 16GB-Apple-Silicon release target).
#
# Usage (on the training box, after `uv sync`):
#     ./train/export_gguf.sh round10
#
# Reads   train/runs/<run>/final            (the adapter)
# Writes  train/runs/<run>/merged/          (fp16 merged HF model, ~15GB)
#         train/runs/<run>/gguf/account-intelligence-7b-<run>.f16.gguf
#         train/runs/<run>/gguf/account-intelligence-7b-<run>.Q4_K_M.gguf
#
# The merge runs on CPU (needs ~32GB RAM), so it can run alongside a GPU job.
# Clones and builds llama.cpp in ~/llama.cpp on first use.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run-name>   (e.g. round10)"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
RUN_NAME="$1"
RUN="$HERE/runs/$RUN_NAME"
OUT="$RUN/gguf"
NAME="account-intelligence-7b-$RUN_NAME"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_DIR="$HOME/llama.cpp"
CONVERT_VENV="$HOME/gguf-venv"

if [[ ! -f "$RUN/final/adapter_config.json" ]]; then
    echo "ERROR: $RUN/final/adapter_config.json not found."
    exit 1
fi
mkdir -p "$OUT"

echo "== [1/4] llama.cpp (quantizer + converter deps)"
[[ -d "$LLAMA_DIR" ]] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release > "$LLAMA_DIR/build.log" 2>&1
cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)" >> "$LLAMA_DIR/build.log" 2>&1
# The converter needs llama.cpp's own gguf package; keep it out of the project venv.
[[ -d "$CONVERT_VENV" ]] || uv venv -q "$CONVERT_VENV" --python 3.11
VIRTUAL_ENV="$CONVERT_VENV" uv pip install -q \
    -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

echo "== [2/4] merge adapter into fp16 base (CPU)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="" uv run python - "$BASE_MODEL" "$RUN" <<'PY'
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model, run = sys.argv[1], sys.argv[2]
# The adapter was trained on the bnb-4bit build; merge into the full-precision
# base so quantization happens once, in llama.cpp.
base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16)
model = PeftModel.from_pretrained(base, f"{run}/final").merge_and_unload()
model.save_pretrained(f"{run}/merged", safe_serialization=True)
AutoTokenizer.from_pretrained(f"{run}/final").save_pretrained(f"{run}/merged")
print(f"merged -> {run}/merged")
PY

echo "== [3/4] convert to GGUF f16"
"$CONVERT_VENV/bin/python" "$LLAMA_DIR/convert_hf_to_gguf.py" "$RUN/merged" \
    --outtype f16 --outfile "$OUT/$NAME.f16.gguf" 2>&1 | tail -3

echo "== [4/4] quantize Q4_K_M"
"$LLAMA_DIR/build/bin/llama-quantize" "$OUT/$NAME.f16.gguf" "$OUT/$NAME.Q4_K_M.gguf" Q4_K_M 2>&1 | tail -3

ls -la "$OUT"
echo "Done: $OUT/$NAME.Q4_K_M.gguf"
