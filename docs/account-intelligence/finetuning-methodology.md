# Fine-Tuning Methodology — Account Intelligence ELM

> **Purpose of this document.** A complete, self-contained handoff of how we
> fine-tune a small Expert Language Model (ELM) for the account-intelligence
> task, written so another engineer or model can reconstruct, run, or extend
> the training pipeline. It captures the *why* behind each choice, the exact
> configuration, the data flow, the eval harness, and the failure modes we
> already hit and fixed. Read the "Known issues & open decisions" section
> before building on this — there is one architectural problem (input framing)
> that is not yet resolved.

---

## 1. Thesis and goal

**ELMs > LLMs for narrow enterprise tasks.** A small model (7B) fine-tuned on
the right data should beat a frontier general model *prompted at runtime* on a
narrow, schema-bound synthesis task — faster, cheaper, locally hosted, with
better schema adherence.

**Target product:** `account-intelligence-7b-v1` — a model that synthesizes
account data into structured JSON "briefs" across six enterprise surfaces.
Distilled/quantized to Q4_K_M GGUF for 16GB Apple Silicon deployment.

**The six surfaces** (each has its own expected section set — see the schema):
`meeting_prep`, `qbr`, `handoff`, `renewal_alert`, `onboarding`, `escalation`.

**Success bar:** schema adherence ≥ 95% (the floor gate), then low
hallucination, good section coverage, sourced claims. See §6.

---

## 2. The methodology in one paragraph

Lock the output contract first (a strict JSON Schema). Build a handful of gold
examples by hand. Use a frontier model (Claude) as a teacher to generate
synthetic input/output pairs at scale that cover the full distribution of
account shapes, data-sparsity levels, and edge cases. Freeze a held-out eval
split. QLoRA-fine-tune a 7B base on the pairs with **completion-only loss**.
Score every checkpoint against locked metrics, with schema adherence as a hard
gate. Guarantee valid JSON at inference with constrained decoding as a safety
net. Then quantize for release.

---

## 3. Hardware & environment (and its sharp edges)

| Component | Value | Notes |
|---|---|---|
| Training box | NVIDIA **DGX Spark**, GB10 Grace-Blackwell | **Unified memory** — GPU shares system RAM |
| Memory | **121 GB** unified | `nvidia-smi --query-gpu=memory.*` returns **N/A**; use `free -h` instead |
| CUDA capability | **12.1** (Blackwell) | Newer than this PyTorch build officially supports (max 12.0) — warns but runs |
| Torch / CUDA | torch 2.10.0+cu128, CUDA toolkit 12.8 | Installed from the `pytorch-cu128` index (see `pyproject.toml`) |
| Key libs | unsloth 2026.5.7, transformers 5.5.0, trl 0.24.0, peft, bitsandbytes, accelerate | Pinned in `pyproject.toml` |
| Package manager | **uv** | Always run via `uv run python …` so the `.venv` is active |
| Data-gen box | Apple M4 Mac Mini, 16GB | Runs `generate.py` (Claude API), not training |

### Sharp edges that shaped the design
- **Flash Attention 2 does not work on the GB10** in this build. Unsloth falls
  back (`FA [Xformers = None. FA2 = False]`) to **eager / naive O(n²)
  attention**. This is the single most important performance fact: attention
  cost is quadratic in sequence length, so long contexts are brutally slow.
  *Fixing this (a working Blackwell attention kernel) is the real lever for
  going back to longer sequences.*
- **Unified memory** means "VRAM" and system RAM are the same pool. A
  co-resident vLLM server + training run can exceed the pool and hard-crash the
  box (this happened — "round 4 OOM"). Kill other GPU users before training.
- **No co-tenancy assumption.** Training assumes sole-tenant use of the GPU.

---

## 4. The stack

- **Base model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (4-bit pre-quantized).
- **Method:** QLoRA (4-bit frozen base + trainable LoRA adapters) via **Unsloth**
  `FastLanguageModel` for the patched kernels + memory savings.
- **Trainer:** TRL **`SFTTrainer`** with **`SFTConfig`** (not bare
  `TrainingArguments` — see the gotcha in §7).
- **Chat template:** Qwen2.5 ChatML (`<|im_start|>role\n…<|im_end|>`).

---

## 5. Data pipeline

```
schemas/account-intelligence/schema.json   # THE contract (strict)
prompts/account-intelligence/              # generator system + user prompts
generate.py                                # Claude teacher → synthetic examples
datasets/account-intelligence/             # example-NNN.json gold/synthetic (~572)
  └─ eval_split.txt                        # held-out IDs (50) — frozen eval set
train/format_jsonl.py                      # examples → train.jsonl / eval.jsonl
train/data/{train.jsonl, eval.jsonl}       # chat-format training records
```

### 5.1 Schema-first (`schema.json`)
- The schema is the **product spec**. It defines every field, type, and
  optionality before any data is generated.
- It is **strict**: `additionalProperties: false` at the object levels. Any
  field the model emits that isn't in the schema → **schema-invalid**. (This is
  deliberate and is what makes M1 meaningful — but it also means the model must
  learn *exact* section membership; see the cross-section-leak failure in §9.)
- Each surface has an expected set of sections (encoded in `eval.py`
  `SURFACE_SECTIONS`). Claim-bearing sections carry a `sources[]` array per
  item (encoded in `eval.py` `CLAIM_PATHS`).

### 5.2 Synthetic generation (`generate.py`)
- Claude is the **teacher model**. Given a system prompt (the "synthetic example
  generator" prompt) and a slot-filled user prompt with knobs
  (`surface`, `account_shape`, `edge_cases`, `source_mix`), it emits one
  schema-valid brief.
- Output is JSON-validated against the schema with a retry loop; there is a
  cost cap (`--cost-cap`) and a `--dry-run`.
- Each generated `example-NNN.json` carries a top-level **`_meta`** block:
  generator bookkeeping (`synthetic: true`, `shape_constraints`,
  `edge_cases_included`, `surface`, …). **`_meta` is annotation, not contract.**

### 5.3 Formatting to training records (`format_jsonl.py`)
For each `example-NNN.json`, `build_record` produces a 3-message chat record:
- **system** = the generator's system prompt (verbatim, ~25KB).
- **user** = `reconstruct_user_prompt(meta, surface)` — rebuilds the
  knob-based generation request ("Generate one synthetic example with these
  parameters: surface, account_shape, edge_cases, source_mix").
- **assistant** = the brief as a JSON string **with `_meta` stripped**
  (`assistant_doc = {k: v for k, v in doc.items() if k != "_meta"}`).

Records whose ID is in `eval_split.txt` go to `eval.jsonl`; the rest to
`train.jsonl`. Split is roughly 520 train / 50 eval.

> ⚠️ **This input framing is the open architectural problem.** See §9.1. The
> model is currently trained on `(generator prompt + knobs) → brief`, i.e. it
> learns to *impersonate the data generator*, not to ingest real account data.

---

## 6. Eval harness (`eval.py`) — eight metrics, schema is the gate

`eval.py` **scores** a directory of pre-generated prediction files
(`<eval_id>.json`) against the held-out split. It does **not** run the model —
generation is a separate step (see §8). Run with `uv run python eval.py
--model <dir>` (or `--gold` to score the gold set as a Claude baseline).

| # | Metric | v0.1 gate | Automated? | Notes |
|---|---|---|---|---|
| M1 | **Schema adherence** | **≥ 95% (FLOOR)** | ✅ | jsonschema Draft2020-12. Hard gate — report nothing else until cleared |
| M2 | Hallucination rate | product-story metric | ❌ | needs the input source data |
| M3 | Section coverage | ≥ 90% | ✅ | populated-or-declared-empty over the surface's relevant sections |
| M4 | Empty-sections accuracy | secondary | ❌ | needs input source data |
| M5 | Source attribution rate | ≥ 80% | ✅ | fraction of claim items carrying a non-empty `sources[]` |
| M6 | Confidence calibration | secondary | ❌ | derived from M2 |
| M7 | Latency | secondary | ❌ | measured at inference time |
| M8 | Style | secondary | ❌ | manual review |

The report card breaks down per-surface and overall, and lists schema-failure
details. Full metric spec: `docs/account-intelligence/evaluation.md`.

---

## 7. Training script (`train/train_lora.py`)

Run via `train/run.sh <run-name> [extra args…]` (wraps it in tmux + a log file
so it survives SSH drops; forwards extra args to the script).

### 7.1 Hyperparameters (locked in-file, versioned with the run)
```
BASE_MODEL        = unsloth/Qwen2.5-7B-Instruct-bnb-4bit
MAX_SEQ_LENGTH    = 8192          # see §7.4 — was 16384
LORA_RANK         = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.0           # Unsloth-recommended for speed
TARGET_MODULES    = q,k,v,o,gate,up,down  (all attention + MLP projections)
LEARNING_RATE     = 1e-4          # dropped from 2e-4 after round 1 over-memorized
PER_DEVICE_BATCH  = 1
GRAD_ACCUM_STEPS  = 16            # effective batch = 16
NUM_EPOCHS        = 1             # ~33 optimizer steps over ~520 examples
WARMUP_STEPS      = 5
WEIGHT_DECAY      = 0.01
LR_SCHEDULER      = cosine
OPTIM             = adamw_8bit
PRECISION         = bf16
SAVE_STEPS = EVAL_STEPS = 10 ; LOGGING_STEPS = 1
```

### 7.2 Completion-only loss (the `CompletionOnlyCollator`)
Loss must be computed **only on the assistant turn**, not the prompt. Without
this, the model learns to regurgitate the (huge) system prompt and collapses
into repetition (observed in round 1).

Implementation detail that matters: the collator masks the prefix using
**Qwen2.5 special token IDs directly** (`<|im_start|>` id + the BPE pieces of
`"assistant"` + the following newline), **not** by string-matching a rendered
template. Rationale: the BPE tokenization of a standalone template string can
differ from how those tokens appear mid-sequence after `apply_chat_template`,
which silently broke masking before.

The collator **self-tests on init** against a synthetic chat sequence and
**refuses to construct** if masking would mask everything or nothing — this is
a guardrail against the round-2 failure (`train_loss=0` because everything was
masked). It also dumps the first real batch once for inspection.

### 7.3 The `SFTConfig` gotcha (do not regress this)
`max_seq_length` **must** be passed via `SFTConfig`, not to `SFTTrainer` or
`TrainingArguments`. In TRL 0.20+, a `max_seq_length` passed elsewhere is
**silently ignored** and defaults to ~1024, truncating every long record. This
cost us "round 3" (everything quietly truncated to 1024). `dataset_text_field`
is `"text"` (we pre-apply the chat template into a `text` column because
auto-detection from `messages` is unreliable across TRL/Unsloth versions);
`packing=False`.

### 7.4 Sequence length = 8192 (a deliberate trade-off)
The dataset has records up to ~13.4K tokens (p95 ~12K). Ideally seq_len would
cover that (16384). **But** with no working flash attention on the GB10 (§3),
16384 makes attention quadratically slow — eval inference became unusable and
training crawled. We run at **8192**, accepting that the longest records are
**truncated**. Because this is completion-only training with the JSON answer at
the end, truncation can cut off part of the target for the longest accounts —
a real quality cost on those examples. **The correct long-term fix is a working
Blackwell attention kernel, not more memory** (there's plenty: 11.6GB used of
121GB at 8192/batch-1).

### 7.5 Callbacks
- **`VramLogCallback`** — logs peak alloc/reserved/free every N steps to
  `vram.log`. (Confirms headroom; we run at ~11.6GB peak.)
- **`SanityCheckCallback`** — on each save, generates output for **exactly 3**
  eval examples into `<checkpoint>/sanity_outputs/`. **No scoring, no parsing,
  no eval.py.** Just a "model isn't broken" smoke test (minutes, not hours).
  - *This replaced a `PostCheckpointEvalCallback` that ran full generation on
    all 50 eval examples at every checkpoint and wedged training for 8+ hours.*
    **Do not put heavy generation/eval inside the training loop.** Full eval is
    a separate manual step after training.

### 7.6 Resume
Resume support is `--resume-from-checkpoint <dir>`, routed to
`trainer.train(resume_from_checkpoint=…)`. **Note:** the
`resume_from_checkpoint` field on `SFTConfig`/`TrainingArguments` is **not read
by the Trainer** (HF docs: "not directly used by Trainer") — it only works when
passed to `train()`. The checkpoint dir must contain optimizer/scheduler/
`trainer_state.json`, not just the adapter, or step state won't restore.

### 7.7 GPU preflight
`preflight_gpu_check` refuses to start unless ≥60GB is free (via
`torch.cuda.mem_get_info`, which *does* work on unified memory even though the
nvidia-smi CSV query returns N/A). `--force` overrides — don't, it's what the
OOM crash was about.

---

## 8. Inference for eval (`infer_eval.py`)

Because generation was deliberately removed from the training loop (§7.5),
there is a standalone script to produce predictions for the held-out set:

```bash
uv run python infer_eval.py --adapter train/runs/<run>/final \
    --out train/runs/<run>/eval_outputs [--limit N]
uv run python eval.py --model train/runs/<run>/eval_outputs
```

It loads the finished adapter (Unsloth resolves base + adapter from
`adapter_config`), generates a brief per eval record, and writes
`<id>.raw.txt` (always) and `<id>.json` (if parseable). Key robustness details
learned the hard way:
- **Greedy decoding degenerates** into repetition loops (`"0000…"`,
  `"sf_activity"…`). Mitigated with `--repetition-penalty` (default 1.15).
- **The model appends a trailing `{"_meta": …}` object**, so a naive
  `json.loads` of the whole string dies with "Extra data". We use
  `JSONDecoder().raw_decode` to take the **first** JSON object, tolerating
  trailing junk and leading prose / ```json fences.
- `--limit N` does a quick subset read (schema signal in ~15 min) before
  committing to the full 50.

---

## 9. Known issues & open decisions (READ BEFORE EXTENDING)

### 9.1 ⚠️ The model is a clone of the *generator*, not an inference model
This is the most important open problem. Training pairs are
`(generator system prompt + generation knobs) → brief`. There is **no real
account-data input** in the training records — the synthetic generator invented
the account *and* its evidence from knobs in one shot, so the only "input" is
the knobs. Consequences observed in eval:
- The model **invents accounts** from knobs (same fake names recur under greedy
  decoding because there's no account in the 200-char user prompt).
- It **leaks generator vocabulary** (`_meta`, `synthetic`, `shape_constraints`)
  into output even though `_meta` is stripped from targets — because that
  vocabulary lives in the 25KB *system* prompt it sees every example.

**Two paths:**
- **Path A (cheap, closed-loop):** keep the generator-clone framing; lean on
  decoding fixes + constrained decoding (Outlines) to force schema-valid JSON;
  report v0.1 on "reproduce the teacher's brief from knobs."
- **Path B (correct for production):** change `generate.py` to emit **both** a
  raw source bundle (the CRM rows, Slack/Teams messages, news, KB snippets) and
  the brief. Then `format_jsonl.py` pairs **`source bundle → brief`** with a
  production *inference* system prompt (not the generator prompt). This is what
  makes the model usable on live accounts. Requires regen + reformat + retrain.

The README's production vision ("synthesize Salesforce/Slack/Confluence into
briefs") requires **Path B**. Decide this before investing more training.

### 9.2 Strict schema + cross-section leaks
With `additionalProperties: false`, the model fails M1 if it places a *valid*
field in the *wrong* section (observed: `open_in_meeting`/`watch_for` leaking
from `must_address` items into `pipeline` items). Constrained decoding
(Outlines) is the planned structural fix; more training may reduce it.

### 9.3 Truncation at 8192 (see §7.4) and the Blackwell attention kernel.

### 9.4 Output length / termination
Long surfaces (escalation, meeting_prep) produce very large briefs that can hit
the token cap or fail to terminate cleanly. Repetition penalty helps; a working
schema-constrained decoder would help more.

---

## 10. Run history (what each round taught us)

| Round | Symptom | Root cause / fix |
|---|---|---|
| 1 | Model regurgitated the prompt, repetition loops | LR too high (2e-4) **and** no completion masking → loss on prompt. Fix: completion-only collator + LR 1e-4 |
| 2 | `train_loss = 0` | Collator masked **everything** (string-template token mismatch). Fix: rewrite collator to use Qwen2.5 token IDs + self-test guard |
| 3 | Quietly truncated to 1024 | `max_seq_length` passed to the wrong place; ignored. Fix: pass via **`SFTConfig`** |
| 4 | OOM hard-crash; then 8h-per-checkpoint stall | vLLM co-resident in unified memory (added preflight); eval callback ran full 50-example generation inline (replaced with 3-example SanityCheck); seq 16384 too slow on eager attention (dropped to 8192) |
| (post-4 eval) | parse rate 1/8 | trailing `_meta` object + greedy repetition (fixed in `infer_eval.py`); + the §9.1 framing issue (open) |

---

## 11. End-to-end runbook

```bash
# 0. Environment (DGX Spark, sole-tenant GPU)
pkill -9 -f vllm                 # free the unified-memory pool
free -h                          # confirm ~60GB+ available ("VRAM" == RAM here)

# 1. (If data changed) regenerate training records
uv run python train/format_jsonl.py        # → train/data/{train,eval}.jsonl

# 2. Train (tmux-wrapped, survives disconnect)
./train/run.sh round5
#   resume:  ./train/run.sh round5 --resume-from-checkpoint train/runs/<r>/checkpoint-N
tail -f train/runs/round5/train.log
#   healthy log: [preflight] passes ; [collator] self-test OK ; loss descending

# 3. Sanity eyeball (already produced by SanityCheckCallback)
ls train/runs/round5/checkpoint-*/sanity_outputs/

# 4. Full manual eval (generation is OUTSIDE the training loop)
uv run python infer_eval.py --adapter train/runs/round5/final \
    --out train/runs/round5/eval_outputs
uv run python eval.py --model train/runs/round5/eval_outputs
#   M1 (schema) must clear 95% before reporting anything else
```

---

## 12. Roadmap context

- [ ] Resolve §9.1 (Path A vs B) — **gating decision**
- [ ] Outlines constrained decoding + retry middleware (schema-valid output guarantee)
- [ ] Fine-tune Qwen2.5-32B teacher → distill to 7B
- [ ] Quantize to Q4_K_M GGUF; validate on 16GB Apple Silicon
- [ ] Release `account-intelligence-7b-v1` on Hugging Face
- [ ] (Infra) working flash-attention kernel for GB10/Blackwell → unlock 16384 seq_len
```
