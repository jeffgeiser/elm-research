# ELM Research

> ELMs (Expert Language Models) > LLMs for narrow enterprise tasks.

This repo documents my ongoing work building and releasing small, specialized models fine-tuned for specific domains. The thesis: a 4B model trained on the right data beats a 70B general model on a narrow task — and runs locally for free.

## Why ELMs

Most enterprise AI spend goes to frontier API calls for tasks that don't need frontier intelligence. Meeting prep. Account summaries. Structured data extraction. These are synthesis tasks with predictable inputs and outputs. A well-trained small model handles them faster, cheaper, and with better schema adherence than a general model prompted at runtime.

The goal of this repo is to build the methodology for doing that well — dataset design, eval harness, fine-tuning loop, honest release — and document every step publicly.

## Current projects

### Account intelligence (`schemas/account-intelligence/`)

A fine-tuned model that synthesizes Salesforce, Confluence, Slack, and internal APIs into structured JSON account briefs across six enterprise surfaces:

- `meeting_prep` — pre-call brief with must-address items, pipeline, stakeholder map
- `qbr` — quarterly business review pack with trend deltas
- `handoff` — account history and landmines for rep transitions
- `renewal_alert` — risk assessment triggered by approaching renewal date
- `onboarding` — relationship history for new reps
- `escalation` — context pack for CSM during active incidents

**Status:** Dataset generation in progress. Schema locked at v0.1.0. 8 hand-built gold examples across all six surfaces. Synthetic generator running.

**Target release:** `account-intelligence-7b-v1` on Hugging Face — Q4_K_M GGUF, runs on 16GB Apple Silicon, validated against Claude on schema adherence and hallucination rate.

## Methodology

The approach that makes this work is schema-first design with eval-driven iteration:

1. **Lock the output contract first.** `schema.json` defines every field, type, and optionality before any training data is generated. The schema is the product spec.

2. **Build gold examples by hand.** Before generating synthetic data, build 20-50 examples manually from real data. These become the frozen eval set — every checkpoint is scored against them.

3. **Generate synthetic data at scale.** Claude as teacher model, generating input/output pairs that cover the full distribution of account shapes, data sparsity levels, and edge cases.

4. **Eval-driven training loop.** DeepEval + LLM-as-judge harness scores every checkpoint against 8 locked metrics. Schema adherence is the floor gate (must clear 95% before anything else is reported). Hallucination rate is the product story metric.

5. **Middleware as the safety net.** Outlines constrained decoding + retry middleware guarantees valid JSON output at the inference layer regardless of model behavior.

6. **Distill to 7B for release.** Fine-tune 32B as the teacher, distill to 7B for deployment. The 7B model is what ships on Hugging Face.

## Repo structure

```
schemas/          # Output JSON contracts — the product spec per domain
prompts/          # System + user prompt templates for data generation
datasets/         # Generated examples (gitignored — may contain customer data)
models/           # Released model cards and inference configs
docs/             # Methodology, eval specs, data audit, findings
generate.py       # Synthetic data generator — Claude API + schema validation + retry loop
```

## Running the generator

```bash
# Install dependencies
uv sync

# Dry run — no API calls
uv run python generate.py --dry-run --surface qbr --count 5

# Generate examples
uv run python generate.py --surface qbr --count 10 --cost-cap 5.00

# All surfaces
uv run python generate.py --surface meeting_prep --count 20
uv run python generate.py --surface handoff --count 10
uv run python generate.py --surface renewal_alert --count 10
uv run python generate.py --surface onboarding --count 10
uv run python generate.py --surface escalation --count 10
```

Requires `ANTHROPIC_API_KEY` in a `.env` file at the repo root.

## Eval metrics

Eight metrics with locked thresholds at v0.1 / v0.5 / v1.0. Full spec in `docs/account-intelligence/evaluation.md`.

| Metric | Gate | Why it matters |
|--------|------|----------------|
| Schema adherence | Floor — 95% before others reported | Structural correctness |
| Hallucination rate | Product story metric | Trust |
| Section coverage | Secondary | Completeness |
| Empty sections accuracy | Secondary | Honesty about data gaps |
| Source attribution rate | Secondary | Grounding |
| Action specificity | Secondary | Usefulness |
| Confidence calibration | Secondary | Reliability |
| Surface-section alignment | Secondary | Correctness |

## Hardware

Fine-tuning runs on a DGX Spark (64GB VRAM) using Unsloth + QLoRA. Dataset generation runs on an Apple M4 Mac Mini (16GB unified memory). Released models target Q4_K_M GGUF for 16GB deployment.

## Background

I run a production agentic system (Zenlayer Central) that already uses a local Qwen2.5-32B alongside Claude for enterprise synthesis tasks. 22 paired eval runs showed local was faster but Claude was better on quality. This repo is the work to close that gap through fine-tuning.

Related writing:
- [Substack](https://substack.com/@jeffgeiser) — methodology posts, benchmark findings, model releases
- Llama 3.1 8B quantization benchmark — what Q8_0 on 16GB actually does (and why it's not swap)

## Roadmap

- [ ] Account intelligence dataset — 568 synthetic examples across 6 surfaces
- [ ] DeepEval + LLM-as-judge eval harness
- [ ] LoRA fine-tune Qwen2.5-32B baseline
- [ ] JSON middleware — Outlines constrained decoding
- [ ] Distill to 7B
- [ ] Release `account-intelligence-7b-v1` on Hugging Face
- [ ] Quantization advisor ELM — Apple Silicon scoped v0.1

## License

Apache 2.0
