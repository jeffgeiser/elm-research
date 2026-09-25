#!/usr/bin/env python3
"""Synthetic example generator for the account intelligence model dataset.

Walks a generation matrix (one row per example), calls the model with the
system prompt + per-row user prompt, validates each output against
schema.json, retries with validation errors fed back on failure. Writes
successful examples to dataset/example-NNN.json (gitignored).

Usage:
    python generate.py --surface qbr --count 5
    python generate.py --surface handoff --shape mid_market_active --count 3
    python generate.py --surface qbr --count 50 --start-id 9 --cost-cap 20
    python generate.py --dry-run --surface qbr --count 5

Env:
    GATEWAY_API_KEY     API key for the gateway (preferred).
    GATEWAY_BASE_URL    Base URL for the gateway (default: https://gateway.theturbo.ai/v1).
    ANTHROPIC_API_KEY   Fallback if GATEWAY_API_KEY is not set (uses Anthropic SDK directly).

The generator uses the OpenAI-compatible /v1/chat/completions interface so it
works with any gateway or provider that speaks that format (Anthropic, OpenAI,
Qwen, etc.). Pass --model to select the model; default is claude-sonnet-4-5-20250929.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from jsonschema import Draft202012Validator


# ----- paths --------------------------------------------------------------

HERE = Path(__file__).parent.resolve()
ROOT = HERE
SYSTEM_PROMPT_PATH = HERE / "prompts" / "account-intelligence" / "system-prompt.md"
SOURCE_BUNDLE_PROMPT_PATH = HERE / "prompts" / "account-intelligence" / "source-bundle-generator.md"
INFERENCE_PROMPT_PATH = HERE / "prompts" / "account-intelligence" / "inference-system-prompt.md"
SCHEMA_PATH = HERE / "schemas" / "account-intelligence" / "schema.json"
DATASET_DIR = HERE / "datasets" / "account-intelligence"


# ----- defaults -----------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_BASE_URL = "https://gateway.theturbo.ai/v1"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_TOKENS_PATH_B = 24000  # source_bundle + brief is ~2x a brief alone
DEFAULT_TEMP = 0.6
DEFAULT_RETRIES = 5
DEFAULT_RETRIES_PATH_B = 8  # more iterations to converge complex dual output
DEFAULT_COST_CAP_USD = 20.0

# Sonnet pricing — used only for the up-front cost estimate, not billing.
COST_INPUT_PER_M_USD = 3.0
COST_OUTPUT_PER_M_USD = 15.0
# Rough average per example based on dry-runs. Input includes the full
# system prompt (~5k tokens) every call.
EST_INPUT_TOKENS = 6000
EST_OUTPUT_TOKENS = 6000
# Empirical retry factor — ~30% of examples need one retry from the
# 3-attempt dry-runs. Used only for cost estimation.
EST_RETRY_FACTOR = 1.3


# ----- matrix-row options -------------------------------------------------

SURFACES = (
    "meeting_prep",
    "qbr",
    "handoff",
    "renewal_alert",
    "onboarding",
    "escalation",
)

ACCOUNT_SHAPES = (
    "enterprise_steady",
    "enterprise_expanding",
    "mid_market_active",
    "mid_market_quiet",
    "small_strategic",
    "stalled_at_risk",
    "fresh_handoff",
    "incident_acute",
)

EDGE_CASES = (
    "subsidiary_explosion",
    "null_industry",
    "mis_categorized_industry",
    "multi_currency_amount",
    "sparse_activity",
    "activity_overlogged",
    "bot_filed_case_noise",
    "p1_open_long",
    "closed_lost_recent",
    "champion_churn",
    "non_latin_contact",
    "duplicate_contacts",
    "dc_code_gap",
    "null_priority_cases",
    "audit_cycle",
    "public_announcement_anchor",
    "utilization_drop",
    "opp_name_typo",
    "vp_newly_visible",
)

SOURCE_MIXES = (
    "sf_only",
    "with_news",
    "with_teams",
    "with_prior_prep",
    "rich",
    "minimal",
)


# ----- types --------------------------------------------------------------


@dataclass
class GenParams:
    """One row of the generation matrix — params for one Claude call."""

    surface: str
    account_shape: str
    edge_cases: list[str] = field(default_factory=list)
    source_mix: str | None = None
    account_hint: str | None = None

    def format_user_prompt(self) -> str:
        """Render as the slot-filled user prompt per user-prompt-template.md."""
        lines = [
            "Generate one synthetic example with these parameters:",
            "",
            f"- surface: {self.surface}",
            f"- account_shape: {self.account_shape}",
        ]
        if self.edge_cases:
            lines.append(f"- edge_cases: {json.dumps(self.edge_cases)}")
        if self.source_mix:
            lines.append(f"- source_mix: {self.source_mix}")
        if self.account_hint:
            lines.append(f"- account_hint: {self.account_hint}")
        return "\n".join(lines)


@dataclass
class GenResult:
    """One generation attempt's outcome."""

    doc: dict | None
    attempts: int
    errors_log: list[str] = field(default_factory=list)
    last_raw_text: str | None = None
    elapsed_s: float = 0.0


# ----- matrix building ----------------------------------------------------


def build_matrix(
    *,
    surface: str,
    count: int,
    shape: str | None,
    edge_cases: list[str] | None,
    source_mix: str | None,
    account_hint: str | None,
    seed: int,
) -> list[GenParams]:
    """Build a generation matrix. Unset slots get sampled from the option
    lists; deliberately weighted so most examples have 1-2 edge cases."""
    rng = random.Random(seed)
    rows: list[GenParams] = []
    for _ in range(count):
        s = shape or rng.choice(ACCOUNT_SHAPES)
        if edge_cases is not None:
            ec = list(edge_cases)
        else:
            n_ec = rng.choices([0, 1, 2, 3], weights=[2, 4, 3, 1])[0]
            ec = rng.sample(EDGE_CASES, n_ec)
        # `None` source_mix means "let Claude pick a realistic one"
        sm: str | None
        if source_mix is not None:
            sm = source_mix
        else:
            sm = rng.choice(list(SOURCE_MIXES) + [None])
        rows.append(
            GenParams(
                surface=surface,
                account_shape=s,
                edge_cases=ec,
                source_mix=sm,
                account_hint=account_hint,
            )
        )
    return rows


# ----- JSON extraction ----------------------------------------------------


_FENCED_RE = re.compile(r"```json\s*(.+?)```", re.DOTALL)


def extract_json_block(text: str) -> str | None:
    """Pull the ```json ... ``` block. Falls back to brace-matched first
    object if no fenced block is found (covers occasional output drift)."""
    m = _FENCED_RE.search(text)
    if m:
        return m.group(1).strip()
    if "{" not in text:
        return None
    start = text.index("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ----- one-example generation ---------------------------------------------


_PATH_B_STRUCTURE_REMINDER = """\
Your output must be a single JSON object with this top-level structure:
{
  "_meta": { "account": "...", "synthetic": true, "surface": "...", "shape_constraints": "...", "edge_cases_included": [...] },
  "source_bundle": { "surface": "...", "pulled_at": "...", "sf_account": {...}, "sf_opportunities": [...], ... },
  "brief": { "account": {...}, "surface": "...", "generated_at": "...", "executive_summary": "...", ... }
}
The brief schema is strict — use ONLY defined field names. Keep the source_bundle intact."""


def _validate_doc(doc: dict, validator: Draft202012Validator, mode: str) -> list:
    """Return schema errors. In path-b mode validate doc['brief']; in path-a
    validate the top-level doc (which IS the brief)."""
    if mode == "path-b":
        if "brief" not in doc or "source_bundle" not in doc:
            class _FakeError:
                absolute_path = []
                message = (
                    "Output is missing required wrapper keys. "
                    + _PATH_B_STRUCTURE_REMINDER
                )
            return [_FakeError()]
        return list(validator.iter_errors(doc["brief"]))
    return list(validator.iter_errors(doc))


def generate_one(
    *,
    client: OpenAI,
    system_prompt: str,
    validator: Draft202012Validator,
    params: GenParams,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    mode: str = "path-b",
) -> GenResult:
    """Call the model via OpenAI-compatible API with retry-on-validation-failure.
    Feeds schema errors back into the conversation as a corrective user turn.

    Returns a GenResult with `doc` populated on success or `None` on
    failure-after-retries; the errors_log captures each attempt's
    diagnostics; last_raw_text holds the final output for debugging
    failed runs.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": params.format_user_prompt()},
    ]
    result = GenResult(doc=None, attempts=0)
    t0 = time.time()

    for attempt in range(retries):
        result.attempts = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors_log.append(f"attempt {attempt + 1}: API call failed: {exc}")
            break

        text = resp.choices[0].message.content if resp.choices else ""
        result.last_raw_text = text

        body = extract_json_block(text)
        if body is None:
            result.errors_log.append(
                f"attempt {attempt + 1}: no JSON block in response"
            )
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "Your output didn't contain a ```json``` code block. "
                    "Emit ONLY the brief wrapped in a single fenced "
                    "json code block — no prose before or after."
                )},
            ]
            continue

        try:
            doc = json.loads(body)
        except json.JSONDecodeError as exc:
            result.errors_log.append(
                f"attempt {attempt + 1}: JSON parse error: {exc}"
            )
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    f"Your JSON failed to parse: {exc}. "
                    "Regenerate with valid JSON syntax."
                )},
            ]
            continue

        errors = _validate_doc(doc, validator, mode)
        if not errors:
            result.doc = doc
            result.elapsed_s = time.time() - t0
            return result

        # Validation errors — feed them back for the next attempt.
        err_summary = "\n".join(
            f"- at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors[:15]
        )
        result.errors_log.append(
            f"attempt {attempt + 1}: {len(errors)} schema errors\n"
            + "\n".join(
                f"  - at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in errors[:15]
            )
        )
        if mode == "path-b":
            structure_hint = (
                "\n\nRemember: output must be "
                "`{\"_meta\": {...}, \"source_bundle\": {...}, \"brief\": {...}}`. "
                "Fix only the `brief` schema errors; keep source_bundle unchanged."
            )
        else:
            structure_hint = ""
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                f"Schema validation found {len(errors)} error(s) in the brief:"
                f"\n\n{err_summary}"
                f"\n\nFix these errors only — use ONLY defined field names "
                f"and exact enum values. Brief content stays the same."
                f"{structure_hint}"
            )},
        ]

    result.elapsed_s = time.time() - t0
    return result


# ----- Path B: two-call generation ----------------------------------------


def generate_source_bundle(
    *,
    client: OpenAI,
    source_bundle_prompt: str,
    params: GenParams,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict | None:
    """Call 1 of 2 for path-b: generate a realistic source bundle from params.
    No schema validation — just parse the JSON. Returns None on failure."""
    messages = [
        {"role": "system", "content": source_bundle_prompt},
        {"role": "user", "content": params.format_user_prompt()},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature, messages=messages,
        )
    except Exception as exc:
        print(f"    [bundle] API error: {exc}")
        return None
    text = resp.choices[0].message.content if resp.choices else ""
    body = extract_json_block(text)
    if body is None:
        print(f"    [bundle] no JSON block in response")
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        print(f"    [bundle] JSON parse error: {exc}")
        return None


def generate_one_path_b(
    *,
    client: OpenAI,
    source_bundle_prompt: str,
    inference_prompt: str,
    validator: Draft202012Validator,
    params: GenParams,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int,
) -> GenResult:
    """Path B two-call generation:
    1. Generate source bundle (no schema validation — just JSON).
    2. Feed bundle into the inference system prompt and generate the brief
       with the retry-on-schema-error loop. This IS the production task.

    Separating the calls means schema errors only relate to the brief,
    the source bundle is fixed across retries, and the model has a clear
    focused task on each call.
    """
    import time as _time
    t0 = _time.time()
    result = GenResult(doc=None, attempts=0)

    # ---- Call 1: source bundle ----
    bundle = generate_source_bundle(
        client=client,
        source_bundle_prompt=source_bundle_prompt,
        params=params,
        model=model,
        max_tokens=max_tokens // 2,  # bundle is ~half the total tokens
        temperature=temperature,
    )
    if bundle is None:
        result.errors_log.append("call 1 (source bundle) failed")
        result.elapsed_s = _time.time() - t0
        return result

    # ---- Call 2: brief from bundle (retry loop) ----
    # Format the bundle as the user message — same as format_jsonl.py does.
    from train.format_jsonl import format_source_bundle
    bundle_user_msg = format_source_bundle(bundle)

    # The inference prompt is also the training-time system prompt, so it stays
    # concise and schema-free. The generator model has never seen the schema,
    # though — append it here (generation only) so it knows the exact shape.
    schema_json = json.dumps(validator.schema, indent=2)
    generation_prompt = (
        f"{inference_prompt}\n\n"
        "## Brief schema (authoritative)\n\n"
        "The brief MUST validate against this JSON Schema. Use exactly these "
        "field names, nesting, types, and enum values; do not emit `_meta`.\n\n"
        f"```json\n{schema_json}\n```\n"
    )
    messages: list[dict] = [
        {"role": "system", "content": generation_prompt},
        {"role": "user", "content": bundle_user_msg},
    ]

    for attempt in range(retries):
        result.attempts = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens // 2,
                temperature=temperature,
                messages=messages,
            )
        except Exception as exc:
            result.errors_log.append(f"attempt {attempt + 1}: API error: {exc}")
            break

        text = resp.choices[0].message.content if resp.choices else ""
        result.last_raw_text = text
        body = extract_json_block(text)

        if body is None:
            result.errors_log.append(f"attempt {attempt + 1}: no JSON block")
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "Your output didn't contain a ```json``` code block. "
                    "Emit ONLY the brief as a single fenced json block."
                )},
            ]
            continue

        try:
            brief = json.loads(body)
        except json.JSONDecodeError as exc:
            result.errors_log.append(f"attempt {attempt + 1}: JSON parse error: {exc}")
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"JSON parse error: {exc}. Fix the syntax."},
            ]
            continue

        errors = list(validator.iter_errors(brief))
        if not errors:
            result.doc = {
                "_meta": {
                    "account": bundle.get("sf_account", {}).get("name", "unknown"),
                    "synthetic": True,
                    "surface": bundle.get("surface"),
                    "shape_constraints": params.account_shape,
                    "edge_cases_included": params.edge_cases,
                },
                "source_bundle": bundle,
                "brief": brief,
            }
            result.elapsed_s = _time.time() - t0
            return result

        err_summary = "\n".join(
            f"- at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors[:10]
        )
        result.errors_log.append(
            f"attempt {attempt + 1}: {len(errors)} schema errors\n"
            + "\n".join(f"  {l}" for l in err_summary.splitlines())
        )
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                f"The brief had {len(errors)} schema error(s):\n\n{err_summary}\n\n"
                "Emit the brief again with these fixed. Schema is strict — "
                "use ONLY defined field names and exact enum values. "
                "Do not add fields that aren't in the schema."
            )},
        ]

    result.elapsed_s = _time.time() - t0
    return result


# ----- ID + dataset-dir management ----------------------------------------


_ID_RE = re.compile(r"^example-(\d{3})\.json$")


def next_example_id(dataset_dir: Path, override: int | None) -> int:
    """Find the next monotonic example ID. Considers both the dataset/
    directory (synthetic) AND the parent docs/account-intelligence/
    directory (hand-built examples 001-008)."""
    if override is not None:
        return override
    dataset_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(dataset_dir.glob("example-*.json"))
    candidates.extend(ROOT.glob("example-???.json"))
    ids: list[int] = []
    for p in candidates:
        m = _ID_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return max(ids) + 1 if ids else 1


# ----- CLI entrypoint -----------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic example generator for the account intelligence model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--surface", required=True, choices=SURFACES)
    p.add_argument("--count", type=int, default=1)
    p.add_argument(
        "--shape",
        choices=ACCOUNT_SHAPES,
        help="Fix the account shape (default: varied per row).",
    )
    p.add_argument(
        "--edge-cases",
        help=(
            "Comma-separated edge cases to apply to every row "
            "(default: sampled per row)."
        ),
    )
    p.add_argument("--source-mix", choices=SOURCE_MIXES)
    p.add_argument(
        "--account-hint",
        help="1-line nudge passed to every row in this batch.",
    )
    p.add_argument(
        "--start-id",
        type=int,
        help="First example ID (default: max existing + 1).",
    )
    p.add_argument("--mode", choices=["path-a", "path-b"], default="path-b",
                   help="path-b (default): emit source_bundle + brief — the correct "
                        "training signal for a real inference model. "
                        "path-a: legacy knob-only generation (brief only, no source data).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model ID to use (default: {DEFAULT_MODEL}).")
    p.add_argument("--base-url", default=None,
                   help=f"API base URL (default: $GATEWAY_BASE_URL or {DEFAULT_BASE_URL}). "
                        "Use https://api.anthropic.com/v1 for direct Anthropic access.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cost-cap",
        type=float,
        default=DEFAULT_COST_CAP_USD,
        help="Abort if estimated total cost exceeds USD (default 20).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matrix + cost estimate; don't call the API.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print validation errors for retried-but-succeeded examples.",
    )
    p.add_argument(
        "--dataset-dir",
        default=str(DATASET_DIR),
        help=f"Output directory (default: {DATASET_DIR}).",
    )
    return p.parse_args()


def estimate_cost_usd(count: int) -> float:
    """Rough up-front estimate including retry factor. Used only for
    the cost-cap gate; actual cost will vary."""
    per_call = (
        EST_INPUT_TOKENS * COST_INPUT_PER_M_USD / 1_000_000
        + EST_OUTPUT_TOKENS * COST_OUTPUT_PER_M_USD / 1_000_000
    )
    return count * per_call * EST_RETRY_FACTOR


def main() -> int:
    args = parse_args()

    if args.mode == "path-b":
        for p in (SOURCE_BUNDLE_PROMPT_PATH, INFERENCE_PROMPT_PATH):
            if not p.exists():
                print(f"ERROR: prompt not found at {p}", file=sys.stderr)
                return 2
        source_bundle_prompt = SOURCE_BUNDLE_PROMPT_PATH.read_text()
        inference_prompt = INFERENCE_PROMPT_PATH.read_text()
        system_prompt = None  # not used in path-b
    else:
        if not SYSTEM_PROMPT_PATH.exists():
            print(f"ERROR: system prompt not found at {SYSTEM_PROMPT_PATH}", file=sys.stderr)
            return 2
        system_prompt = SYSTEM_PROMPT_PATH.read_text()
        source_bundle_prompt = inference_prompt = None

    if not SCHEMA_PATH.exists():
        print(f"ERROR: schema not found at {SCHEMA_PATH}", file=sys.stderr)
        return 2

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    print(f"Mode: {args.mode}")

    edge_cases = (
        [s.strip() for s in args.edge_cases.split(",") if s.strip()]
        if args.edge_cases
        else None
    )
    matrix = build_matrix(
        surface=args.surface,
        count=args.count,
        shape=args.shape,
        edge_cases=edge_cases,
        source_mix=args.source_mix,
        account_hint=args.account_hint,
        seed=args.seed,
    )

    est = estimate_cost_usd(args.count)
    print(f"\nMatrix: {args.count} example(s) × surface={args.surface}")
    for i, params in enumerate(matrix, start=1):
        ec_str = f"{params.edge_cases}" if params.edge_cases else "[]"
        sm_str = params.source_mix or "(auto)"
        print(
            f"  {i:3}. shape={params.account_shape:22} ec={ec_str} "
            f"source_mix={sm_str}"
        )
    print(f"\nEstimated cost: ~${est:.2f} (cap ${args.cost_cap:.2f})")

    if est > args.cost_cap:
        print(
            f"ERROR: estimated cost ${est:.2f} exceeds cap ${args.cost_cap:.2f}. "
            "Raise --cost-cap or reduce --count.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("\n--dry-run set; not calling API.")
        return 0

    api_key = (
        os.environ.get("GATEWAY_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print(
            "ERROR: set GATEWAY_API_KEY (or ANTHROPIC_API_KEY) in environment or .env.",
            file=sys.stderr,
        )
        return 2

    base_url = args.base_url or os.environ.get("GATEWAY_BASE_URL") or DEFAULT_BASE_URL
    # Apply mode-appropriate defaults if the user didn't override them explicitly.
    max_tokens = args.max_tokens
    retries = args.retries
    if args.mode == "path-b":
        if max_tokens == DEFAULT_MAX_TOKENS:   # user didn't override
            max_tokens = DEFAULT_MAX_TOKENS_PATH_B
        if retries == DEFAULT_RETRIES:         # user didn't override
            retries = DEFAULT_RETRIES_PATH_B

    print(f"Gateway: {base_url}  Model: {args.model}  max_tokens: {max_tokens}  retries: {retries}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    next_id = next_example_id(dataset_dir, args.start_id)

    successes = 0
    failures = 0
    print()
    for i, params in enumerate(matrix, start=1):
        ec_str = ",".join(params.edge_cases) if params.edge_cases else "-"
        print(
            f"[{i}/{args.count}] id={next_id:03} shape={params.account_shape} "
            f"ec={ec_str}"
        )
        if args.mode == "path-b":
            result = generate_one_path_b(
                client=client,
                source_bundle_prompt=source_bundle_prompt,
                inference_prompt=inference_prompt,
                validator=validator,
                params=params,
                model=args.model,
                max_tokens=max_tokens,
                temperature=args.temperature,
                retries=retries,
            )
        else:
            result = generate_one(
                client=client,
                system_prompt=system_prompt,
                validator=validator,
                params=params,
                model=args.model,
                max_tokens=max_tokens,
                temperature=args.temperature,
                retries=retries,
                mode=args.mode,
            )
        if result.doc is None:
            print(
                f"  ✗ failed after {result.attempts} attempt(s) "
                f"({result.elapsed_s:.1f}s)"
            )
            for log in result.errors_log:
                print(f"    {log}")
            # Use a unique-per-failure path. The slot id is taken by the
            # next successful run; failures get a monotonic FAIL-NNN
            # suffix derived from the timestamp + row index so a batch
            # with 5 failures doesn't overwrite one log file 5 times.
            fail_path = dataset_dir / (
                f"FAILED-batch{int(time.time())}-row{i:02}.txt"
            )
            fail_path.write_text(
                "PARAMS:\n"
                + params.format_user_prompt()
                + "\n\nERRORS:\n"
                + "\n".join(result.errors_log)
                + "\n\nLAST RAW OUTPUT:\n"
                + (result.last_raw_text or "(none)")
            )
            print(f"    saved failed output to {fail_path.name}")
            failures += 1
        else:
            path = dataset_dir / f"example-{next_id:03}.json"
            path.write_text(json.dumps(result.doc, indent=2))
            print(
                f"  ✓ {path.name} ({result.elapsed_s:.1f}s, "
                f"{result.attempts} attempt(s))"
            )
            if args.verbose and result.errors_log:
                for log in result.errors_log:
                    print(f"    {log}")
            successes += 1
            next_id += 1

    print()
    print("=" * 60)
    print(f"Done: {successes} succeeded, {failures} failed")
    if failures:
        print(
            f"Investigate FAILED-NNN.txt files in {dataset_dir} "
            "before retrying."
        )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
