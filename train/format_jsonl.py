#!/usr/bin/env python3
"""Format the gold dataset as chat-format JSONL for SFT.

IMPORTANT — what this trains:
    The gold examples are (constraints → brief) pairs from the synthetic
    pipeline. The user prompt is the slot-filled generation template
    (surface + account_shape + edge_cases + source_mix), the assistant
    response is the brief JSON. This trains a SYNTHESIS model: given
    constraints, fabricate a plausible brief.

    To train a SUMMARIZATION model (real SF data → brief), Phase 3
    needs to reconstruct input packs (SF rows, opps, contacts, cases,
    activities, Teams messages, news) for each example, then re-pair
    those inputs with the brief. This script does not do that.

Splits:
    - Holds out the 50 IDs in datasets/account-intelligence/eval_split.txt
      to a separate eval.jsonl (used for in-training validation).
    - Everything else goes to train.jsonl.

Output per line:
    {
        "id": "example-NNN",
        "messages": [
            {"role": "system",    "content": "<system prompt>"},
            {"role": "user",      "content": "<slot-filled user prompt>"},
            {"role": "assistant", "content": "<brief as JSON string, _meta stripped>"}
        ]
    }

The top-level "id" field is for the post-checkpoint eval callback to
bind predictions back to gold examples robustly — it does NOT depend
on line order. TRL's SFTTrainer ignores unknown top-level fields when
preparing the dataset, so this is safe.

Usage:
    python train/format_jsonl.py                      # default paths
    python train/format_jsonl.py --out train/data/    # custom output dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "account-intelligence" / "system-prompt.md"
DATA_DIR = ROOT / "datasets" / "account-intelligence"
EVAL_SPLIT_PATH = DATA_DIR / "eval_split.txt"


# Canonical tokens — must match generate.py's SURFACES / ACCOUNT_SHAPES /
# EDGE_CASES / SOURCE_MIXES. If those drift, update here too.
CANONICAL_SHAPES = [
    "enterprise_steady", "enterprise_expanding",
    "mid_market_active", "mid_market_quiet",
    "small_strategic", "stalled_at_risk",
    "fresh_handoff", "incident_acute",
]
SHAPE_RE = re.compile(r"(" + "|".join(CANONICAL_SHAPES) + r")")

CANONICAL_SOURCE_MIXES = [
    "sf_only", "with_news", "with_teams", "with_prior_prep",
    "rich", "minimal",
]
SOURCE_MIX_RE = re.compile(r"(" + "|".join(CANONICAL_SOURCE_MIXES) + r")")

# Edge-case canonical tokens. We extract these by fuzzy-matching against
# the descriptive _meta.edge_cases_included entries Claude wrote.
CANONICAL_EDGE_CASES = [
    "subsidiary_explosion", "null_industry", "mis_categorized_industry",
    "multi_currency_amount", "sparse_activity", "activity_overlogged",
    "bot_filed_case_noise", "p1_open_long", "closed_lost_recent",
    "champion_churn", "non_latin_contact", "duplicate_contacts",
    "dc_code_gap", "null_priority_cases", "audit_cycle",
    "public_announcement_anchor", "utilization_drop", "opp_name_typo",
    "vp_newly_visible",
]
# Map canonical token → list of substrings that, if present in a
# descriptive edge-case line, identify it. Kept loose intentionally.
EDGE_CASE_HINTS: dict[str, list[str]] = {
    "subsidiary_explosion": ["subsidiar", "multiple sf row"],
    "null_industry": ["null industry", "industry comes back null", "industry is null"],
    "mis_categorized_industry": ["mis-categori", "industry tagged incorrectly", "wrong industry"],
    "multi_currency_amount": ["multi_currency", "multi-currency", "raw amount"],
    "sparse_activity": ["sparse activity", "activity logged >30", "last activity"],
    "activity_overlogged": ["overlogged", "50+ activities", "null who"],
    "bot_filed_case_noise": ["bot-filed", "bot filed", "rbl", "abuse-bot"],
    "p1_open_long": ["p1 case", "p1 open", "p1_open"],
    "closed_lost_recent": ["closed lost", "closed_lost"],
    "champion_churn": ["champion churn", "champion departed", "primary contact depart"],
    "non_latin_contact": ["non-latin", "non latin", "japanese", "chinese character", "kanji"],
    "duplicate_contacts": ["duplicate contact", "duplicate sf contact", "duplicate row"],
    "dc_code_gap": ["dc code", "dc_code"],
    "null_priority_cases": ["null priority", "priority is null", "priority field null"],
    "audit_cycle": ["audit cycle", "soc2", "ffiec", "hitrust", "infosec audit"],
    "public_announcement_anchor": ["public announcement", "press release", "earnings call"],
    "utilization_drop": ["utilization drop", "utilization below", "utilization_drop"],
    "opp_name_typo": ["opp name typo", "data-entry error", "typo"],
    "vp_newly_visible": ["vp newly visible", "senior executive appears", "vp_newly_visible"],
}


def extract_shape(meta: dict) -> str | None:
    s = (meta.get("shape_constraints") or "").lower()
    m = SHAPE_RE.search(s)
    if m:
        return m.group(1)
    # Hyphenated-form fallback — Claude occasionally wrote "mid-market"
    # or "enterprise" instead of the canonical underscore token.
    if "mid-market" in s or "midmarket" in s:
        if any(w in s for w in ("active", "high engagement", "multi-deal", "expanding pipeline")):
            return "mid_market_active"
        return "mid_market_quiet"
    if "enterprise" in s:
        if any(w in s for w in ("expanding", "expansion", "growth")):
            return "enterprise_expanding"
        return "enterprise_steady"
    if "small" in s or "strategic" in s:
        return "small_strategic"
    if "stalled" in s or "at risk" in s or "at-risk" in s:
        return "stalled_at_risk"
    if "handoff" in s or "hand-off" in s:
        return "fresh_handoff"
    if "incident" in s or "p1" in s:
        return "incident_acute"
    return None


def extract_source_mix(meta: dict) -> str | None:
    s = meta.get("shape_constraints", "")
    m = SOURCE_MIX_RE.search(s or "")
    return m.group(1) if m else None


def extract_edge_cases(meta: dict) -> list[str]:
    descriptive = meta.get("edge_cases_included") or []
    found: list[str] = []
    for entry in descriptive:
        low = entry.lower()
        for token, hints in EDGE_CASE_HINTS.items():
            if any(h in low for h in hints) and token not in found:
                found.append(token)
                break
    return found


def reconstruct_user_prompt(meta: dict, surface: str) -> str:
    """Rebuild the slot-filled user prompt that originally produced this brief."""
    shape = extract_shape(meta) or "enterprise_steady"  # fallback
    edge_cases = extract_edge_cases(meta)
    source_mix = extract_source_mix(meta)

    lines = [
        "Generate one synthetic example with these parameters:",
        "",
        f"- surface: {surface}",
        f"- account_shape: {shape}",
    ]
    if edge_cases:
        lines.append(f"- edge_cases: {json.dumps(edge_cases)}")
    if source_mix:
        lines.append(f"- source_mix: {source_mix}")
    return "\n".join(lines)


def load_eval_ids() -> set[str]:
    ids: set[str] = set()
    for line in EVAL_SPLIT_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def build_record(example_id: str, doc: dict, system_prompt: str) -> dict:
    meta = doc.get("_meta", {})
    surface = doc.get("surface") or meta.get("surface")
    user_prompt = reconstruct_user_prompt(meta, surface)
    # Strip _meta from the assistant output — it's annotation, not contract.
    assistant_doc = {k: v for k, v in doc.items() if k != "_meta"}
    return {
        "id": example_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_doc, ensure_ascii=False)},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data",
                    help="Output directory for train.jsonl and eval.jsonl")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    eval_path = args.out / "eval.jsonl"

    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    eval_ids = load_eval_ids()

    files = sorted(DATA_DIR.glob("example-*.json"))
    print(f"Loaded {len(files)} gold examples")
    print(f"Eval split: {len(eval_ids)} held-out IDs")

    train_count = 0
    eval_count = 0
    by_surface_train: dict[str, int] = defaultdict(int)
    coverage_warnings: list[str] = []

    with open(train_path, "w") as ftrain, open(eval_path, "w") as feval:
        for f in files:
            doc = json.loads(f.read_text())
            rec = build_record(f.stem, doc, system_prompt)
            surface = doc.get("surface") or doc.get("_meta", {}).get("surface")

            # Sanity-check: did slot reconstruction find a shape?
            meta = doc.get("_meta", {})
            if extract_shape(meta) is None:
                coverage_warnings.append(f"{f.stem}: no canonical shape extracted from _meta")

            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if f.stem in eval_ids:
                feval.write(line)
                eval_count += 1
            else:
                ftrain.write(line)
                train_count += 1
                by_surface_train[surface] += 1

    print(f"\nWrote {train_path} ({train_count} examples)")
    print(f"Wrote {eval_path} ({eval_count} examples)")
    print("\nTrain split by surface:")
    for s, n in sorted(by_surface_train.items()):
        print(f"  {s}: {n}")

    if coverage_warnings:
        print(f"\n{len(coverage_warnings)} reconstruction warning(s):")
        for w in coverage_warnings[:10]:
            print(f"  {w}")
        if len(coverage_warnings) > 10:
            print(f"  ...and {len(coverage_warnings) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
