#!/usr/bin/env python3
"""Eval harness for the account intelligence model.

Computes the automatable metrics from docs/account-intelligence/evaluation.md
against a set of model outputs (or against the gold dataset as a sanity
baseline). Skips metrics that require the original input data
(hallucination, empty_sections accuracy) or human judgment (style).

Usage:
    # Sanity check — score the gold examples themselves (Claude baseline)
    python eval.py --gold

    # Score model outputs from a directory of example-NNN.json files
    python eval.py --model path/to/model-outputs/

Outputs a per-surface and overall report card.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


HERE = Path(__file__).parent.resolve()
SCHEMA_PATH = HERE / "schemas" / "account-intelligence" / "schema.json"
DATA_DIR = HERE / "datasets" / "account-intelligence"
EVAL_SPLIT = DATA_DIR / "eval_split.txt"


# Surface-relevance map (from docs/account-intelligence/evaluation.md).
# Sections the model SHOULD consider for each surface (populate or
# declare in empty_sections).
SURFACE_SECTIONS: dict[str, set[str]] = {
    "meeting_prep": {
        "must_address", "pipeline", "relationship_map", "recent_activity",
        "external_signal", "knowledge_context", "pricing_history",
        "asks_and_next_steps", "landmines", "talking_points",
        "carryover", "meeting_opener",
    },
    "qbr": {
        "must_address", "pipeline", "support_health", "relationship_map",
        "recent_activity", "external_signal", "knowledge_context",
        "pricing_history", "renewal", "asks_and_next_steps",
        "talking_points",
    },
    "handoff": {
        "pipeline", "support_health", "relationship_map", "recent_activity",
        "timeline", "asks_and_next_steps", "landmines",
    },
    "renewal_alert": {
        "must_address", "pipeline", "support_health", "relationship_map",
        "external_signal", "pricing_history", "renewal",
        "asks_and_next_steps", "landmines",
    },
    "onboarding": {
        "relationship_map", "recent_activity", "knowledge_context",
        "timeline", "asks_and_next_steps",
    },
    "escalation": {
        "must_address", "support_health", "recent_activity",
        "asks_and_next_steps",
    },
}

# Sections whose items each carry a sources[] array (per evaluation.md
# Metric 2 + Metric 5 — the claim-bearing surfaces). For nested cases
# (relationship_map.contacts, external_signal.news, etc.) the walker
# below handles the descent.
CLAIM_PATHS = [
    ("must_address", None),
    ("pipeline", "items"),
    ("recent_activity", None),
    ("asks_and_next_steps", None),
    ("landmines", None),
    ("timeline", None),
    ("relationship_map", "contacts"),
    ("relationship_map", "champions"),
    ("relationship_map", "decision_makers"),
    ("relationship_map", "detractors"),
    ("external_signal", "news"),
    ("external_signal", "internal_chatter"),
    ("renewal", "risk_signals"),
    ("support_health", "items"),
]


def load_eval_ids() -> list[str]:
    ids = []
    for line in EVAL_SPLIT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def load_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    return Draft202012Validator(schema)


# ----- per-example metrics --------------------------------------------------

def metric_schema_adherence(doc: dict, validator: Draft202012Validator) -> tuple[bool, list[str]]:
    """M1. Returns (valid, error_messages)."""
    errors = list(validator.iter_errors(doc))
    if not errors:
        return True, []
    return False, [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors[:5]
    ]


def metric_section_coverage(doc: dict) -> tuple[int, int]:
    """M3. Returns (considered, relevant)."""
    surface = doc.get("surface")
    relevant = SURFACE_SECTIONS.get(surface, set())
    if not relevant:
        return 0, 0
    empty = set(doc.get("empty_sections") or [])
    considered = 0
    for section in relevant:
        val = doc.get(section)
        populated = False
        if isinstance(val, list):
            populated = len(val) > 0
        elif isinstance(val, dict):
            # Populated if any nested list is non-empty or it has a non-empty string field
            populated = any(
                (isinstance(v, list) and v)
                or (isinstance(v, str) and v.strip())
                or (isinstance(v, dict) and v)
                for v in val.values()
            )
        elif isinstance(val, str):
            populated = bool(val.strip())
        if populated or section in empty:
            considered += 1
    return considered, len(relevant)


def metric_source_attribution(doc: dict) -> tuple[int, int]:
    """M5. Returns (claims_with_sources, total_claims)."""
    total = 0
    sourced = 0
    for parent, child in CLAIM_PATHS:
        section = doc.get(parent)
        if section is None:
            continue
        if child is None:
            items = section if isinstance(section, list) else []
        else:
            items = (section.get(child) if isinstance(section, dict) else None) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            total += 1
            srcs = item.get("sources")
            if isinstance(srcs, list) and len(srcs) > 0:
                sourced += 1
    return sourced, total


# ----- report ---------------------------------------------------------------

def score_one(path: Path, validator: Draft202012Validator) -> dict:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return {
            "id": path.stem, "surface": "?", "schema_valid": False,
            "schema_errors": [f"json parse: {e}"],
            "coverage": (0, 1), "attribution": (0, 0),
        }
    valid, errs = metric_schema_adherence(doc, validator)
    return {
        "id": path.stem,
        "surface": doc.get("surface", "?"),
        "schema_valid": valid,
        "schema_errors": errs,
        "coverage": metric_section_coverage(doc),
        "attribution": metric_source_attribution(doc),
    }


def report(scores: list[dict]) -> None:
    by_surface: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        by_surface[s["surface"]].append(s)

    print(f"\n{'Surface':<16} {'N':>3} {'M1 schema':>11} {'M3 cover':>11} {'M5 attrib':>11}")
    print("-" * 56)

    def fmt_pct(num, den, target=None):
        if den == 0:
            return "  n/a"
        pct = 100 * num / den
        marker = ""
        if target is not None:
            marker = " ✓" if pct >= target else " ✗"
        return f"{pct:5.1f}%{marker}"

    overall_n = 0
    overall_schema = 0
    overall_cov_n = 0; overall_cov_d = 0
    overall_attr_n = 0; overall_attr_d = 0

    for surface in sorted(by_surface):
        group = by_surface[surface]
        n = len(group)
        schema_ok = sum(1 for s in group if s["schema_valid"])
        cov_n = sum(s["coverage"][0] for s in group)
        cov_d = sum(s["coverage"][1] for s in group)
        attr_n = sum(s["attribution"][0] for s in group)
        attr_d = sum(s["attribution"][1] for s in group)
        print(
            f"{surface:<16} {n:>3} "
            f"{fmt_pct(schema_ok, n):>11} "
            f"{fmt_pct(cov_n, cov_d):>11} "
            f"{fmt_pct(attr_n, attr_d):>11}"
        )
        overall_n += n
        overall_schema += schema_ok
        overall_cov_n += cov_n; overall_cov_d += cov_d
        overall_attr_n += attr_n; overall_attr_d += attr_d

    print("-" * 56)
    print(
        f"{'OVERALL':<16} {overall_n:>3} "
        f"{fmt_pct(overall_schema, overall_n, target=95):>11} "
        f"{fmt_pct(overall_cov_n, overall_cov_d, target=90):>11} "
        f"{fmt_pct(overall_attr_n, overall_attr_d, target=80):>11}"
    )
    print("\nTargets (v0.1): M1 ≥ 95%, M3 ≥ 90%, M5 ≥ 80%")
    print("\nFailures detail:")
    fail_count = 0
    for s in scores:
        if not s["schema_valid"]:
            fail_count += 1
            print(f"  {s['id']} [{s['surface']}]:")
            for e in s["schema_errors"][:3]:
                print(f"    - {e}")
    if fail_count == 0:
        print("  (none)")

    print("\nMetrics NOT scored automatically:")
    print("  M2 Hallucination       — requires input source data")
    print("  M4 empty_sections acc  — requires input source data")
    print("  M6 Confidence calib    — derived from M2")
    print("  M7 Latency             — collected at inference time")
    print("  M8 Style               — manual review")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gold", action="store_true",
                   help="Score the gold examples themselves (Claude baseline)")
    g.add_argument("--model", type=Path,
                   help="Directory containing model outputs (example-NNN.json)")
    args = ap.parse_args()

    ids = load_eval_ids()
    validator = load_validator()
    src_dir = DATA_DIR if args.gold else args.model
    if not src_dir.exists():
        print(f"ERROR: directory {src_dir} not found", file=sys.stderr)
        return 1

    scores = []
    missing = []
    for eid in ids:
        path = src_dir / f"{eid}.json"
        if not path.exists():
            missing.append(eid)
            continue
        scores.append(score_one(path, validator))

    label = "GOLD (Claude baseline)" if args.gold else f"MODEL: {args.model}"
    print(f"\n=== Eval report — {label} ===")
    print(f"Eval set: {EVAL_SPLIT.name} ({len(ids)} IDs)")
    if missing:
        print(f"WARNING: {len(missing)} missing outputs: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    report(scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
