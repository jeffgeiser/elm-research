"""Schema-constrained decoding for the account intelligence brief (Outlines).

The fine-tuned model writes good content but invents field names and enum
values, which fails the strict (`additionalProperties: false`) schema. Outlines
compiles schema.json into a token-level mask, so every generated token keeps
the output on a path to a schema-valid brief.

Shared by infer_eval.py and any in-process serving path:

    generator = build_generator(model, tokenizer)
    text = generator(prompt, max_new_tokens=8192, do_sample=False)
    brief = finalize(json.loads(text))

Two things the mask can NOT guarantee:
  - `uniqueItems` (not expressible as a regex) — the array is length-capped
    and finalize() dedupes instead.
  - termination — a generation cut off by max_new_tokens is a truncated,
    unparseable prefix. Callers must still handle a parse failure.

Key order is fixed to the order of `properties` in schema.json. A section the
model would have emitted "late" can no longer be emitted once passed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SCHEMA_PATH = HERE / "schemas" / "account-intelligence" / "schema.json"


def load_constraint_schema(schema_path: Path = SCHEMA_PATH) -> dict:
    """schema.json minus `_meta`.

    `_meta` is dataset annotation, not model output, and it is a free-form
    object — leaving it in would give the model an unconstrained escape hatch
    at the very start of the brief.
    """
    schema = json.loads(schema_path.read_text())
    schema["properties"].pop("_meta", None)
    # A `uniqueItems` enum array can never validly hold more entries than the
    # enum has values. The mask can't enforce uniqueness, so without this cap
    # the model is free to cycle through enum values until max_new_tokens
    # (observed on `sources_used`).
    for prop in schema["properties"].values():
        enum = prop.get("items", {}).get("enum")
        if prop.get("uniqueItems") and enum:
            prop.setdefault("maxItems", len(enum))
    return schema


def build_generator(model, tokenizer, schema_path: Path = SCHEMA_PATH):
    """Wrap a transformers-compatible model (PEFT/Unsloth included) in an
    Outlines generator constrained to the brief schema.

    Building the token index takes ~30s for the Qwen2.5 vocabulary; build once
    and reuse the generator across requests.
    """
    import outlines
    from outlines.types import JsonSchema

    schema = load_constraint_schema(schema_path)
    return outlines.Generator(
        outlines.from_transformers(model, tokenizer),
        JsonSchema(json.dumps(schema)),
    )


def finalize(doc: dict, schema_path: Path = SCHEMA_PATH) -> dict:
    """Dedupe the root arrays the schema marks `uniqueItems` (order-preserving)."""
    schema = load_constraint_schema(schema_path)
    for key, prop in schema["properties"].items():
        if prop.get("uniqueItems") and isinstance(doc.get(key), list):
            doc[key] = list(dict.fromkeys(doc[key]))
    return doc
