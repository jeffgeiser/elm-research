# Account Intelligence Model — Phase 1 artifacts

A fine-tuned ~4B model specialized in account intelligence synthesis,
covering six enterprise surfaces:

1. **meeting_prep** — pre-call briefing
2. **qbr** — quarterly business review
3. **handoff** — rep transition / new owner briefing
4. **renewal_alert** — renewal-window risk briefing
5. **onboarding** — new rep getting up to speed on a territory
6. **escalation** — incident / issue triage

**Release story:** one model, six surfaces, runs locally, outputs
validated JSON, $0 per call after hardware, matches Claude on schema
adherence at 10× the speed.

---

## Phase 1 deliverables (this directory)

| File                          | Status | Purpose |
|-------------------------------|--------|---------|
| `schema.json`                 | ✓      | The output contract. JSON Schema Draft 2020-12. Every training example and every model output validates against this. |
| `data-audit.md`               | ✓      | What we can actually pull per source (SF, Confluence, Teams, SharePoint, news, knowledge, pricing history). Format quirks, reliability columns, gaps. |
| `example-001.template.json`   | ✓      | Hand-build template. Pick a real account, fill in actual values from every source, save as `example-001.json` (not committed publicly — keep customer detail internal). |
| `evaluation.md`               | ✓      | The eight metrics every checkpoint is scored against. Locked before any training code lands. |

---

## How the pieces relate

```
schema.json ────────► every training example MUST conform
   │
   ├─► data-audit.md ──► tells us what input data can populate which schema fields
   │
   ├─► example-001.json (hand-built) ──► the "gold standard" — every synthetic example
   │                                     should be at least this good
   │
   └─► evaluation.md ──► defines pass/fail per checkpoint, locked before training
```

---

## What's NEXT (Phase 2 — not yet started)

1. **Export existing meeting_prep pairs.** ~100 real briefs from
   `meeting_preps` table. Tag each with `surface: meeting_prep`.
   Re-shape into the schema if any fields are missing.
2. **Generate synthetic examples.** ~80–100 per non-meeting-prep
   surface using Claude with the schema + example-001 + this audit
   as the system prompt. Cover edge cases deliberately (empty
   sections, ambiguous accounts, multi-currency opps, etc.).
3. **Manually validate a 50-example eval split.** Hand-check every
   example against its source data. This becomes the locked
   eval set.
4. **Train.** Qwen 2.5 4B base, LoRA, on a single H100. The 600
   examples should fit comfortably in a few-hour run.
5. **Score against `evaluation.md`.** Iterate until v0.1 thresholds
   are met.

---

## Why this is a real product, not a personal tool

- **Releasable artifact.** Hugging Face model card + X thread. The
  numbers tell the story; no marketing required.
- **Schema is the moat.** The six-surface JSON contract is harder
  to copy than the model weights. Anyone with a GPU can train a
  4B model; few have thought through the schema across all six
  surfaces.
- **Local + private + free per call.** Three things the API-only
  alternatives can't claim simultaneously.
- **Dataset is the asset.** ~600 high-quality examples specific to
  enterprise account intelligence don't exist elsewhere on the
  internet. The model is the side effect.

---

## Cross-references

- `api/lib/llm/prompts/meeting_prep.txt` — current Claude-target
  prompt; informs the schema we're committing to.
- `api/modules/learning/` — the Phase 7.12 Learning module is the
  delivery surface for "AI literacy + Zenlayer-specific workloads"
  training. The account intelligence model is a separate but
  adjacent project.
- `api/lib/integrations/salesforce_queries.py` — the five SF helper
  functions that produce the input data for every training example.
