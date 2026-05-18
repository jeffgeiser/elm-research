# Evaluation — what "good" means for the account intelligence model

Locked before any fine-tuning code lands. If the eval metric isn't
defined, we'll tune to whatever's easy to measure rather than what
matters.

The goal: a fine-tuned ~4B model that matches Claude-class on the
metrics below at 10× the speed and $0 per call after hardware. The
metrics are the contract — every checkpoint is scored against the
same fixed eval set.

---

## The eval set

- **Size:** 50 held-out briefs at v0.1, growing to 200 by v1.0.
- **Composition:** stratified across the six surfaces (8–10 per
  surface), with explicit coverage of edge cases (empty sections,
  ambiguous account resolution, multi-currency opps, sparse
  activity).
- **Source:** real production preps + manually-corrected synthetic
  examples. Every eval example has been hand-validated to match the
  schema and be factually correct against its source data.
- **Locked:** the eval set is **frozen** once v0.1 ships. New
  examples land in a separate `eval-v0.2/` directory and roll into
  the next major version. Tuning against the live eval set is
  reward hacking.

---

## Metric 1 — Schema adherence (% of outputs that validate)

**Definition:** percentage of model outputs that parse as valid JSON
AND validate against `schema.json` (Draft 2020-12).

**Why:** the model is useless downstream if the output doesn't parse.
This is the floor — every checkpoint must clear 95% before any other
metric is reported.

**How to measure:** run the eval set through the model; pipe each
output through `jsonschema.validate(instance, schema)`. Count
passes / total.

**Target:**
- v0.1: 95%
- v0.5: 99%
- v1.0: 99.9%

**Common failure modes to log:**
- Truncated JSON at end of response (token-limit hits)
- Missing required fields
- Wrong types on optional fields (e.g. `pipeline.items[].amount` as
  number when schema says string)
- Enum values outside the allowed set

---

## Metric 2 — Hallucination rate (% of claims with no source support)

**Definition:** ratio of material claims in the output that have
NO valid source attribution to the input data, vs. total material
claims.

**Why:** a brief that confabulates is worse than no brief — reps
trust it, get burned, stop using the tool. This is the metric that
makes or breaks the product story.

**How to measure:** semi-automated.
1. For each output, extract every claim from `must_address`,
   `pipeline.items`, `recent_activity`, `relationship_map.*`,
   `external_signal.*`, `asks_and_next_steps`, `landmines`,
   `renewal.*`.
2. For each claim, check whether its `sources[]` array (or the
   parent section's source attribution) references a real ID in
   the input data.
3. A claim with no `sources[]` AND no traceable phrase in any input
   block counts as a hallucination.

**Target:**
- v0.1: ≤ 5%
- v0.5: ≤ 2%
- v1.0: ≤ 0.5%

**Edge cases:**
- A `talking_points` entry is opinion, not a factual claim — exempt.
- `executive_summary` is synthesis; check that every named entity
  appears in the input.
- `landmines` based on `prior_prep` sources count as grounded.

---

## Metric 3 — Section coverage (% of relevant sections populated)

**Definition:** for each surface, the schema implies a set of
sections that should be considered. Coverage = (sections populated
OR declared in `empty_sections`) / (relevant sections for the
surface).

**Why:** a brief that just skips half the schema isn't useful even
if what it does emit is accurate. The model needs to consider every
relevant section and either fill it or declare it empty.

**Surface-relevance map:**

| Section            | meeting_prep | qbr | handoff | renewal_alert | onboarding | escalation |
|--------------------|--------------|-----|---------|---------------|------------|------------|
| must_address       | ✓            | ✓   |         | ✓             |            | ✓          |
| pipeline           | ✓            | ✓   | ✓       | ✓             |            |            |
| support_health     |              | ✓   | ✓       | ✓             |            | ✓          |
| relationship_map   | ✓            | ✓   | ✓       | ✓             | ✓          |            |
| recent_activity    | ✓            | ✓   | ✓       |               | ✓          | ✓          |
| external_signal    | ✓            | ✓   |         | ✓             |            |            |
| knowledge_context  | ✓            | ✓   |         |               | ✓          |            |
| pricing_history    | ✓            | ✓   |         | ✓             |            |            |
| renewal            |              | ✓   |         | ✓             |            |            |
| timeline           |              |     | ✓       |               | ✓          |            |
| asks_and_next_steps| ✓            | ✓   | ✓       | ✓             | ✓          | ✓          |
| landmines          | ✓            |     | ✓       | ✓             |            |            |
| talking_points     | ✓            | ✓   |         |               |            |            |
| carryover          | ✓            |     |         |               |            |            |
| meeting_opener     | ✓            |     |         |               |            |            |

**Target:**
- v0.1: 90%
- v1.0: 99%

---

## Metric 4 — `empty_sections` accuracy

**Definition:** of the sections the model marked as empty, what
fraction were correctly empty (no signal in input) vs incorrectly
empty (signal existed but the model missed it)?

**Why:** declaring absence is the anti-hallucination guard, but
it has to be honest. A model that dumps everything into
`empty_sections` to dodge the work is also a failure.

**How to measure:**
- True empty: section is in `empty_sections` AND input has no
  relevant signal. ✓
- False empty: section is in `empty_sections` BUT input has signal
  the model missed. ✗ (counts as a miss)
- Hidden empty: section is omitted AND not in `empty_sections`,
  AND input has no signal. Also ✗ (model didn't declare absence)

**Target:**
- v0.1: 90% true-empty rate
- v1.0: 98%

---

## Metric 5 — Source attribution coverage

**Definition:** of all material claims, what fraction have a
non-empty `sources[]` array?

**Why:** sources are the audit trail. A brief that's accurate but
unsourced can't be trust-verified at scale.

**How to measure:**
1. Material claims = same set as Metric 2.
2. Count claims with `len(sources) >= 1` / total.

**Target:**
- v0.1: 80%
- v1.0: 95%

(Lower target than other metrics because some claims — e.g.
`executive_summary` synthesis, `confidence.rationale` — are
legitimately unsourced at the claim level.)

---

## Metric 6 — Confidence calibration

**Definition:** correlation between `confidence.overall` and actual
quality. When the model says `high`, hallucination rate should be
near-zero; when it says `low`, the rate is allowed to be higher.

**Why:** downstream consumers route low-confidence briefs to human
review. If `low` doesn't correlate with actual problems, the
routing is noise.

**How to measure:**
- For each confidence tier, compute the hallucination rate (Metric 2)
  across briefs at that tier.
- A well-calibrated model has: `high` < 1%, `medium` 1–5%, `low` > 5%.
- Mis-calibrated: `high` confidence correlates with high hallucination.

**Target:** monotonic relationship by v0.5; tight thresholds
(above) by v1.0.

---

## Metric 7 — Latency

**Definition:** p50 and p95 time to full output, including JSON
parse, for the eval set on a single L40S.

**Why:** the product story is "10× faster than frontier-API." Has
to be measurable.

**How to measure:** straightforward — wall-clock from request send
to last token received.

**Target:**
- v0.1: p50 < 5s, p95 < 12s
- v1.0: p50 < 2s, p95 < 5s

---

## Metric 8 — Style / tone (qualitative)

**Definition:** does the output read like the production prompt
guidance — lead with results, no marketing tone, no "as we
discussed," no apologetic framing?

**Why:** quantitative metrics can be high while output reads like
a robot wrote it. The reps won't keep using a tool that sounds
generated.

**How to measure:**
- Manual review of 10 random outputs per surface, per checkpoint.
- 1–5 scale on: leads-with-results, no-filler, customer-email-ready.
- Pass = average ≥ 4 across all dimensions.

**Target:** human pass on every release checkpoint.

---

## Reporting

Each fine-tune checkpoint produces a single report:

```
Checkpoint: v0.3.2  (2026-06-15)
Eval set: v0.1 (50 examples, 6 surfaces)

Schema adherence:        96%   (target 95)  ✓
Hallucination rate:      3.2%  (target ≤5)  ✓
Section coverage:        92%   (target 90)  ✓
Empty-sections accuracy: 88%   (target 90)  ✗  ← regression
Source attribution:      81%   (target 80)  ✓
Confidence calibration:  monotonic           ✓
Latency p50 / p95:       3.1s / 7.4s        ✓
Style review:            4.2 / 5.0          ✓
```

Any ✗ blocks promotion to the next checkpoint. A regression on a
✓ metric (drop > 2 pp vs prior checkpoint) is investigated before
moving on.

---

## What we deliberately do NOT measure (yet)

- **"Insight depth"** — too subjective to score reliably. Implicit in
  the style review.
- **Coverage vs Claude head-to-head** — flatters the model when
  Claude underperforms (Claude isn't perfect either). Replaced by
  the absolute thresholds above.
- **Token efficiency / cost** — separate concern; measured by
  operations, not eval set.
- **User adoption / NPS** — product metric, not model metric. Will
  matter for the release story; doesn't gate checkpoint promotion.
