# User prompt template — per-call parameters

The system prompt establishes the contract. Each user prompt instantiates **one** synthetic example by filling in the slots below. The generator script (forthcoming) walks a list of these and calls Claude once per row, validating each output against `schema.json` before saving.

Treat the parameters as **soft constraints** — Claude picks specifics inside the constraints, doesn't slavishly fill every field. The goal is varied, realistic examples, not parametric enumeration.

---

## Slot 1: surface (required)

One of: `meeting_prep` · `qbr` · `handoff` · `renewal_alert` · `onboarding` · `escalation`

Drives section emphasis per the matrix in the system prompt.

## Slot 2: account shape (required)

Pick one of these archetypes (or a hybrid you describe). Drives the financial / pipeline / case-load shape.

- **`enterprise_steady`** — $100k-300k MRR, multi-year tenure, low growth, mostly retention. Think NorthBank-style.
- **`enterprise_expanding`** — $50k-200k MRR, recent expansion deal in flight, mid-tenure. Think Vexa-style.
- **`mid_market_active`** — $20k-80k MRR, multi-deal pipeline ($50k+), 1-2 year tenure, high engagement. Think Zscaler-style.
- **`mid_market_quiet`** — $30k-100k MRR, low recent activity, stable but uncertain. Like a tenured customer between conversations.
- **`small_strategic`** — <$10k MRR existing footprint but big pipeline ($30k+) or strategic intent. Think Maersk-style.
- **`stalled_at_risk`** — $20k-100k MRR, utilization dropping, champion churn, ticket-volume uptrend. Think Polaris-style.
- **`fresh_handoff`** — any size, mid-transition between reps (the rep change is the operative context). Think MeridianAir-style.
- **`incident_acute`** — any size, mid-P1 (incident is the operative context). Think ArboBio-style.

## Slot 3: edge cases to include (0-3)

Pick zero to three of these for deliberate distribution coverage. Some won't fit naturally with some account shapes — skip rather than force.

- **`subsidiary_explosion`** — multiple SF rows for the same logical company; brief picks one and notes the disambiguation
- **`null_industry`** — Account.Industry comes back null; brief handles gracefully
- **`mis_categorized_industry`** — Industry tagged incorrectly in SF (e.g. a shipping firm tagged "Financial services")
- **`multi_currency_amount`** — an opp has Opportunity_Total_MRR__c set + a weird raw Amount that would mislead if used
- **`sparse_activity`** — last activity logged >30 days ago; brief honestly notes this
- **`activity_overlogged`** — 50+ activities in 45 days but most have null Who; brief filters appropriately
- **`bot_filed_case_noise`** — 50+ open cases dominated by RBL/abuse-bot tickets; brief separates noise from signal
- **`p1_open_long`** — a P1 case has been open >30 days; brief addresses it in must_address
- **`closed_lost_recent`** — a significant Closed Lost opp in the last 90 days; brief surfaces the loss-reason gap
- **`champion_churn`** — primary contact departed in last 90 days, replacement low-engagement
- **`non_latin_contact`** — a CN/JP/AR/KR character name in SF; brief handles without choking
- **`duplicate_contacts`** — same person appears twice in SF Contacts with slightly different titles; brief notes the duplicate
- **`dc_code_gap`** — pricing_historical references a DC code not in pricing_dc_cache; brief flags in talking_points
- **`null_priority_cases`** — bot-filed cases come back with null Priority; schema handles via nullable enum
- **`audit_cycle`** — InfoSec / FFIEC / SOC2 audit is the operative customer rhythm; case + activity reflect it
- **`public_announcement_anchor`** — the customer made a public statement (earnings call, press release) that anchors a strategic ask
- **`utilization_drop`** — measured usage is below contract capacity; renewal-risk signal
- **`opp_name_typo`** — an opp name in SF has a minor data-entry error (e.g. "Q22926" instead of "Q22026"); brief notes in talking_points
- **`vp_newly_visible`** — a senior executive appears on the thread for the first time; brief flags the implication

## Slot 4: source mix (optional)

If you want to constrain which sources the brief cites — useful for controlling distribution.

- **`sf_only`** — only SF sources. Realistic for many briefs; news/Teams empty.
- **`with_news`** — include relevant external_signal.news.
- **`with_teams`** — include Teams chatter (rare; only ~10% of accounts).
- **`with_prior_prep`** — assume a prior brief exists; populate carryover.
- **`rich`** — wide source mix (SF + news + Teams + knowledge + prior_prep).
- **`minimal`** — SF account + opps only (low-data customer baseline).

If omitted, Claude picks a realistic mix for the surface + account shape.

## Slot 5: account hint (optional)

A 1-line nudge on what kind of customer to make up. Examples:

- "Streaming media customer expanding into LATAM"
- "Regional bank with strict regulatory audit cycle"
- "Logistics firm with shrinking traffic on legacy IPT"
- "Fintech startup in early Discover stage, no Zenlayer footprint yet"

Skip this slot to let Claude pick. Forcing it constrains diversity — only use when you need a specific industry covered.

---

## Example user prompts

These are what the generator script feeds Claude. Each produces one example.

**Example A — fill an underrepresented surface**

```
Generate one synthetic example with these parameters:

- surface: handoff
- account_shape: mid_market_active
- edge_cases: ["champion_churn", "duplicate_contacts"]
- source_mix: sf_only
- account_hint: SaaS analytics customer, west-coast based
```

**Example B — stress the data-quality dimension**

```
Generate one synthetic example with these parameters:

- surface: meeting_prep
- account_shape: enterprise_steady
- edge_cases: ["non_latin_contact", "mis_categorized_industry", "activity_overlogged"]
- source_mix: with_news
```

**Example C — escalation under audit pressure**

```
Generate one synthetic example with these parameters:

- surface: escalation
- account_shape: enterprise_expanding
- edge_cases: ["p1_open_long", "audit_cycle", "vp_newly_visible"]
- source_mix: rich
- account_hint: Healthcare-adjacent, HITRUST audit in flight
```

**Example D — minimal data, small strategic customer**

```
Generate one synthetic example with these parameters:

- surface: qbr
- account_shape: small_strategic
- edge_cases: ["sparse_activity", "closed_lost_recent"]
- source_mix: minimal
```

---

## Batch-generation strategy (for the v0.1 dataset)

Aim for ~100 synthetic examples per surface to hit ~600 total. Suggested split:

| Surface         | Count | Notes                                                  |
|-----------------|-------|--------------------------------------------------------|
| `meeting_prep`  | 60    | Plus the 3 hand-built real ones → 63 total            |
| `qbr`           | 100   | Plus 1 hand-built synth → 101                          |
| `handoff`       | 100   | Plus 1 hand-built synth → 101                          |
| `renewal_alert` | 100   | Plus 1 hand-built synth → 101                          |
| `onboarding`    | 100   | Plus 1 hand-built synth → 101                          |
| `escalation`    | 100   | Plus 1 hand-built synth → 101                          |
| **Total**       | **560** | **+ 8 hand-built = 568**                              |

Within each surface, balance across `account_shape` archetypes (target ~12 per shape per surface) and ensure each `edge_case` appears in at least 5% of examples. The generator script enumerates the matrix and feeds it through this prompt one example at a time.

The first 50 examples after the 8 hand-built ones should be hand-reviewed before generating the rest — that's the eval-set freeze point per `evaluation.md`. Generation drift is real; one human pass on the early batch catches the systematic biases before they propagate to 500 examples.
