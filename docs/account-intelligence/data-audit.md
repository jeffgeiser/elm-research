# Data audit — what we can actually pull per source

What an account intelligence training example *can* contain is bounded
by what each upstream system actually returns. This audit documents
fields available today, quirks to know about, and the gaps we'll have
to live with (or fill synthetically).

Updated 2026-05-13.

---

## Salesforce (primary)

Single source of truth for accounts, opportunities, cases, contacts,
activities. Read via `lib/integrations/salesforce.py` connector +
`lib/integrations/salesforce_queries.py` helpers. OAuth client-
credentials grant; rate limits handled by the connector's TTL cache.

### Account

Helper: `query_accounts_by_ids(salesforce_ids)` and
`find_accounts(name)`.

| Field                    | Type   | Reliability | Notes |
|--------------------------|--------|-------------|-------|
| `Id`                     | string | always      | 18-char SFID. Stable. |
| `Name`                   | string | always      | Canonical name. Subsidiary disambiguation is messy — see "Zoom Video Communications" vs "Zoom Video Communications Inc.". |
| `Type`                   | enum   | mostly      | `Customer` / `Prospect` / `Partner` / `Other`. Some rows are `null`. |
| `Industry`               | string | spotty      | ~60% populated. NULL on small-customer rows. |
| `AnnualRevenue`          | number | spotty      | NULL on most rows; reps don't fill it. Don't depend on it. |
| `NumberOfEmployees`      | number | spotty      | Same — sparsely populated. |
| `BillingCountry`         | string | mostly      | Useful for routing / sovereignty. Free-text so spellings vary ("USA" / "United States" / "US"). |
| `Description`            | text   | rare        | Sometimes has useful context, often empty or stale. |
| `Owner.Name`             | string | always      | The AE on record. |
| `ZL_CID__c`              | string | sometimes   | Zenlayer Customer ID. Set when known; NULL otherwise. Phase 7.5.4 added a resolve-from-CID flow. |

### Opportunities

Helper: `query_opportunities(account_names, since_days)`. Returns
open + closed-within-window.

| Field                          | Type   | Reliability | Notes |
|--------------------------------|--------|-------------|-------|
| `Id` / `Name`                  | string | always      | Deal name is rep-authored; quality varies. |
| `StageName`                    | string | always      | Discovery / Qualify / Solutioning / Proposal / Negotiate / Verbal Commit / Closed Won / Closed Lost. |
| `Amount`                       | number | usually     | **Multi-currency, unnormalized.** An INR row can show `$200M` when the real USD is $3M. Use `Opportunity_Total_MRR__c` / `Opportunity_Total_NRR__c` instead when present. |
| `Opportunity_Total_MRR__c`     | number | mostly      | USD-normalized monthly recurring. Primary financial signal. |
| `Opportunity_Total_NRR__c`     | number | mostly      | USD-normalized non-recurring. |
| `CloseDate`                    | date   | always      | |
| `Probability`                  | number | always      | Tied to stage. |
| `Type`                         | string | sometimes   | New / Renewal / Expansion. |
| `NextStep`                     | text   | spotty      | When populated, the rep's note on the next move. High-signal when present. |
| `Description`                  | text   | spotty      | Free-text deal context. |

### Cases

Helper: `query_cases(account_names, since_days)`.

| Field             | Type   | Reliability | Notes |
|-------------------|--------|-------------|-------|
| `Id` / `CaseNumber` | string | always   | CaseNumber is the human-readable ID. |
| `Subject`         | string | always      | |
| `Status`          | string | always      | New / Working / Escalated / Closed. |
| `Priority`        | string | always      | P1 / P2 / P3 / P4. |
| `CreatedDate` / `ClosedDate` | datetime | always | |
| `Description`     | text   | usually     | Body. Sometimes blank on internal-filed cases. |
| `Contact.Name`    | string | mostly      | Who filed it customer-side. |
| `RecordTypeId`    | string | always      | **Filter on this.** Non-support records (security advisories, internal change requests, bot-filed) come through Case; the briefing flow excludes them at compute time. |

### Contacts

Helper: `query_contacts(account_names)`.

| Field        | Type   | Reliability | Notes |
|--------------|--------|-------------|-------|
| `Id`         | string | always      | |
| `Name`       | string | always      | |
| `Title`      | string | mostly      | Free-text — every variation of "VP Engineering" exists. |
| `Email`      | string | mostly      | |
| `Department` | string | spotty      | |
| `LastActivityDate` | date | mostly   | Last time anyone touched this contact in SF. Proxy for engagement. |

### Activities (Tasks)

Helper: `query_activities(account_names, since_days)`.

| Field             | Type   | Reliability | Notes |
|-------------------|--------|-------------|-------|
| `Id` / `Subject`  | string | always      | Subject is rep-authored — quality varies wildly. |
| `Status`          | string | always      | Completed / In Progress / Not Started. |
| `ActivityDate`    | date   | always      | |
| `Priority`        | string | always      | High / Normal / Low. |
| `Description`     | text   | spotty      | When present, the rep's notes on the call/meeting. |
| `Who.Name`        | string | mostly      | Contact involved. |

### SF data quality reality check

- **Activity logging is uneven.** Some reps log every call; others log nothing for months. An empty `activities` block doesn't mean the account is quiet — it might mean the rep is.
- **Multi-currency `Amount` is a hazard.** Always use the `__c` MRR/NRR fields when present, fall back to `Amount` only as a last resort.
- **Subsidiary explosion.** "Cisco" has Jasper, Webex, Saudi, etc. as separate SF Accounts. Account resolution at training time has to pick a canonical row.

---

## Confluence (internal SE knowledge)

Read via `lib/integrations/confluence.py`. Spaces:
- `GSE` — global SE team training + playbooks
- `SALES` — sales playbooks, account plays
- `DELIVERY` — post-sale runbooks

| Surface                  | What's there | Reliability |
|--------------------------|--------------|-------------|
| Account play pages       | Customer-specific notes / strategy. **Sparse** — only the top ~30 accounts have one. | Low |
| Playbook pages           | Process docs (resource checks, QRF flow, etc.). Useful for `knowledge_context` section. | High when present |
| Customer case studies    | Won deals written up. | Spotty |
| New-hire onboarding pages | Phase 7.3.x — used by `new_hire_coach`. Good signal for the `onboarding` surface. | High |

**Practical:** Confluence content is most useful as `knowledge_context`
attribution — playbooks + case studies — not raw account signal.
Most account-level signal still lives in SF.

---

## Teams (chatter)

Read via `lib/integrations/teams_graph.py` with delegated user
permission. Limited to channels the requesting user is a member of.

| Surface                            | What's there | Reliability |
|------------------------------------|--------------|-------------|
| Channel name search                | "Has the user starred a channel named after this account?" — surfaces dedicated customer channels. | Low — fewer than 10% of accounts have one |
| Message search inside found channels | Tenant-wide search of message body for the account name. Returns recent posts mentioning the account. | Medium — depends on team's chat habits |

**Caveat:** Phase 7.9.4 attempted tenant-wide chat search with app-only
permission (`Chat.Read.All`); IT did not grant the scope. We work with
delegated permission only — that limits us to channels the rep is in.

**Sources_used implication:** `teams` source attribution is honest only
when the rep has actually surfaced relevant chatter. Most briefs will
not have Teams source attribution.

---

## SharePoint (documents)

Read via Microsoft Graph search with delegated permission. Same
auth constraints as Teams.

| Surface           | What's there | Reliability |
|-------------------|--------------|-------------|
| Doc search by account name | RFPs, statements of work, technical responses, decks. | Medium |
| Site navigation   | Not used — too noisy. Search is the right shape. | n/a |

**Practical:** SharePoint is useful for handoff + QBR (where you want
"what docs exist") and less useful for meeting prep (where you want
fresh signal).

---

## News (external)

Read via `lib/integrations/news_rss.py`. Topic search across Google
News RSS feeds.

| Surface           | What's there | Reliability |
|-------------------|--------------|-------------|
| Recent headlines  | 5–10 results per account-name query. | High when account has news; many small accounts have nothing recent. |
| Article body      | Headline + snippet only. We don't deep-crawl. | Low body fidelity |

**Quality:** "Cisco" gets 100x more news than "Cook Medical" — long-tail
accounts will have a sparse or empty `external_signal.news` for most
briefs. That's honest; `empty_sections` should reflect it.

---

## Internal knowledge corpus (RAG)

Read via `lib/rag/retrieval.py` against `knowledge_chunks` (pgvector).
Backed by:
- Public Zenlayer docs (ingested via `scripts/ingest_zenlayer_docs.py`)
- Internal Confluence pages (when SE team gets around to ingesting)

| Surface           | What's there | Reliability |
|-------------------|--------------|-------------|
| Top-5 chunks by semantic similarity | Used to populate `knowledge_context` items. | High for product / process questions, low for account-specific signal |

---

## Pricing history

Read via `pricing_quotes` table. Per-rep quotes for the account.

| Field            | Type   | Reliability | Notes |
|------------------|--------|-------------|-------|
| `display_id`     | string | always      | SQUSWYYYYMMDDNNN format. |
| `created_at`     | datetime | always    | |
| `quote_name`     | string | always      | Rep-authored. |
| `customer_name`  | string | always      | |
| `contract_term`  | string | always      | 12mo / 24mo / 36mo. |
| Line items summary | derived | always   | Summed MRC / NRC per quote. |

**Practical:** Useful for `pricing_history` section. Most accounts have
zero quotes; populated reliably for the ~150 accounts with active
quote history.

---

## Prior preps

Read via `meeting_preps` table. Phase 7.9.1.

| Field         | Type   | Reliability | Notes |
|---------------|--------|-------------|-------|
| `brief`       | JSONB  | always      | Full prior brief as the model emitted it. |
| `generated_at`| datetime | always    | |
| `sources_used`| array  | always      | |

**Practical:** When a brief exists for the account, the `carryover`
section can lift live action items from it. Most accounts won't have
a prior prep.

---

## What's NOT available (gaps the synthetic data has to fill)

- **Slack** — not integrated. Zenlayer's internal comms are on Teams,
  not Slack. If the dataset has Slack-shaped examples (per the
  original plan), they're synthetic only.
- **Email content** — we have delegated Mail.Read but it's not
  currently surfaced in the brief pipeline. Could be added if a
  customer-thread surface becomes important.
- **Calendar / meeting transcripts** — Phase 7.9.2 reads
  `/me/calendarView` for the calendar-driven flow, but transcripts
  aren't pulled. Even if they were, MS Graph requires explicit
  user-meeting consent per recording.
- **Customer-side platform usage** — no telemetry from the products
  customers actually use (BMC console, ZEC API call patterns). This
  would be the single biggest signal addition; out of scope.
- **CSM-side notes outside SF** — some CSMs keep notes in OneNote /
  personal docs. Not ingestable.
- **Renewal contract details** — we don't have the contract docs
  themselves, just the SF Opportunity records. `renewal.contract_value`
  + `next_renewal_date` are inferred from SF, not from contracts.

---

## Synthetic-data implications

When generating the ~500 synthetic examples to cover surfaces beyond
meeting_prep:

1. **Stay within the format constraints above.** A synthetic
   "renewal alert" example with a contract-doc citation is unrealistic
   — we don't read contracts. Cite SF + Teams + Knowledge.
2. **Match the reliability columns.** Half the synthetic examples
   should have `industry: null`, sparse activities, partial
   `relationship_map`. Real-world data is patchy; the model has to
   handle it.
3. **Vary the empty_sections array deliberately.** A QBR with
   no Teams chatter, an escalation with no news, a handoff with no
   prior prep — these are common cases the model needs to handle
   without confabulating.
4. **Subsidiary accounts.** Include synthetic examples where account
   resolution is ambiguous — multiple matches, parent/child rows.
   This is one of the most common live failures.

The empty / sparse / messy cases are AT LEAST AS IMPORTANT for
training as the clean ones. The real-world distribution is skewed
toward incomplete data.
