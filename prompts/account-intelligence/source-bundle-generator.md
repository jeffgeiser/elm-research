# Source bundle generator

Generate a realistic raw source bundle representing data pulled from Zenlayer's
production systems for one account. This is the INPUT the account intelligence
model will receive at serving time — not the brief itself.

The user prompt supplies: `surface`, `account_shape`, `edge_cases`, `source_mix`.
Use these to shape the data realism and sparsity.

## Output

A single fenced ` ```json ``` ` block containing one JSON object matching this schema:

```json
{
  "surface": "<meeting_prep|qbr|handoff|renewal_alert|onboarding|escalation>",
  "pulled_at": "<ISO timestamp — a realistic 2026 date>",
  "sf_account": {
    "id": "<18-char 001... SF Account ID>",
    "name": "<fictional company name>",
    "type": "<Customer|Prospect|Partner>",
    "industry": "<industry string — or null if edge_cases includes null_industry>",
    "cid": "<4-6 digit Zenlayer CID as string, e.g. '6929', '32600'>",
    "owner_name": "<Zenlayer rep full name>",
    "billing_country": "<ISO country code>"
  },
  "sf_opportunities": [
    {
      "id": "<18-char SF ID>",
      "name": "<rep-authored name, e.g. 'Acme: KSA IPT , Q32026'>",
      "stage": "<SF stage>",
      "amount_usd": "<dollar string or null>",
      "mrr_usd": "<number or null>",
      "close_date": "<YYYY-MM-DD>",
      "probability": <0-100>,
      "next_step": "<string or null>",
      "type": "<New Business|Renewal|Expansion>"
    }
  ],
  "sf_contacts": [
    {
      "id": "<SF Contact ID>",
      "name": "<Full Name>",
      "title": "<Job Title or null — null ~40% of the time>",
      "email": "<firstname.lastname@domain>",
      "department": "<string or null>"
    }
  ],
  "sf_activities": [
    {
      "id": "<SF Task/Event ID>",
      "type": "<Task|Event>",
      "subject": "<subject text>",
      "date": "<YYYY-MM-DD>",
      "description": "<text or null — often null>",
      "who_name": "<contact name or null>"
    }
  ],
  "sf_cases": [
    {
      "case_number": "<8-digit zero-padded, e.g. '01765221'>",
      "id": "<SF Case ID>",
      "subject": "<subject text>",
      "status": "<Open|Closed|Pending Customer>",
      "priority": "<P1|P2|P3|P4|null — null is common>",
      "opened_at": "<ISO timestamp>",
      "closed_at": "<ISO timestamp or null>",
      "contact_name": "<string or null>"
    }
  ],
  "teams_messages": [
    {
      "channel": "<e.g. '#apac-accounts'>",
      "sent_at": "<ISO timestamp>",
      "sender": "<name>",
      "text": "<message text>"
    }
  ],
  "news": [
    {
      "headline": "<headline>",
      "source": "<publication>",
      "published_at": "<ISO timestamp>",
      "url": "<URL or null>",
      "snippet": "<≤200 char excerpt>"
    }
  ],
  "pricing": {
    "recent_quotes": [
      {
        "display_id": "<SQUSW... format>",
        "created_at": "<YYYY-MM-DD>",
        "summary": "<what the quote covered>",
        "amount": "<dollar string>"
      }
    ],
    "period_breakdown": [
      {
        "period": "<Q1 2026>",
        "product_line": "<BMC|IPT|ZEC|SDN|Colocation|CDN|IPLC|MHS|Cloud Connect>",
        "dc_code": "<zone, e.g. DFW-A>",
        "mrc_usd": <number>,
        "nrc_usd": <number>,
        "note": "<string or null>"
      }
    ]
  },
  "knowledge_snippets": [
    {
      "title": "<doc title>",
      "url": "<URL or null>",
      "snippet": "<≤300 char excerpt>"
    }
  ]
}
```

## Realism rules

- **Zenlayer products only:** BMC, ZEC, IPT, Cloud Connect, SDN, Colocation,
  CDN, IPLC, MHS. DC codes like DFW-A, HKG-B, JED1, SGP-A, LON1A.
- **Sparse by default.** Many fields are null. `title` on contacts null ~40%.
  `description` on activities often null. `priority` on cases often null.
- **Source mix drives what's populated.** `sf_only` → empty `teams_messages`,
  `news`, `knowledge_snippets`. `rich` → all sections populated. `minimal` →
  even SF data is thin. `with_news` → 1–3 news items.
- **Realistic identifiers.** SF Account IDs start `001`, 18 chars. Case numbers
  8-digit zero-padded. CIDs are 4–6 digit strings with no prefix.
- **Messy opp names.** Real SF data has imperfect punctuation: spaces before
  commas, mixed capitalisation. If edge_cases includes `opp_name_typo`, add a
  visible typo in one opp name.
- **Subsidiary explosion** (if in edge_cases): include 2–3 subsidiary entity
  names in opp names and contacts, with distinct SF IDs.
- **Amount values are messy:** $8,787 MRR not $10,000. `mrr_usd` is the raw
  number (`8787`), `amount_usd` is the display string (`"$8,787 MRR"`).

No prose before or after the code block. The bundle is the output.
