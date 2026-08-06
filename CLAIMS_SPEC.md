# Claims ledger — Phase 2 spec (agreed 2026-08-03, not yet built)

The contextual layer. Scope was settled in conversation: **automated ingestion**
(Zach confirmed manual logging is not realistic), free sources only, sporadic
human check-ins. This file is the contract for the build so the next session
starts from decisions, not re-derivation.

## The one job

The measured finding that shapes everything: **the model cannot predict who
gets a bigger role** (vacated targets r = −0.04, prior quality 0.02 against
next-season role growth), and the promotion screen therefore takes "role
growing" as an *input*. The ledger's only job is producing that input with a
confidence grade: **estimate role change per player, from reported claims.**

Not sentiment. Not buzz. Explicitly excluded: performance takes ("looked
explosive"), rankings debates, trade speculation without a transaction. Those
are the high-volume/low-value tier and they are also what ADP already prices.

## The atom is a claim, not a player

One row per claim:

| field | what |
|---|---|
| `claimed_on` | date the claim was published |
| `player_id` | gsis_id via the ids layer; name + team as reported |
| `claim_type` | `depth_chart` / `first_team_reps` / `coach_usage` / `injury_teammate` / `departure` / `role_change_observed` |
| `direction` | role growing / shrinking |
| `source` | outlet + author, normalized |
| `source_tier` | 1 beat writer with access, 2 national reporter, 3 aggregator (hand-assigned to start; coarse on purpose) |
| `specificity` | falsifiable and concrete ("70% of first-team reps Wednesday") vs vibes ("more involved") |
| `novelty` | first report vs repetition — dedupe by claim, not by mention; 40 aggregators quoting one beat report is one claim |
| `resolves_by` | date + condition where falsifiable, null otherwise |
| `resolved` | did the claimed usage materialize (filled later, from `weekly_trust`) |
| `adp_at_claim` | player's ADP the day the claim landed, from the snapshot history |
| `url`, `quote` | audit trail — every score must decompose back to its claims |

Claim score = tier weight × specificity × novelty × recency decay. Aggregated
per player into the role-growing flag + grade the Screen tab consumes.

## Sources (free, in build order)

1. **Sleeper trending adds/drops** — free API already wrapped in `sleeper.py`.
   Not a claim source; a *detector* that says "the market noticed something,
   find the claim." Divergence between trending and no-claim is itself signal.
2. **nflverse depth charts** — weekly, structured, no extraction needed.
   Depth-chart moves are machine-generated claims of type `depth_chart`.
3. **Google News RSS per team query** — the beat-report firehose. This is the
   one that needs LLM extraction.

## Extraction

Anthropic API per the global stack conventions: modular client class, model in
`config.py`, prompts versioned in `src/prompts.py`, never inline. Haiku-class
model for extraction (high volume, structured output), with the schema above as
the tool/output contract. Extraction failure mode to design against:
**hallucinated specificity** — the model upgrading "could see more work" into
"named the starter". The prompt must force a verbatim `quote` field, and the
claim is only as specific as the quote supports.

## Resolution and validation

- A resolvable claim resolves against `promotion.weekly_trust`: did the
  player's share move within k weeks. That builds the source-grade history —
  Brier-style, per source, converging over seasons, so tiers stay coarse.
- **The layer cannot be backtested** — there is no historical claims corpus.
  Season one *is* the labeled-data collection. Log everything, resolve
  everything, judge the layer next August.
- **Already-priced check:** a claim's value is only the part not in price.
  `adp_at_claim` against the rolling snapshots (`half-ppr/10`, `ppr/12`,
  `2qb/10` — all accumulating daily) is what makes "was it priced" answerable.
- August coachspeak gets a standing discount: track the hit rate of
  "we want to get him more involved" as its own claim type before trusting it.

## Failure modes, ranked

1. Double counting news that ADP already moved on — the timestamp join is the fix.
2. Vibes with extra steps — if a player's flag can't decompose to quoted claims, delete it.
3. Hallucinated extraction — verbatim-quote contract, spot-check sample weekly.
4. Coverage bias — no claims about a boring veteran starter is bullish stability, not missing data.
5. Source-grade overconfidence — a season gives a handful of resolvable claims per source; keep three tiers, resist decimals.

## Cadence

Daily pull during camp/season (bootstrap step or cron), append-only parquet in
`data/` (claims are data, but they contain no league secrets — verify nothing
personal lands in the repo before any commit; `data/` stays gitignored
regardless). Screen tab reads the ledger and pre-fills the "role growing" box
with flagged names, grades attached; the user can override in either direction.
