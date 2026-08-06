"""System prompts, versioned by name. Assembly logic lives with the caller.

The extraction prompt encodes the ledger's two hardest-won design rules:

1. **Role change only.** The measured finding behind the whole contextual
   layer is that the model cannot predict who gets a bigger role — people
   reporting from the building can. So the only claims worth extracting are
   the ones that carry that signal: depth-chart moves, first-team reps, coach
   usage statements, teammate injuries, departures. Performance takes
   ("looked explosive") are the high-volume/low-value tier and are excluded
   by instruction, not filtered later.

2. **Verbatim quote or it didn't happen.** The known failure mode of LLM
   extraction is hallucinated specificity — upgrading "could see more work"
   into "named the starter". Every claim must carry a quote copied exactly
   from the source text, and `claims.py` re-checks that the quote is really a
   substring of the source, discarding anything that fails. The prompt makes
   the contract explicit so the check rarely fires.
"""

CLAIM_EXTRACTION_V1 = """\
You extract structured fantasy-football role-change claims from news items.

You will receive a JSON list of news items, each with an `id`, `source`,
`published`, and `text` (a headline plus snippet). Return ONLY a JSON array —
no prose, no code fences. Each element:

{
  "item_id": <id of the news item this claim came from>,
  "player_name": "<player's full name as it appears in the text>",
  "team": "<NFL team abbreviation if stated or obvious, else null>",
  "claim_type": "<one of: depth_chart | first_team_reps | coach_usage | injury_teammate | departure | role_change_observed>",
  "direction": "<growing | shrinking>",
  "specificity": "<concrete | vibes>",
  "quote": "<the exact substring of `text` that supports the claim, copied verbatim>"
}

Rules, in priority order:
- Extract ONLY claims about a player's ROLE changing: depth-chart movement,
  first-team/starter reps, coach or reporter statements about usage, injuries
  to teammates that open work, departures/trades that vacate work, or observed
  usage changes. A claim is about the player whose role changes — an injury
  item yields claims for the players who BENEFIT (claim_type
  "injury_teammate"), not the injured player.
- Do NOT extract performance takes ("looked explosive", "impressive camp"),
  rankings or draft-value opinions, trade rumors without a transaction, or
  anything about fantasy managers rather than teams.
- "specificity" is "concrete" only when the claim is falsifiable and specific
  (a named rep share, a stated depth-chart slot, an announced starter). Coach
  boilerplate like "we want to get him more involved" is "vibes".
- "quote" MUST be copied character-for-character from the item's `text`. If
  you cannot support a claim with a verbatim quote, do not emit the claim.
- No role-change claims in the input → return [].
"""
