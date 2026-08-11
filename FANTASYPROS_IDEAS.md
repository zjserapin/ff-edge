# FantasyPros integration — brainstorm

**Written 2026-08-10, twelve days before the draft.** Ideas only; nothing here is
built. Two empirical findings come first because they change what the API key is
*for*.

---

## Finding 1 — consensus ECR, its dispersion, and seven years of history are already free

`nflverse.ff_rankings("all")` is not a current snapshot. It is an archive:

```
1,806,760 rows    359 distinct scrape_dates    2019-12-27 → 2026-08-07
```

| year | snapshots | of which July/Aug |
|---|---:|---:|
| 2020 | 53 | 9 |
| 2021 | 64 | 10 |
| 2022 | 53 | 9 |
| 2023 | 54 | 10 |
| 2024 | 51 | 9 |
| 2025 | 51 | 9 |
| 2026 | 32 | 6 (so far) |

It carries `ecr`, `sd`, `best`, `worst`, `rank_delta` — and `ecr_type` includes
**`rsf`, redraft superflex**, which is this league's actual format:

| rsf preseason (July/Aug) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| snapshots | 10 | 9 | 10 | 9 | 9 | 6 |

`sd` is 100% populated on the redraft board. `ff_rankings("draft")` — what
`peek.market_disagreement` already reads — is the single latest snapshot only
(`2026-08-07`), which is why this archive was easy to miss.

**Consequence: do not spend one call of the budget on consensus ranks.** Five
full superflex preseasons of dated, dispersion-bearing consensus are sitting in
the cache already. It also means the idea you already ruled out — paste ECR onto
the board — is ruled out twice: it is not just shallow, it is redundant.

Caveat: `rsf` starts in 2021, so superflex has five historical preseasons, not
six. The 1QB board (`ro`) has six (2020+). And `pos` in this archive includes
IDP, so filter positions the way `peek.market_disagreement` does.

## Finding 2 — what the API actually adds, and the one constraint that shapes everything

Genuinely additive over nflverse:

| payload | in nflverse? | notes |
|---|---|---|
| **Per-expert individual rankings** | **no** | the only truly unique bulk payload |
| Projections, full stat lines (weekly + ROS) | no | components matter, totals don't — see §Tier 3 |
| Injuries w/ practice-report probabilities, by season/week | no | structured, dated, weekly-resolving |
| News by category / player / recency | partially (`news.py`) | filterable, structured |
| Expert tiers | no | human tier breaks, vs. your empirical ones |
| ADP by scoring × team count × format | FFC covers this | but a *second* market — see §Tier 5 |
| Consensus ECR + sd/best/worst | **yes, with history** | do not spend calls here |

**The constraint that matters more than 50/day: historical and bulk access is
Commercial-tier only.** At the non-commercial tier the API is a **forward-only
feed**. You cannot backfill it, ever, at this price.

### The budget principle that follows

50/day is ~18,000 calls a year. Calls are scarce *per day*, not in total. So:

> **A call is well spent if and only if the data is perishable** — unrecoverable
> if you don't capture it today. Never spend one on something nflverse will hand
> you later for free.

Per-expert rankings on a given date are perishable: FantasyPros overwrites them
and the free tier has no history. Season stats are not perishable. So the
highest-value standing use of the key is a small daily snapshot, append-only with
`scrape_date`, deduped — **exactly the pattern `claims.py` already uses.** Three
years of that and you own a historical per-expert corpus that cannot be bought at
the non-commercial tier.

Rough allocation: ~6-8 calls/day for the perishable snapshot (positions ×
format), ~40 held back for draft-day burst and ad-hoc work.

---

## Tier 1 — free today, no key, backtestable over 5-6 preseasons

These need no API key at all. They are first because they are cheap, immediate,
and two of them bear on the draft.

**1. Is ECR a stronger "price" than FFC ADP?** The project's central null —
`breakout` AUC 0.528 vs 0.493, `projection` rho 0.497 vs 0.494 — is measured
against ADP. If ECR is the better baseline, every one of those nulls has been
scored against a soft benchmark and the honest gap is worse than recorded. If ADP
wins, then drafter *behaviour* beats expert *opinion*, which is a finding worth
writing down. Five superflex preseasons, season-forward only.

**2. `ECR − ADP` as a second opinion that costs nothing.** Two different markets:
what experts say, and what drafters do. The residual is a disagreement signal not
derived from your model, so it is a genuine companion to `valuation.value_gap`
rather than a restatement of it. Backtestable across the same five preseasons —
which is more than the ledger or the claims layer will have for years.

**3. Dispersion → payoff asymmetry.** This is the one genuinely open question the
handoff raised: the spread around the ADP curve is over five times the step
between ranks, and nothing here has tested whether *landing high in your tier's
distribution* is predictable. Expert dispersion is a forward-looking variance
proxy, and it is free and historical. Regress `sd` (and `worst − best`) at draft
time on the **absolute** residual `|finish − implied finish|`, not the signed
one. A variance model needs no per-expert data and no skill estimates. Testable
this week.

**4. Preseason drift.** Nine to ten snapshots per July/August × six years, and
`rank_delta` already ships. Does a player rising through late August keep rising,
or is chasing it buying the top? Cheap, and it directly informs how much to trust
the 2026 board on the 22nd versus the 7th.

**5. Do human tiers break on round boundaries where the points don't?** Handoff
finding #4 measured running back as a *slope* — 0.68 points per rank, range
0.50-0.98 — and found that **RB12 and RB24 are not in the data.** Those are
exactly where a 12-team round boundary falls. If ECR and published tiers cliff
there anyway, that is a quantifiable market bias with an actionable corollary:
players sitting just below a false break are systematically cheap. `landscape.
tier_breaks` already computes your side of the comparison.

## Tier 2 — needs the key; per-expert granularity

**6. Bimodality, which `sd` cannot express.** An `sd` of 8 means one of two
completely different things: everyone is mildly unsure, or sixty experts say RB14
and seventy say RB30. The second is a *discrete* question — a role battle, a
suspension, a knee — and a mean is the worst possible summary of a coin flip. Fit
two clusters (or a dip test) on the per-expert ranks. This is also the cheapest
available estimator of the **two-humped payoff distribution** that idea #3 is
groping toward with a single variance number.

**7. Is ECR 130 opinions or five?** If experts cluster — and they will, they read
each other — then the effective number of independent opinions is far below 130,
and consensus confidence built on `sd/√n` is overstated. Estimate it from the
eigenvalue spread of the expert × player rank-correlation matrix. This is a
methodological correction in the same family as "the test was checking half the
join": the number isn't wrong, the thing it's assumed to mean is.

**8. Who moves first.** Snapshot per-expert ranks daily and find the experts
whose revisions *lead* consensus. Their current position is then a leading
indicator of next week's ECR, and ADP follows ECR. The reason this is the most
practical per-expert idea: **it predicts the market, not the player**, so the
feedback loop is days rather than seasons. It is the one per-expert question that
can be answered inside a single year.

**9. Per-expert skill, measured on deviations rather than accuracy.** The wrong
question is "which expert is accurate" — they all sit roughly where the market
sits. The right one is "whose *departures from consensus* carry information":
regress `expert_rank − ecr` on `finish − implied finish`, per expert. Then build a
skill-weighted consensus, which is something ECR structurally cannot be, because
ECR is equal-weighted by definition.

**Be honest about the timeline on this one.** 130 experts against one season of
labels is noise, and no amount of bootstrapping fixes n=1. This is corpus
building — log from now, judge around 2029 — and it should carry the same framing
`claims.py` already uses for itself: season one is labeled-data collection, the
weights are priors, not fitted parameters. **Do not start this in the next twelve
days.**

## Tier 3 — projections, used as components and never as a ranking

`ANALYSIS_SPEC` §5 already refused to invent a projection, and projected fantasy
*points* would just be another ADP-correlated ordering. The value is the stat
line underneath it.

**10. The projection budget audit.** My favourite idea here, and the cheapest
real edge on the list. Consensus projections are built bottom-up, one player at a
time, by people who are not reconciling them — so they are **not team
constrained**. Every offense has a finite budget: roughly 1,000 plays, ~550 pass
attempts, ~450 carries, ~35 touchdowns.

    aggregate projected targets / carries / TDs  →  team level
    compare against a top-down budget from expected.team_environment
      (implied team total, spread) + historical pace
    the residual is the market's arithmetic error, localized to one offense

Then name the overflow: inside an inflated team, which player is not getting the
volume he's projected for? Cross that against `path_score` and quality
percentile, which you already have. Label-free, immediate, roughly 4-8 calls, and
it leans on the finding the handoff says outweighs board edges anyway — 45.3
points per point of implied team total.

**11. Decompose ADP into implied volume × implied efficiency.** `valuation.py`
rests on a thesis: volume is what the market sees, efficiency is its blind spot.
But right now the market's beliefs are *inferred* from one ADP number. Projected
stat lines make them explicit — projected targets and carries are the market's
volume forecast, stated. So you can compute market-implied YPT / YPC and compare
against the player's own stability-weighted efficiency, and attack **only** the
half you've measured as repeating (WR rho +0.502, RB +0.330, QB +0.367). That is
a sharper and more defensible subtraction than percentile-minus-percentile.

## Tier 4 — injuries into the claims ledger

**12. A structured, zero-LLM, weekly-resolving claim source.** `claims.py` today
extracts from prose with an LLM and cannot be backtested; its own docstring says
season one is collection and judgement waits until next August. The injuries
endpoint changes the arithmetic:

- **Structured and dated already** — no extraction step, no `ANTHROPIC_API_KEY`,
  consistent with depth-chart claims already working without one.
- **Practice participation is an ordinal ladder** (DNP → limited → full) with a
  well-defined conditional: practice status → game status. That has a **knowable
  base rate within a single season.**
- Most importantly, it **resolves weekly** against `promotion.weekly_trust`
  rather than once per season. That is what lets you validate the ledger's
  *scoring machinery* — tier × specificity × novelty × recency — during 2026
  instead of waiting for 2027.

The ledger's slowest property is its resolution horizon. This is the one feed
that shortens it.

## Tier 5 — draft mechanics

**13. Dispersion-aware survival and sim noise.** `simulate.py` draws opponent
picks from ADP with noise scaled by each player's observed scoring `stdev`. But
your leaguemates aren't sampling from a scoring distribution — they are reading
rankings. Per-expert rank distributions are a better-shaped estimate of "where
will he actually go," and the bimodal cases (#6) are where it bites: bimodal
expert opinion implies a **bimodal draft outcome** — either someone in your
league read the bullish take or nobody did. A unimodal survival curve prices that
player wrong in both tails.

**14. FP ADP against FFC ADP, same format.** Two independent superflex markets
priced on the same roster shape. Where they disagree, a player's cost depends on
which site your leaguemates use. `profiles.py` exists precisely because a format
and the market pricing it must not drift apart — this is the natural test of how
much the *choice of market* is worth. In a ten-team league where you know all ten
people, that is not academic.

## Tier 6 — in-season 2026

**15. The waiver layer the sim explicitly excludes.** `ANALYSIS_SPEC` Track 3
states the limitation up front: it sims drafting, not in-season management, and
waivers and trades are a large share of real outcomes. Weekly and ROS projections
are exactly that missing input, and in-season cost is 1-2 calls a week — trivial
against the budget. Not a draft-day project; the obvious September thread.

---

## Engineering notes, so none of this fails silently

- **A call counter that raises at 50**, rather than degrading into empty frames.
  A rate limit that returns `[]` looks exactly like a player having no data, and
  this repo's whole ethos is refusing that class of bug. Same reasoning as
  `profiles.resolve()` raising on an unknown name.
- **Cache-first, with fixtures for tests.** 14 of 16 test files touch the cache;
  `pytest` must never spend a call.
- **Append-only parquet keyed on `scrape_date`**, deduped on write, following
  `claims.py`. The archive is the asset — a snapshot you overwrite is a call
  wasted.
- **IDs**: `fantasypros_id` is already in `ff_playerids` and `ids.crosswalk()`.
  Prefer it to names — `ids.py` already records that FantasyPros writes "Marvin
  Harrison Jr." where nflverse writes "Marvin Harrison".
- **Team codes both sides, every time**, including bare equality checks, per
  `CLAUDE.md`. A new feed is a new chance to meet Arizona.
- **Key from the shell, never committed** — `FF_EDGE_FP_API_KEY`, and remember
  shell exports don't reach the Bash tool.
- **Verify per-expert access with exactly one call** before designing on it. The
  docs describe a per-expert rankings endpoint but are vague about which tier
  exposes it, and every idea in Tier 2 dies without it.

## If it were my twelve days

Ranked by whether it can change a pick on 2026-08-22:

1. **#10, the projection budget audit** — label-free, ~8 calls, and it names
   specific players whose projected volume doesn't exist. Highest draft-day value
   per unit of work on this page.
2. **#2, `ECR − ADP`** — free, no key, and gives the Board a second opinion that
   isn't derived from ADP, which is the exact hole finding #1 of the handoff
   describes.
3. **#5, false tier breaks** — free, and it converts an existing finding (the
   slope, not the staircase) into a buy rule.
4. **#12, injuries into the ledger** — set it running before the season so the
   corpus starts accumulating; the analysis comes later.

Everything in Tier 2 is a multi-year corpus play. Start the *snapshotting* now
because the data is perishable; start the *modelling* never before 2027.
