# Dashboard refactor — spec

**Status:** agreed 2026-08-08. All three open items resolved — Players tab stays
whole, breakout and projection both stay at full length, the `Board` tab keeps
its name. Building from here.
**Written:** 2026-08-08. **Draft day:** 2026-08-22 (14 days).

---

## What this is for

Two answers from the interview set everything else:

**Purpose — research archive and draft day, at equal billing.** Not a draft-day
tool with the research bolted on, and not a research notebook with a board
attached. Both are the product.

**Audience — Zach, plus people evaluating his work.** This is the answer that
does the most work in this document. It means the honest negative results are
not overhead to be trimmed; they are the most credible thing in the app. It also
means every chart has to survive being read by someone with no context, which
raises the bar on captions rather than lowering it.

The consequence worth stating plainly: **this is not a "cut half the app"
refactor.** The problem is not that there is too much, it is that the shape does
not match what the project turned out to be. Three specific mismatches:

1. `src/board.py` — the actual draft board, fully built and tested — has **no UI
   at all**, two weeks before the draft.
2. The tab named **"Board" is not the board.** It is `valuation.py`, the
   quality-vs-price scatter. The name is already taken by the wrong thing.
3. The **Players tab is 581 lines** holding four unrelated questions, so the
   things that belong together are apart and the things that don't are together.

---

## Information architecture

### Decided

**A new Draft Day tab, absorbing Screen.** The promotion screen and claims
ledger stop being their own destination and become the tie-break layer inside
the draft view, which is the only moment they are read under time pressure.

**The Players tab stays whole.** Trim inside it; do not split it. (Chosen
explicitly over a Players/Research split.)

**The `Board` tab keeps its name.** Raised and decided: `valuation.py` stays
"Board". The consequence to hold in mind while building — the tab called Board is
the value scatter, and `src/board.py` surfaces under **Draft Day**. In code the
existing `_tab_board` therefore stays pointed at valuation, and the new function
is `_tab_draft_day`. Anything else invites an edit to the wrong tab.

Resulting tab strip — six tabs, same count as today:

```
Draft Day | Landscape | Players | Strategy | Board | Glossary
```

Screen is gone; Draft Day is new. **That is the entire structural change.**

---

## Tab 1 — Draft Day (new)

Everything here already exists in `src/board.py` and is currently unreachable.
This tab is **surfacing, not new analysis**, which is why it is first: highest
value, lowest risk, and it is the one with a deadline.

Built around a single control: **which pick am I on?**

### 1.1 Your picks

From `board.picks`. Round, pick number, who it came from, whether a keeper
consumed it, usable or not.

> 17 picks owned, 15 usable. Five inside an eight-pick window:
> `R5#44, R5#47, R5#48, R5#50, R6#51`. No R10, no R13.

That clustering is a real planning constraint and is currently invisible.

### 1.2 The board

From `board.build()["players"]`. Columns, in reading order:

```
board_rank · name · pos · adp → adj_adp · exp_pick · PAR · tier · env_swing
```

**`adp → adj_adp` needs its own caption**, because it is the least obvious and
most useful column on the page. Public ADP is priced in leagues where nobody is
kept; every keeper here sits inside the top 150, so everyone behind them moves
up. Any mock that ignores keepers shows players ~15 picks late.

`exp_pick` is the column to compare against your own picks — `adj_adp` is a
selection index, and comparing that to a pick number double-counts the
adjustment. The UI must not let those two be confused; label them, don't just
show them.

### 1.3 Available at your next pick

From `board.targets`. Best PAR among players who plausibly last, using FFC's
draft-slot dispersion. Two players at the same ADP with different dispersion are
completely different planning problems, and this is the only view that says so.

### 1.4 Context flags

From `board.context_flags`. Pairs where the environment gap exceeds the board
edge — the picks where the board is not really the deciding input.

> The live example: **McBride over Loveland is a 2.4-point board edge against a
> 42.8-point environment gap**, because Arizona is the lowest-priced offence in
> the NFL.

Caption must carry the caveat honestly: this is an **upper bound**, because some
of the discount is already in the ADP level and `env_swing` cannot know how much.

### 1.5 Promotion screen + claims (absorbed from Screen)

Tie-breaks inside a tier, which is exactly what they are for. Trimmed from 8
tables to the screen output plus the claims ledger; the per-criterion diagnostic
tables move behind an expander.

---

## Tab 2 — Landscape (mostly keep)

The praised tab. Five charts, one table, and the PAR-over-time charts were
called out specifically. **Keep the charts. No structural change.**

Two edits only:
- Add a caption noting the league's own rules changed in 2023, 2024 and 2026, so
  history plotted under current scoring is deliberate but not what was played.
- The superflex change means past QB value in particular is not comparable.
  `scoring.scoring_history` already computes this and it is not surfaced.

---

## Tab 3 — Players (keep whole, trim inside)

581 lines, four questions, in this order: comparables → coverage → stability →
breakout → projection → rookies. The order is right — stability comes before the
backtest deliberately, because whether a measurement repeats at all has to be
settled before an AUC means anything. **Keep the order. Cut the surface.**

| section | now | after | what changes |
|---|---|---|---|
| Comparables | 97 lines, 3 tables | keep | It answers "who does this player look like", which is a draft question. |
| k-means removal note | full `st.info` | **keep** | A deleted feature with a measured reason is exactly what the evaluator audience is here for. Silhouette 0.19-0.29 is a partition of a continuum. |
| Stability | 92 lines, 2 charts | **keep** | Praised, and it is the precondition for everything after it. |
| Breakout + Projection | 246 lines, 10 tables, 3 charts | **keep at length** | Decided. The detail is the point for the evaluator audience. |
| Rookies | 146 lines | trim tables | Keep the chart, keep the finding. |

**So the Players tab barely changes**, and that is the correct outcome rather
than a failure to refactor. Once the negative results stay at length and the tab
stays whole, the only edit is trimming a few rookie and comparable tables. The
refactor is therefore almost entirely *additive*: build Draft Day, retire Screen.

### Not merging breakout + projection — but they need one shared caption

Both modules ask one question with two framings, and presenting them as two
unrelated sections hides that. They stay two sections at full length; a short
lead-in above the first one states the shared question and the two answers, so a
reader who stops after four sentences still has the finding:

> **The question.** Predicting who scores points is easy — usage persists.
> Beating the *price* is the whole game, because the market read the same box
> scores.
>
> **Binary framing** (`breakout.py`): did he finish inside 60% of his ADP
> positional rank? AUC 0.513 stratified, **+0.037 over ADP-only, CI [-0.011,
> +0.087]** — covers zero.
>
> **Continuous framing** (`projection.py`): predict percentile of positional
> finish, so RB4 and RB40 are not the same answer. Spearman **0.4946 vs 0.4948
> for price alone. Delta -0.0002, CI [-0.010, +0.011].**
>
> Using the sample better bought precision, and the precision confirmed the
> null. It did not soften it.

Three findings that must survive the trim, because each is a real result:

- **The first label design was killed by measuring it.** "Beat ADP by a tier"
  had base rates climbing monotonically with how late you were drafted
  (0.000 → 0.451) and was structurally impossible for tier 1. That is a measure
  of price, not performance.
- **Pooling positions was actively harmful** — AUC 0.468, below chance, with
  inverted calibration. Widening the data window pushed it *toward* chance,
  which is what happens to a fit that was mostly variance.
- **QB is significantly worse than price.** Spearman delta **-0.099, CI [-0.173,
  -0.033]**, excluding zero.

Nothing is cut from either section. `sample_adequacy`, the calibration bins and
the per-position tables all stay in the reading path, because the audience that
justifies keeping the sections is the audience that reads them.

---

## Tab 4 — Strategy (keep, one caption fix)

Already carries its own honest disclaimer: pinned to the pre-superflex format,
because its board is 1QB ADP and its templates are two-FLEX shaped. Running it
under superflex would report that QBs are nearly free *and* start twice.

No change beyond making the pin more prominent. **Do not unpin it as part of this
refactor** — that is a board swap plus new templates, and FFC does publish 2QB
ADP back to 2018, so it is cheap but it is separate work.

---

## Tab 5 — Board (unchanged name, unchanged content)

`valuation.py` — quality against price, and the undervalued/overvalued splits.

Keep the two-axis scatter. Keep the framing that this is a **disagreement score,
not a projection**: it says where the project's read differs from the market's,
which is a reason to look closer, not a reason to be right. That caveat is more
important now that the null result above is stated sharply — the two have to be
consistent with each other, and they are.

---

## Tab 6 — Glossary (keep)

214 lines, no charts. Higher value for the evaluator audience than for Zach.
No change.

---

## What is explicitly NOT in this refactor

- **Unpinning the strategy simulator.** Separate work.
- **Adding the 2025 label season.** FFC has backfilled 2025 at every format this
  project uses, so `config.ADP_MISSING_YEARS` is stale and a seventh label season
  is available. That moves every number in the project and deserves its own pass.
- **A dynasty view.** Out of scope by decision.
- **Anything requiring an LLM in the loop.** The point of §3 in the handoff is
  that these conclusions become reachable without one.

---

## Build order

Deadline-driven. Draft Day is the only part with a date attached.

1. **Draft Day tab** — surfacing `board.py`. Nothing here needs new analysis.
2. **Retire the Screen tab** now that its content is absorbed.
3. **Shared lead-in** above breakout, stating the question and both answers.
4. **Trim** rookies and comparables tables.
5. **Captions** — Landscape scoring-history note, strategy pin.

Steps 1-2 have to land before 8/22. Steps 3-5 do not.

---

## Open items

None from round one. All three resolved 2026-08-08.

---

# Round two — after reviewing the built dashboard

Notes taken 2026-08-08 on the live app. The theme underneath all of them:

> **Rank players by metrics that provably repeat, priced against draft cost,
> split by position.**

That is a sharper thesis than "research archive + draft day", and it changes what
each tab is *for*. Stability stops being a finding on a tab nobody reads and
becomes the engine.

## The finding that drives the rest: PAR is a price, not a rating

`exp_points` maps a player's *positional ADP rank* to what players at that rank
have historically scored. So within a position the board reproduces the market's
order exactly, and PAR answers "what is this draft slot worth" — never "how good
is this player".

The case that made it visible:

| | 2026 ADP | pos rank | PAR | 2025 actual | 2025 finish |
|---|---|---|---|---|---|
| Nabers | 60.2 | WR21 | **14.2** | 48.1 | WR81 *(4 games)* |
| Waddle | 72.4 | WR25 | **0.0** | 148.1 | WR9 |

Waddle outscored Nabers by 100 points and carries the worse PAR, purely because
he is drafted twelve picks later. That is `board.py` working as designed, and it
is the honest consequence of the measured null. **But the dashboard presented it
as a player rating, which it is not.** Fixed with a warning callout, and with a
second opinion beside it.

## Decided in round two

**Add `value_gap` beside PAR — a second opinion not derived from ADP.** Quality
percentile within position minus price percentile within position, from
`valuation.board`, weighted by how much each metric repeats. Positive means this
project rates him above his cost. Keep PAR; the interesting column is the
disagreement.

**Per-position sectors.** The cross-position board stays (that is PAR's job), and
a one-position-at-a-time quality-vs-price view sits below it.

**Keeper accounting expander: deleted.** Keepers were already filtered off the
board, so it was redundant. The unmatched-keeper warning stays — that one is a
real safety check.

**Promotion screen weekly trend: data layer, filtering, and the plot.** Add
receiver metrics to `weekly_trust` (routes, air yards, red-zone targets), stop
offering carry-share metrics to pass-catchers, and retire the current line plot,
which does not show what it needs to show.

**Landscape: all four.** Bigger and interactive; tier bands on the dropoff chart;
concentration broken out by position and season; and the scarcity layer fed into
the rankings rather than left as a separate view.

**Strategy tab: deferred to 2027.** Only useful as a live co-pilot — "take the QB
now or wait a round" — which needs live rankings, roster state and league draft
rates. Not before 8/22.

**Players tab: only stability survives as a *feature*.** The rest stays as the
record (agreed in round one), but the sticky-metric weighting is what gets
promoted into the rankings.

## Two limits that must stay visible

**Quarterbacks are not quality-scored.** `SKILL_POSITIONS` is WR/RB/TE, because
yards per route run, separation and yards after contact have no QB analogue. All
23 QBs on the board come back null. In a superflex league that is a hole exactly
where the league is deepest, so the UI must read a blank as *not measured*,
never *bad*.

**Vegas player props are not available free.** nflverse ships *game* lines only.
`preseason_environment` already turns those into implied team totals, which is
the free proxy and is what `env_swing` is built on. Yardage and TD over-unders
are a paid product; the alternative is scraping sportsbooks, which is fragile and
ToS-dubious.

## Round-two build order

1. ~~`value_gap` on the draft board + per-position quality-vs-price view~~ done
2. ~~Delete keeper accounting~~ done
3. ~~Promotion screen: receiver metrics in `weekly_trust`, fix filtering, retire
   the line plot~~ done — `air_yards_share_wk` and `rz_target_share_wk` added,
   `TRUST_METRICS` gives each position its own markers, and the single-metric
   line plot is replaced by `role_shift` (early vs late levels, stated) plus a
   faceted small multiple across all three markers at once
4. ~~Landscape: sizing and interactivity, then tier bands, then concentration~~
   done — charts enlarged and pan/zoom enabled, `landscape.tier_breaks` draws
   boundary rules on the dropoff, and concentration is now faceted per position
   with the ordered top-N on the ordinal ramp
5. Feed the scarcity layer into rankings — **scoped and not recommended as
   written; see below**

## Two findings the Landscape work turned up

**The dropoff is a slope, not a staircase.** Running back falls 1.8, 1.0 and 1.7
points per game across the first four ranks, then settles at 0.5–1.0 per rank the
whole way down — mean 0.68, range 0.50–0.98 from rank 10 to 48. The named cliffs
of draft folklore, RB12 and RB24, are not in the data. Reaching pays at the very
top and steadily less after, which is a different instruction than "reach for the
tier break".

**Concentration has barely moved, and tight end moved the wrong way.** Over
2018-2025, QB, RB and WR concentration all shifted by under two points, which on
shares this size is flat. Tight end is the exception and it went *down*: top five
from 31.7% of the position to 23.9%, top fifteen from 64.7% to 58.6%. The
elite-tight-end premium is a claim that a few players own the position, and over
this window the position spread out instead.

## Chart conventions this pass established

- **The y-axis is not anchored at zero on the dropoff**, because the panel's job
  is the shape between ranks rather than each rank's size against nothing. With a
  zero baseline the RB curve sat in the top half of the plot and the cliff it
  exists to show flattened into the margin. Bars would need the baseline; a line
  encoding shape does not.
- **Ordered quantities take the ordinal ramp, not categorical hues.** Top 5/15/30
  is ordered, so it is one hue light-to-dark and position gets its own panel.
  The chart previously had this backwards.
- **The position palette is validated, not eyeballed** — it passes the CVD and
  lightness checks in both modes. Light mode raises a contrast warning on WR and
  TE, which obliges a table view; every chart here already ships one.

---

# Item 5 scoped — "feed the scarcity layer into rankings"

**Recommendation: do not build it as written. Build the reframed version
instead.** Two reasons, both measured rather than argued.

## Reason 1 — the *level* of scarcity is already in the board, by construction

This is definitional, not empirical:

```
par              = exp_points - replacement_points
replacement_points  comes from  scoring.starter_demand(roster_positions, teams)
```

`starter_demand` is positional scarcity. It is the reason replacement quarterback
moved from QB11 to QB21 when the superflex slot arrived, and the reason the
league's whole board reordered behind it. **PAR is the scarcity-adjusted
ranking.** Multiplying it by a second scarcity term would count the same fact
twice, and the result would look like a refinement while being an error — the
same class of mistake as reading `adj_adp` against a pick number.

## Reason 2 — the *trend* in scarcity is not a signal

If positional scarcity moved reliably year to year, last season's level would be
worth carrying into 2026. Spearman of PAR-per-starting-slot against season,
2018-2025, bootstrapped over seasons:

| position | ρ | 95% CI | 2018 → 2025 | verdict |
|---|---|---|---|---|
| QB | +0.05 | [−0.62, +0.93] | 3.97 → 3.51 | indistinguishable from flat |
| RB | +0.10 | [−0.62, +1.00] | 4.74 → 4.25 | indistinguishable from flat |
| WR | −0.26 | [−0.79, +0.86] | 3.42 → 2.60 | indistinguishable from flat |
| **TE** | **−0.90** | **[−0.95, −0.36]** | **3.85 → 1.28** | **real decline** |

Three of four positions have intervals spanning nearly the whole range. And the
year-over-year autocorrelation at those three is *negative* (QB −0.14, RB −0.35,
WR −0.42): a scarce year tends to be followed by a slack one. Feeding last
season's scarcity forward would push the board the wrong way at three positions
out of four — though on eight seasons that mean-reversion is itself too noisy to
lean on in the opposite direction either.

**Tight end is the one real finding**, and it agrees with the concentration
result from the Landscape pass: PAR per starting slot fell from 3.85 to 1.28 and
the top five went from 31.7% of the position to 23.9%. The position got
shallower *and* flatter. That is one number worth knowing, not a layer worth
building.

## What is worth building: the cost of waiting, at your actual picks

The board says what a player is worth. It does not say what it costs to wait,
and that is the question actually being asked on the clock. It needs three things
the board never combines: the *shape* of the curve rather than its level, the
draft-slot dispersion FFC publishes, and your own pick list.

Expected PAR of the best player still available, at each of your first six
usable picks:

| pos | 4 | 17 | 24 | 37 | 44 | 47 |
|---|---|---|---|---|---|---|
| QB | 41.8 | 35.0 | 21.9 | 15.9 | 14.6 | 13.5 |
| RB | 61.2 | 61.2 | 59.6 | 42.3 | 33.9 | 28.1 |
| WR | 62.4 | 52.0 | 48.4 | 25.9 | 20.4 | 19.7 |
| TE | 33.8 | 33.8 | 33.8 | 33.7 | 33.6 | 33.5 |

Cost of waiting one slot:

| pos | 4→17 | 17→24 | 24→37 | 37→44 | 44→47 |
|---|---|---|---|---|---|
| QB | 6.8 | **13.1** | 6.0 | 1.3 | 1.1 |
| RB | 0.0 | 1.6 | **17.3** | 8.4 | 5.8 |
| WR | **10.4** | 3.7 | **22.4** | 5.6 | 0.7 |
| TE | 0.0 | 0.0 | 0.1 | 0.1 | 0.1 |

Three readable instructions fall straight out, and none of them is on the board
today:

- **Waiting on tight end costs nothing** — 0.3 points across forty-three picks.
  The best available TE is expected to survive every early pick you own, because
  the top of the position is kept and the rest is flat.
- **Running back is free to wait until 24 and expensive after** — 1.6 points to
  wait from 17, then 17.3 from 24 to 37.
- **The 24 → 37 gap is where the draft happens for you.** WR loses 22.4 and RB
  17.3 across it. That is the pick to trade up into or to plan two rounds ahead
  of.

This is also the honest version of the co-pilot the Strategy tab was wanted for —
"take the quarterback now or wait a round" — computed from pieces that already
exist rather than from a simulator pinned to the wrong format.

## A correctness bug found while scoping — needs a decision

`adp.survival` computes P(available) from **raw ADP against a pick number**, and
`board.targets` and the Draft Day availability panel both sit on it. In a keeper
league that is the double-count trap in the other direction: keepers are off the
board, so players go *earlier* than public ADP, and survival is overstated.

Measured at pick 24:

| player | adp | exp_pick | P(avail) raw | P(avail) adjusted |
|---|---|---|---|---|
| James Cook III | 19.4 | 12.4 | **0.137** | **0.003** |
| Derrick Henry | 13.9 | 9.8 | 0.002 | 0.000 |

A 45× overstatement on the player it matters for. `keeper_adjusted_adp` already
computes `exp_pick` for exactly this comparison; `survival` simply is not using
it. The fix is to let `survival` take a column argument and pass `exp_pick` from
the board, which changes availability numbers on the draft-day tab and is worth
doing before 8/22.

## Item 5, built — and two bugs it uncovered

**Cost of waiting shipped** as `board.cost_of_waiting`, on the Draft Day tab
under the board. Expected PAR of the best player of each position still
available at each pick you own, and the difference between adjacent picks.

**The survival bug is fixed.** `adp.survival` now takes an `adp_col`, and
`board.targets` and `cost_of_waiting` both pass `exp_pick`. The default stays
`adp` because a bare FFC board has no keepers to correct for.

Fixing it **changed the answer**, which is the argument for having fixed it
before building on top. On raw ADP the quarterback wait looked cheap early; on
the keeper-adjusted line, waiting from pick 4 to 17 costs **20.3 points at
quarterback** — the largest single number on the grid, and exactly what a
superflex league should produce. The standing advice is unchanged in shape but
sharper: tight end is free to wait on (0.3 across all six picks), running back
is cheap until 24 and costs 19.5 after it.

**A second, unrelated bug surfaced from a failing test.** nflverse's *2026
roster feed alone* spells Arizona `AZ`; every other table it publishes — 2025
rosters, weekly stats, schedules, draft picks — says `ARI`. `rookies.py` decided
whether a player left his team by comparing those two codes directly, so **every
Cardinal who stayed read as gone** and the entire team's production landed in the
vacated pool. Arizona's vacated target share was 1.00, which would have ranked it
first in the league; corrected it is 0.209, *below* the league mean of 0.249. The
analysis was pointing at the opposite of the truth.

`AZ → ARI` now lives in `ids._PFR_TEAMS`, and both sides of that comparison are
normalized. The test that caught it was itself only testing half the join —
mapped codes against a *raw* roster — and passed for as long as nflverse agreed
with itself. It now normalizes both sides, which is the rule the function exists
to enforce.
