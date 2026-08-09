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
3. Promotion screen: receiver metrics in `weekly_trust`, fix filtering, retire
   the line plot
4. Landscape: sizing and interactivity, then tier bands, then concentration
5. Feed the scarcity layer into rankings *(largest, most speculative)*
