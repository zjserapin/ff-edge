# Dashboard refactor — spec v2

**Status:** AGREED 2026-08-10, all six decisions yes. **Built the same day** —
see "What actually shipped" at the bottom for the deltas between this document
and the code, including two things the spec got wrong.
**Written:** 2026-08-10. **Draft day:** 2026-08-22 — **12 days.**
**Source:** Zach's pass on 2026-08-08, re-read against the code on 08-10.

Supersedes nothing. `DASHBOARD_SPEC.md` (agreed 08-08, built) still describes
why the six tabs exist. This document is about what that build got wrong.

---

## How to read this

Your notes were written on 08-08. A session ran on 08-09 and addressed some of
them without you seeing the result. So every item below is tagged:

| tag | meaning |
|---|---|
| **DONE** | Already fixed on 08-09. Listed so you know, not to be re-done. |
| **ANSWERED** | You asked a question. Here is the answer from the code. |
| **OPEN** | Still outstanding. Has a proposal and needs your yes/no. |
| **CONFLICT** | What you asked for contradicts a decision already recorded in the repo. You decide; I flag it. |

Six items need a decision from you. They are collected at the bottom.

---

## Part 1 — Your three questions, answered

### Q1. "Is PAR based on 2025 actuals or projected 2026? If JSN outscored Puka in '25, how is his PAR lower?" — **ANSWERED**

**Neither.** PAR is based on *what the ADP slot has historically been worth*,
averaged over seven label seasons. No player's own production enters it at all.

`expected.adp_curve` takes every labeled player-season, buckets by **positional
ADP rank**, averages the fantasy points scored at that rank, and pushes the
result through isotonic regression. `expected.expected_points` then joins this
season's board to that curve on `(position, adp_pos_rank)`. A player's
`exp_points` is a property of his rank and nothing else.

Your example, with the real numbers:

| | 2026 ADP | pos rank | exp_points | ±se | **2025 actual** | 2025 finish |
|---|---|---|---|---|---|---|
| Puka Nacua | 2.8 | WR1 | **177.2** | 10.8 | 212.4 | WR2 |
| Jaxon Smith-Njigba | 5.2 | WR2 | **166.4** | 9.0 | **242.9** | **WR1** |

JSN was **the WR1 in 2025** and outscored Puka by 30.5 points. His PAR is 10.8
lower for exactly one reason: he is drafted 2.4 picks later. That is the board
working as designed and it is not a bug — but it is also not a player rating,
and the tab presented it as one.

**The sharper problem, which nobody has flagged before now:** the gap between
them is **10.8 points against a standard error of 10.8 and 9.0**. The board is
displaying a distinction the curve cannot support. It is not just that PAR
follows price — it is that at the top of a position, PAR differences are inside
the noise band and are still shown to three significant figures.

> A warning callout was added on 08-09 (`app.py:2115`) and `value_gap` was added
> as a second opinion. Both help. Neither addresses the standard-error problem.
> See **D1**.

### Q2. "What is 'env swing'?" — **ANSWERED**

`board.attach_environment`, `src/board.py:790`. In one sentence:

> **The season-long fantasy points a player gains or loses purely from the
> offence he plays in, versus a league-average offence.**

Three numbers multiplied:

```
env_swing = 45.3  ×  (his team's implied total − 22.0)  ×  (his exp_points / 947)
            ↑                    ↑                              ↑
   measured: points of      how far his offence is       his share of a
   fantasy production       priced above/below the       typical team's
   per point of implied     league average, from         fantasy production
   team total               Vegas game lines
```

So a receiver on an offence priced two points above average, who is worth ~10%
of a typical team's production, gains about 9 points over the season.

**Why it is not double-counting the ADP:** the expected-points curve is blind to
team. It assigns this year's TE1 whatever the historical average TE1 scored,
whether he plays for the best offence in football or the worst. The market's
view of the team is in the ADP *level*, not in his positional *rank* — and the
curve only reads rank.

**It is an upper bound, not a correction.** Some of the discount is already in
the price and `env_swing` does not know how much. Read it as "how big is the
argument", never as points to add or subtract.

The app never says any of this. It shows a column called `env_swing` in a
43-column table. **See D4.**

### Q3. "Not following 'where offense outweighs the board' — how to filter, and its relevance?" — **ANSWERED, and I agree with your instinct**

`board.context_flags`, `src/board.py:906`. What it actually does:

1. Take the **top 6 players by PAR** at each position.
2. Form every pair within those 6.
3. Flag a pair when `env_edge > par_edge + 1.0` — i.e. the lower-ranked player's
   environment advantage exceeds the higher-ranked player's board advantage.

**Three things are wrong with it as presented:**

- **The top-6 cutoff is arbitrary and undisclosed.** At QB in a superflex league
  you are choosing among ~20 quarterbacks, not 6. The panel is structurally
  blind to the position where you have the most picks to make.
- **It compares PAR edges that are already inside the standard error** (Q1). A
  "par_edge of 3.2" between WR2 and WR4 is not a real edge, so "environment
  outweighs it" is true of almost any pair and means very little.
- **It produces pairs, not decisions.** On the clock you have one pick and a
  list of available players. A cross-product of six names is the wrong shape.

**Proposal: cut it, and replace the idea with a column.** See **D5**.

---

## Part 2 — Already fixed on 08-09 (no action)

| your note | status |
|---|---|
| "WR debate has too many RB-style metrics" | **DONE.** `promotion.TRUST_METRICS` is now position-keyed: WR/TE get `target_share`, `air_yards_share`, `rz_target_share`; only RB gets carry metrics. |
| "Target share on a line plot doesn't tell you anything" | **PARTLY DONE.** `promotion.role_shift` was added — early-window vs late-window levels with a delta, instead of asking you to eyeball a trend. **But the line plot is still there too** (`app.py:1885-1924`). Both render. See **D7**. |
| "Sticky metrics … bake into player evaluation/rankings" | **ALREADY TRUE, AND INVISIBLE.** See below — this is the biggest miss in the current app. |

### The sticky-metrics finding is already in the rankings. You just can't see it.

`archetypes._weighted` (`src/archetypes.py:111`) weights every quality metric by
its **year-over-year correlation** — exactly the stability table on the Players
tab. That weighted score becomes `quality_score` → `quality_pct` → `value_gap`,
which is the second-opinion column on the Draft Day board.

It measurably worked: weighting by stability moved the score's rank correlation
with *next* season's points from 0.464 → 0.502 at WR and 0.455 → 0.492 at TE.

So your instinct was right and it was already acted on. The failure is
**navigational**: the finding is presented as research on a tab you called
useless, and consumed silently on a different tab, with nothing connecting them.
**See B2 and P2.**

---

## Part 3 — What this audit found that was not in your notes

### F1. QB quality is computed, then thrown away — in a superflex league

`archetypes.scores()` returns quality and opportunity percentiles for **32
quarterbacks** right now. I ran it. Lamar Jackson comes back at the 100th
quality percentile, Josh Allen 96.9, Drake Maye 93.8.

`valuation.board()` then drops all of them, because:

```python
SKILL_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")   # src/valuation.py:49
```

Which is why every quarterback on the Draft Day board has a blank `value_gap`.
The docstring justifies it — "yards per route run, separation and yards after
contact have no quarterback analogue" — and that is true of *those* metrics. But
QB has its own quality set (`ypa`, `pts_over_exp_per_att`, `ypc`) and the
stability table says the rushing side of it is the **stickiest thing measured
anywhere in this project**:

| QB metric | r_yoy | verdict |
|---|---|---|
| `rush_share` | **0.82** | sticky — highest of all 72 metrics, any position |
| `ypc` | 0.601 | sticky |
| `exp_ppg` | 0.561 | sticky |
| `ypa` | 0.360 | usable |

Your structural edge in this league is that six teams need seven QB-capable
starters from ~3 startable quarterbacks and you need zero. The one position
where you hold the whip hand is the one position the value model refuses to
score — and the data to score it is sitting in the cache.

**This is one tuple.** Highest value per line changed in the entire spec.
**See D3.**

### F2. `data/win_totals_2026.csv` should not be filled — `HANDOFF.md` is stale on this

The handoff lists filling it as priority #3, "the one manual input that would
improve the preseason environment layer." The module docstring at
`src/expected.py:53` records the opposite, with the measurement:

```
actual wins        -> team fantasy points    0.615
weeks 1-3 implied  -> team fantasy points    0.619   ← already posted, free, auto-updating
full-season implied-> team fantasy points    0.870   (concurrent, not available preseason)
```

Actual wins is the **ceiling** a preseason win total could reach, and the game
lines already there match it. All 32 teams have 3-4 priced games for 2026. Hand
entering 32 numbers buys nothing. **Recommend: delete the blank file, strike the
item from the handoff.** No decision needed unless you disagree.

### F3. The Strategy tab reports results for a format you do not play

`config.SIM_ROSTER_POSITIONS` is deliberately pinned to the **old two-FLEX**
format, because the simulator's board is 1QB ADP and its templates are two-FLEX
shaped. `CLAUDE.md` says the pin is intentional and must not be "fixed" without
also swapping in 2QB ADP and rewriting the templates.

That pin is correct as engineering and wrong as a user-facing tab. You play
superflex. A tab that silently answers a two-FLEX question is the exact category
you flagged — "content that misleads." You independently called the tab not
useful. Those agree. **See D6.**

---

## Part 4 — Proposed changes, by tab

### Tab order

```
current:   Draft Day | Landscape | Players | Strategy | Board | Glossary
proposed:  Draft Day | Board | Landscape | Research | Glossary
```

Five tabs. `Players` and `Strategy` merge into **Research**, which is where the
honest nulls live. **They are not deleted** — the 08-08 spec's audience answer
was "Zach, plus people evaluating his work", and the negative results are the
most credible thing in this repo. They stop being a *destination* and become an
*archive*, which is what your "effectively useless" note is actually describing:
not that they should not exist, but that they should not be in the way.

`Board` moves to second because it is the tab you said you like and it is the
one that answers "who is actually good", which is the question the Draft Day
board is worst at.

---

### Draft Day

**D1 — Show PAR's uncertainty, and stop implying precision it does not have.**
**OPEN.** Three changes:
- Display `exp_points` with its standard error: `177.2 ± 10.8`.
- Grey out or band together any two players at a position whose PAR gap is
  inside one pooled SE. Puka and JSN are one asset, and the board should say so.
- Replace the current wall-of-text warning with one line above the table:
  *"PAR rates the draft slot, not the player. Within a position it reproduces
  ADP's order exactly — read `value_gap` for the independent opinion."*

**D2 — Vegas yard O/U and TD O/U.** **CONFLICT.** You asked for these.
`src/expected.py:48` records a decision not to use them, for three reasons:
they are not free at useful coverage; they exist only for stars who are already
the easiest players to rank; and a single threshold ("10+ TDs") gives one point
on a distribution rather than an expectation.

The first is the binding one — this is a paid-data problem, not a code problem,
and it is 12 days out. What the repo *does* have is team-level implied totals
from free nflverse lines, already flowing into `env_swing`.

**My recommendation: decline for 2026, and instead surface the team-level
environment properly (D4), which is the same signal one level up and costs
nothing.** If you want player props anyway, that is a data-sourcing decision and
I need you to name a source. → **DECISION 2**

**D3 — Score quarterbacks.** **OPEN, highest value.** Add `"QB"` to
`valuation.SKILL_POSITIONS`. Guard rails, because this is a real analytical
change and not just a config edit:
- QB rides on 3 quality metrics vs 10 at WR. The panel must say so — a QB
  `quality_pct` is a thinner measurement than a WR's and should be labelled.
- The stability weighting will make it mostly a **rushing** score (`ypc` 0.601,
  `rush_share` 0.82 on the opportunity side). That is defensible for fantasy QB
  and should be stated out loud rather than discovered.
- `min_routes=100` is meaningless for a QB; the filter needs a position-aware
  floor (games, attempts) or QBs will be dropped by a receiver's gate.
- Verify against 2025: does the QB quality score rank-correlate with next-season
  PPG at all? If it comes back at zero, **we report that and leave QB blank** —
  a measured null, not a feature. → **DECISION 3**

**D4 — Explain `env_swing` where it is used.** **OPEN.** Add the three-number
formula above as an expander next to the board, and rename the column header to
**"Offence swing (pts/season)"**. Keep `env_swing` as the code-level name.
Add the upper-bound caveat as a caption. Low risk, high clarity.

**D5 — Cut "Where the offence outweighs the board".** **OPEN.** Remove the
pairwise panel. Replace with a single sortable column on the main board — the
same `env_swing` number, already computed — so the environment argument travels
with each player instead of living in a cross-product you have to cross-
reference. `board.context_flags` stays in `src/` with its tests; it just loses
its UI. → **DECISION 5**

**D6 — Keeper accounting.** **OPEN.** You asked: get rid of it, or mark players
off the board? **Both, split by which question they answer:**
- **Kept players get marked off the board** — struck through and filtered out by
  default, with a toggle to show them. They are not draftable; showing them as
  available is the misleading part.
- **The pick-consumption math stays**, because it is why "17 owned, 15 usable"
  is two different numbers, and because a keeper slotted onto R6 is why your R6
  pick does not exist. That is a real planning constraint, not accounting noise.

**D7 — Promotion screen.** **OPEN.** You said the idea is good and the practice
is failing. Three changes:
- **Delete the weekly line plot** (`app.py:1885-1924`). `role_shift` already
  answers the question it was asked, and better. Your note was right and the fix
  was half-applied.
- **Add the context the metric misses**, as flags on the player card:
  `new team for 2026`, `games played: 4`, `changed team since the measured
  season`. Your Nabers example is exactly this — 4 games in 2025 — and Waddle
  shows `MIA` in the 2025 stats against `DEN` on the 2026 board. Both are
  already in the data and neither is shown.
- **Gate the verdict on sample size.** A player with 4 observed weeks should not
  receive a grade in the same typeface as one with 17.

**D8 — Tier breaks in PAR terms.** **OPEN.** You asked to "figure out PAR for
tier breaks." `landscape.tier_breaks` exists but runs on the historical PPG
curve and only renders on Landscape. `expected.tiers` runs on `exp_points` and
already assigns a `tier` column to the Draft Day board — it is in the frame at
`app.py:2148` but not visually expressed. Proposal: draw the tier boundaries as
horizontal rules in the board table, with the PAR value at each break and the
count of players remaining in the current tier. That converts "who is next" into
"how many of this asset are left", which is the actual draft-day question.

---

### Board

**B1 — One position at a time.** **OPEN.** Replace the position multiselect with
a segmented control. Same chart, one position, bigger. This is your "per-position
sectors, easier to digest."

**B2 — Sticky stats against draft price.** **OPEN.** This is your "want more of
these graphs", and it is the item that closes the loop on the Players tab.

A small-multiples grid: for the selected position, each **sticky** metric on the
y-axis against draft-price percentile on the x-axis, one panel per metric,
sorted by `r_yoy` descending, with the `r_yoy` printed in each panel title. The
metric list is not hand-picked — it comes from `stability.year_over_year`
filtered to `verdict == "sticky"`, so it defends itself:

| position | sticky metrics (r_yoy) |
|---|---|
| **WR** | exp_ppg .665 · air_yards_share .662 · target_share .661 · tgt_per_game .643 · adot .638 · neutral_target_share .632 · ppg .577 · avg_separation .573 · tprr .571 · exp_td_share .561 · snap_pct .551 · yprr .505 · catch_rate .468 · rz_target_share .464 · ez_target_share .458 |
| **TE** | air_yards_share .699 · snap_pct .661 · target_share .615 · exp_ppg .611 · tgt_per_game .607 · neutral_target_share .592 · ppg .572 · adot .554 · yprr .478 · tprr .474 |
| **RB** | target_share .650 · exp_ppg .606 · carry_per_game .588 · rush_share .570 · neutral_rush_share .567 · snap_pct .566 · ppg .564 · rz_carry_share .517 · route_share .459 |
| **QB** | rush_share .820 · ypc .601 · exp_ppg .561 · ppg .473 |

Fifteen panels at WR is too many. **Proposal: default to the top 6 by `r_yoy`,
with a "show all sticky" toggle.** Note that `exp_ppg` and `ppg` are *outcome*
columns, not inputs — they should be visually separated or excluded, since
plotting last year's points against price is close to plotting price against
itself. → **DECISION 4** (top-6 default, and whether to exclude outcome columns)

**B3 — Add QB to this tab too**, contingent on D3.

---

### Landscape

You like this tab. Nothing structural changes.

**L1 — Bigger.** **OPEN.** Current chart heights: PAR-per-slot 520, dropoff
shape 430, top-of-board 320, concentration 300. Proposal: dropoff → 600,
concentration → 480, and give both `width="container"` with the sidebar
collapsible so a chart can use the full window.

**L2 — Tier segmentation on the dropoff.** **OPEN.** `landscape.tier_breaks` is
already computed and already rendered on this tab (`app.py:533`) but reads as a
table. Proposal: shade the tier bands directly onto the dropoff curve as
alternating background rectangles, so the cliff is visible in the same picture
as the slope. Pairs with **D8** — same tiers, two places, one definition.

**L3 — "Top players taking a bigger slice", deeper.** **OPEN, needs scoping.**
The current panel reports the 2018→2025 change in top-5 and top-15 share. The
finding that TE concentration went *down* (31.7% → 23.9% top-five) contradicts
the elite-TE premium and deserves more room. Candidate directions — I would want
to pick **one**:
- Share by rank band over time as a stacked area, rather than two endpoints.
- Concentration against the dropoff slope, testing whether they say the same
  thing (I suspect they largely do, and that is worth knowing).
- Within-season vs across-season concentration — is the top 5 the *same* five?

**L4 — Feed this layer into the rankings.** **CONFLICT, needs care.** Positional
*scarcity* into rankings was scoped and **declined** on 08-09, for a good reason:
`par = exp_points − replacement` and replacement comes from `starter_demand`, so
the level of scarcity is already in the board by construction. Adding a scarcity
term counts one fact twice.

**Concentration is not the same quantity as scarcity**, so this is not
automatically the same refusal — but it is close enough that I will not build it
without measuring first. Proposal: measure whether concentration adds anything to
`par` out of sample, report the interval, and **build only if it clears zero.**
That is a research task, not a dashboard task, and I would put it after the
draft. → **DECISION 6**

---

### Research (formerly Players + Strategy)

**P1 — Merge and demote.** **OPEN.** Players' four sections (stability, breakout,
projection, rookies) plus Strategy become one tab, collapsed by default, each
section behind an expander. Content unchanged — this is a move, not a rewrite.
The 08-08 decision "the Players tab stays whole" is preserved: it stays whole,
it just stops being 3rd in the tab strip.

**P2 — Stability gets a pointer to where it is used.** **OPEN.** One line under
the stability table: *"These correlations are the weights in `quality_score` —
see the Board tab. Weighting by them moved rank correlation with next season's
points from 0.464 → 0.502 at WR."* Closes the loop you noticed was missing.

**P3 — Strategy gets a format warning.** **OPEN.** A banner stating it simulates
a **two-FLEX, 1QB-ADP** format, not your superflex league, and that its numbers
do not transfer. Cheapest possible fix for F3. The alternative — rebuilding it
for superflex — is a 2027 item you already deferred.

---

## Part 5 — What I am not proposing, and why

- **Deleting the negative results.** `breakout.py` and `projection.py` are
  measured nulls and the repo standard is that a negative result is a result.
  They move; they do not go.
- **Rebuilding Strategy for superflex.** Needs 2QB ADP and new templates. You
  already deferred it to 2027 and I agree. P3 is the 12-days-out answer.
- **Splitting `app.py` into modules.** It is 2,391 lines and violates the
  `Projects/CLAUDE.md` standard. You did not select it and it is invisible to
  you. **After the draft.**
- **Player props.** See D2.
- **A dynasty view.** Out of scope by standing decision — needs an age curve.

---

## Part 6 — Decisions I need from you

| # | question | my recommendation |
|---|---|---|
| **1** | Tab order → `Draft Day \| Board \| Landscape \| Research \| Glossary`, with Players+Strategy merged into Research? | **Yes** |
| **2** | Vegas player yard/TD O/Us — decline for 2026 and rely on team-level implied totals, or do you want to name a paid source? | **Decline for 2026** |
| **3** | Score QBs on the value model (add `"QB"` to `SKILL_POSITIONS`), with the thin-measurement caveats and a validation gate? | **Yes — highest value item here** |
| **4** | Sticky-vs-price panels: default to top 6 by `r_yoy`, and exclude the outcome columns (`ppg`, `exp_ppg`)? | **Yes to both** |
| **5** | Cut the "offence outweighs the board" pairwise panel, replace with an `env_swing` column on the main board? | **Yes** |
| **6** | Concentration into rankings — measure first, build only if the interval clears zero, and do it after the draft? | **Yes, after** |

Everything else (D1, D4, D6, D7, D8, B1, B2, L1, L2, P1, P2, P3) I will build as
specified unless you say otherwise.

---

## Part 7 — Sequencing, against 12 days

**Ship before the draft** — everything that changes what you see on the clock:

1. D3 — score QBs *(biggest analytical gain; your league's edge)*
2. D6 — mark keepers off the board *(actively misleading today)*
3. D1 — PAR ± SE, and drop the false precision
4. D4 — explain `env_swing`; D5 — cut the pairwise panel
5. D7 — promotion screen: kill the line plot, add context flags
6. D8 + L2 — tier breaks, drawn once, shown in both places
7. P3 — Strategy format warning *(one banner, five minutes)*

**Ship if there is room:** B1, B2 (per-position + sticky-vs-price), L1 (bigger),
P1 (tab merge), P2 (the pointer).

**After the draft:** L3, L4, `app.py` module split, F2 cleanup.

**The one thing to protect:** D3 changes an analytical output, not a layout. It
needs the validation gate in its own bullet — if QB quality does not
rank-correlate with next-season PPG, it ships as a measured null and QBs stay
blank. Do not let the deadline turn that gate into a formality.

---

## What actually shipped — 2026-08-10

All six decisions approved and built. 246 tests pass (from 228), app renders
with zero exceptions under `AppTest`. **Two items in this spec were wrong**, and
both are recorded here rather than quietly corrected.

### The spec was wrong about P3

**Already built.** `app.py` has carried a superflex-conditional `st.error` on the
Strategy tab since before this session, saying exactly what §P3 proposed. Nothing
was added. Zach's "Strategy — not useful" note was about the tab's value, not a
missing warning.

### The spec was wrong about L2

§L2 proposed shading tier bands onto the Landscape dropoff curve. The chart
already draws tier boundaries as dashed rules, and the code carries an explicit
argument against bands: *"filling the interior would spend the only free channel
restating the line's own height."* That reasoning holds — colour is already
spent on season. **Not overruled.** What shipped instead: the rules are now
numbered, which is new information rather than a restatement, and the chart went
from 430×330 to 600×380. If bands are still wanted, that is a decision to
overrule a documented one, not an oversight to fix.

### D3, and what the gate found

QB scored, `valuation.SKILL_POSITIONS` renamed to `VALUED_POSITIONS`. The
validation ran before any code was written:

    QB   n=163   rho +0.367   95% CI [+0.213, +0.493]   (3 features)
    RB   n=377   rho +0.330   95% CI [+0.236, +0.419]   (8 features)
    WR   n=610   rho +0.502   95% CI [+0.439, +0.564]   (10 features)

Passed. Two caveats ship with every QB number and are in the module docstring,
the board expander and the Board tab: it is a thinner measurement, and the
2023-2025 window alone gives +0.118 [-0.208, +0.416]. The era windows overlap, so
this is not established decay — a single season's interval here is ~0.9 wide —
but it is not nothing.

**The gate also found a live bug this spec had not anticipated.** A quarterback's
`routes` value is *dropbacks*, not routes run — Josh Allen shows 665 — so the
existing `routes >= 100` filter would not have dropped quarterbacks, it would
have waved them through on a column meaning something else. Taysom Hill cleared
it on **6 pass attempts** with a `ypa` computed off the 6, scoring at the 25th
quality percentile. `MIN_VOLUME` now gates each position on its own denominator.

### D1 — `board.indistinguishable`

New function, grouping players whose PAR gaps sit inside the pooled standard
error, compared against the group *leader* rather than the previous player
(single-linkage chains a whole position into one group down a shallow slope —
pinned by a test). Surfaces as the `same` column. On the live board **the top
five receivers come back as one group.**

### What rendering caught that tests did not

The sticky-vs-price panel passed `AppTest` and was broken. Rendering to PNG
showed it drawing **one of two QB metrics and two of six at RB** — the metrics
were not carried on the valuation board, and the panel silently degraded to
whichever columns happened to exist. Seven columns added to `valuation.board`;
the panel now names anything still absent instead of dropping it.

The same pass killed the colour encoding: `value` on a shared colour scale
against independent per-panel y-scales meant aDOT's 5-20 range swamped target
share's 0.05-0.35 and every share panel rendered the same pale blue.

With all six panels drawing, the panel now shows the module's own thesis
directly: at receiver the four **volume** metrics climb steeply with price and
the two **quality** metrics are flat.

### Deferred, unchanged

L3, L4 (measure first, build only if the interval clears zero), the `app.py`
module split, and player props. `data/win_totals_2026.csv` should be deleted
rather than filled — see §F2.
