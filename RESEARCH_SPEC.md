# ff-edge research spec — rankings first, many leagues

**Written 2026-08-13. Status as of 2026-08-16: partly built — see the
correction block below before reading further.**
Supersedes the product direction in `docs/archive/DASHBOARD_SPEC.md` and
`docs/archive/DASHBOARD_SPEC_v2.md`, which both assumed one league, one format,
and a draft board as the organizing surface. Their *findings* sections stay
authoritative — they record where a spec was wrong against real data, and that
is still the most useful thing in either file.

> ### Corrections — applied 2026-08-16
>
> The header used to read *"Nothing here is built."* That is no longer true and
> three claims below have been overtaken. Left inline rather than rewritten, so
> the original reasoning stays legible.
>
> 1. **§3 is wrong that ADP history had been "accumulating daily."** It had not
>    — each history file held exactly one snapshot, so `adp.movement` returned
>    an empty frame. A launchd agent has run twice daily since 08-13 and it
>    returns real rows now. **ADP history cannot be backfilled**; a day not
>    pulled is gone permanently.
> 2. **§5.2 counts three independent opinions. There are two.** `par` is derived
>    from the ADP curve, so a blend of all three double-counts the market. See
>    `BIG_BOARD_SPEC.md` §2. This is the single most load-bearing correction in
>    this file.
> 3. **§9 Q1 and Q2 are answered.** Three leagues, read live from Sleeper rather
>    than asked for — Shiva Bowl 08-22, The Jungle 08-30 (out of scope), 828
>    Omegle Chat 09-06. The roster is confirmed superflex. Q3 and Q4 remain
>    open, and Q4 now has a shipped answer to react to.
>
> **What has since been built**, against §8's phasing: the Big Board tab (§5.2
> pulled forward from Phase 1), the Footballers and ECR blend, the team-
> environment weight, and the roster-demand cut. **Phase 0 items 2 and 3 —
> surfacing the `peek` screens and adding the profile selector — are still not
> done.** `app.py` imports neither module. See `DRAFT_CHECKLIST.md`.

This document answers four questions Zach asked directly:

> What analysis am I missing? What could & should be added? What is kind of
> useless and not very valuable from the perspective of a sharp fantasy player?
> What is?

and then turns the answers into a build order.

---

## 0. The premise change

The app was built around one league — the Shiva Bowl, 10-team superflex keeper,
read live from Sleeper — with a Draft Day tab as the front door. That premise
is retired. The new premise:

- **Multiple leagues, none of them privileged.** Format is a selection, not a
  fact of the environment.
- **Rankings and research are the front door.** The thing being made is a list
  you can draft from and a way to interrogate any name on it.
- **Live draft tracking is not the product.** It was a two-week detour that
  produced one genuinely valuable thing (`cost_of_waiting`) and a lot of
  league-specific plumbing.

### One prior to challenge before we cut anything

Zach's reason for retiring the superflex board:

> The league I have flagged is going to shake out closer to a normal league
> anyway based off the mocks I have been looking at.

Read carefully, that is not a statement that the format changed. It is a
statement that **the market is pricing a superflex roster like a 1QB roster.**
Those are opposite conclusions.

If the roster still has a `SUPER_FLEX` slot, league-wide QB demand is still
roughly double, replacement QB still sits near QB21 rather than QB11, and every
starting QB is still worth more than a 1QB board says. Managers drafting it like
a normal league does not remove that — **it is the precise condition under which
the edge exists and gets paid.** A market that has correctly repriced QBs
produces no edge; a market that hasn't produces one.

So: **the superflex board should not be deleted. It should be demoted from
"the app" to "one profile."** That costs almost nothing under the design below,
and it preserves the single largest measured edge in the repo. What gets cut is
the *keeper and pick-ownership machinery* around it, which is genuinely
league-specific and genuinely finished.

The thing that would change this conclusion is evidence the roster itself is
changing — a settings change on Sleeper, not mock behavior. Worth checking
before the 22nd. `src/sleeper.py` already reads `roster_positions` live.

---

## 1. The prior worth challenging: the project keeps trying to rank

The central measured finding, twice confirmed on season-forward validation:

| model | result | interval |
|---|---|---|
| `breakout.py` | +0.035 AUC over ADP | covers zero |
| `projection.py` | +0.002 Spearman over ADP | covers zero |

Seven label seasons, 831 labeled player-seasons, 666 out-of-sample rows.
**ADP's ordering is hard to beat, and this repo has not beaten it.**

And yet almost everything built since is another ordering. `par` is an ordering
that reproduces ADP within position by construction. `value_gap` is an ordering.
`vegas_gap` is an ordering. The project measured that the order is efficient and
then kept attacking the order.

The reframe, and the single most important line in this document:

> **ADP is an efficient estimate of the mean and a non-estimate of the
> variance.** Every ADP is a point. The market has no mechanism to express a
> distribution.

`HANDOFF.md` already carries the number that makes this actionable:

> The spread around the ADP curve is more than five times the step between ranks.

Which is to say: **whether a player lands at the top or the bottom of his tier's
outcome distribution is worth roughly five times more than getting the tier
order right.** The project has been optimizing the small term with the whole
apparatus and has never once measured the large one.

Nothing in the repo estimates per-player spread. `board.indistinguishable` uses
the standard error *of the curve*, which answers "how well do we know the mean
at this rank" — a different quantity from "how wide is the outcome distribution
at this rank." The second one is computable from the same label set.

This matters practically because it inverts advice by round: **you want low
variance early and high variance late.** An early pick is about not losing a
season; a late pick is a lottery ticket and a median outcome there is worth
nothing. No ADP tells you which player is which, and this is the rare place
where "the market already knows" is a weak objection — the market has no way to
say it even when it knows.

---

## 2. What is missing

Four gaps, ordered by expected value.

### 2.1 Outcome distribution, not rank — the open thread

Described above. `HANDOFF.md` has called this "the best thread" for two
sessions and it remains unstarted. It is a better use of a week than re-running
a measured null with more parameters.

### 2.2 Availability — points per game is modeled, games are not

Fantasy points = points per game × games played. This repo has a great deal of
machinery for the first term and **nothing at all** for the second. It already
trips on the gap in two places: `landscape.py:237` notes the averaging problem,
and the glossary carries a note about an injured player ranking TE11 on season
totals.

The question is not "can we predict injuries" — nobody can. It is narrower and
answerable: **is games-played predictable at all** from age, position, career
workload, and prior availability? `nv.injuries()` is already cached.

Either answer is useful:

- **If yes**, it is a direct multiplier on every ranking, and it is information
  ADP prices badly because ADP is a consensus of point estimates.
- **If no**, that is a real result with a real consequence: rank on **per-game**
  production and treat games as a lottery. That changes rankings materially
  versus season totals, and it is a defensible position rather than an
  oversight.

### 2.3 Age as a redraft signal

`age` appears in `breakout.py`'s `POSITION_FEATURES` for all four positions and
nowhere else in the repo. It has never been surfaced as a signal in its own
right, because age curves were scoped out as a dynasty concern.

That scoping is too broad. **A dynasty age curve and a redraft age effect are
different questions.**

- Dynasty: what will this player be worth in three years? Needs a curve.
- Redraft: does a 29-year-old RB at ADP 24 underperform ADP 24? Needs one
  coefficient, measured season-forward, controlling for price.

The second is cheap, in scope, and is one of the few heuristics casual drafters
systematically get wrong in both directions. It does not require building the
thing `profiles.py` correctly refuses to build.

### 2.4 Cross-market price disagreement

The board prices against one market (FFC, at the profile's format). Where FFC,
Sleeper, and Underdog ADP *disagree* is where the market is unsettled — and
unsettled is where a private read is worth something. A single-source ADP is
structurally blind to it.

Note `peek.market_disagreement` already does the adjacent thing — FantasyPros
ECR dispersion (`sd`, `best`, `worst`) — and its docstring already documents
both biases in reading it. That is *expert* disagreement. *Price* disagreement
is a second, independent axis, and `adp.multi_format` is already most of the
fetch layer.

---

## 3. What is built and unreachable

Three sharp tools exist in `src/` and cannot be reached from the dashboard.
`app.py` imports 16 modules; **`peek` and `adp` are not among them.** This is
the same failure as `board.py` sitting unreachable two weeks before a draft.

| tool | what it does | status |
|---|---|---|
| `peek.regression_candidates` | points over expected — the canonical buy-low screen | filed as a "worked example" |
| `peek.market_disagreement` | where expert consensus is least settled | same |
| `peek.snap_trend` | late-season snap share vs baseline — role change before the box score | same |
| `adp.movement` | risers and fallers from accumulating `adp_history_*` snapshots | never imported |

`peek.py`'s header calls these "not models and they aren't rankings," which
undersells them badly. `regression_candidates` is the most reliable public edge
in fantasy football and its own docstring makes the argument correctly:

> Large *negative* pts_over_exp: the volume was there and the points weren't.
> Usually the better buy, because volume is much stickier season to season than
> efficiency.

**And this repo has the evidence for that claim** — `stability.year_over_year`
measured that opportunity persists better than quality at every position. The
screen and its justification were built separately and never introduced.

`adp.movement` is free information that has been accumulating in the cache since
bootstrap started running daily. August ADP drift is camp news made numeric.

**Surfacing these is Phase 0 work: additive, low risk, no restructure.**

Caveat, and it must go in the UI: a *screen* is a question generator. Promoting
`pts_over_exp` from a screen to a ranking input requires the measurement in §5.3
first, and it may come back null like the last two.

---

## 4. What is low value to a sharp player

Ordered by lines of code freed.

| surface | lines | verdict | what survives |
|---|---|---|---|
| `simulate.py` / Strategy | 741 | **Cut from the product** | The finding, one sentence, in Method |
| Keeper + pick machinery in `board.py` | ~500 of 1111 | **Cut** | `cost_of_waiting`, `tier_map`, `replacement`, `indistinguishable`, `context_flags`, `attach_*` |
| Concentration over time (`landscape`) | part of 512 | **Demote to Method** | The dropoff-shape chart, which is decision-shaped |
| `valuation.comparables` | ~97 | **Cut** | nothing |
| `par` as a within-position ranking | — | **Reframe** | replacement level / cross-position comparison, which is the real content |
| `breakout.py` + `projection.py` | 1126 | **Freeze, don't grow** | both, in Method, unchanged |
| `claims.py` | 604 | **Keep, deprioritize** | it is a 2027 asset |

Reasoning on the four that need it:

**`simulate.py` — the biggest single block of code in `src/` after `board.py`,
and the least load-bearing.** Three problems compound: `config.SIM_ROSTER_POSITIONS`
is deliberately pinned to a format the league no longer uses; the templates are
two-FLEX shaped and the board is 1QB ADP; and the headline result (+3.5% for
late_qb) sits inside the modeling noise the repo's own summary describes. Beyond
all that it simulates **drafting**, and a sharp player's edge is overwhelmingly
in *managing* — waivers, starts, trades — which this does not touch. Fixing it
means new ADP and rewritten templates for a result that was never actionable.

**`par`'s within-position ordering is tautological with ADP.** It is derived
from the ADP curve, so of course it reproduces ADP's order. `app.py:main` says
so out loud already:

> `par` reproduces ADP's order within a position, so the second opinion needs to
> be one click away

The honest framing is that PAR is a rating of the **draft slot**, and its only
non-tautological content is **cross-position**: how much a QB1 is worth against
an RB1 given this format's replacement levels. That content is real, format-
sensitive, and exactly what a multi-league tool needs. Keep it; stop presenting
it as a player ranking.

**Concentration over time is genuinely interesting and changes no decision.**
"Are the top players taking a bigger slice?" is a good article and not a draft
input. The dropoff-shape work in the same tab is the opposite — it is what tier
breaks are made of — so this is a split, not a tab deletion.

**Comparables is fun and nobody drafts off comps.** It answers "who does this
player look like," which is a browsing behavior, not a decision.

---

## 5. What is valuable — and what to add

### 5.1 The six things worth keeping and promoting

1. **Stability (`r_yoy`)** — the engine, and the genuinely unusual asset. Almost
   every tool ranks; nearly none tell you which measurements repeat at all. It
   is what makes every weight downstream defensible instead of arbitrary.
2. **`value_gap`** — quality percentile minus price percentile, stability
   weighted, and **not derived from ADP**. This is the actual product.
3. **The rookie model** — 0.63 out of sample, 26% below the mean baseline. It
   *works*, which puts it alone among the fitted models here, and rookie
   valuation is where casual drafters are worst.
4. **`cost_of_waiting` + tier breaks + dropoff shape** — format-agnostic and
   shaped like the real question, which is never "who is best" but "who will
   still be there."
5. **`vegas_gap`** — a third opinion, independent of ADP, with real money behind
   it. Trap intact: season-long yardage markets are priced -114/-114 and de-vig
   to exactly 0.500, which is the absence of a signal, not a probability.
6. **`promotion.py`** — role change is the thing usage sees before the market
   does.

### 5.2 The missing product feature: there is no synthesis

The app currently holds **three opinions with no reconciliation** — `par`,
`value_gap`, `vegas_gap` — plus screens on top. A sharp player needs *one*
ranked list to draft from and a visible account of where the inputs disagree.
Right now that reconciliation happens in Zach's head, at the table, on a clock.

Proposed: a **composite rank per profile**, with a disagreement column. Not a
blend tuned to look good — a stated weighting, its inputs visible, and the
biggest disagreements surfaced rather than averaged away. Disagreement is
signal: it is where the private read is worth having.

### 5.3 Four measurements, each with a kill condition

House rules apply: season-forward only, cluster bootstrap by season, report the
interval. **Any of these may come back null, and a null ships as a null.**

| # | question | label | baseline to beat | kill condition |
|---|---|---|---|---|
| M1 | Is outcome **spread** predictable at a given ADP? | abs. residual vs ADP-implied expectation; and P(finish top-5 at position) | spread of the ADP curve at that rank | interval on spread-prediction covers zero |
| M2 | Is **games played** predictable? | next-season games | positional mean games | RMSE interval does not clear the positional mean |
| M3 | Does **negative `pts_over_exp`** predict beating ADP next season? | ratio rule already used by `breakout` | ADP alone | interval covers zero → stays a screen, never a ranking input |
| M4 | Does **age** move next-season outcome, controlling for price? | same as M3 | ADP alone | coefficient interval covers zero |

M1 is the highest-value and the least certain. M2 is the most likely to produce
a useful null. M3 is the one most likely to be *believed* without measurement,
which is exactly why it needs one — the confidence in that docstring is
borrowed from general fantasy lore, not from this repo's label set.

---

## 6. Proposed structure

Three surfaces, down from five tabs.

```
┌─ Rankings ──────────────────────────────────────────────┐
│  Profile selector (the league you are drafting)         │
│  One ranked list · composite + inputs + disagreement    │
│  Tier breaks · cost of waiting · dropoff shape          │
│  Screens: value gap · vegas gap · regression · movement │
└─────────────────────────────────────────────────────────┘
┌─ Player ────────────────────────────────────────────────┐
│  One name, everything known                             │
│  Price across markets · quality · opportunity · props   │
│  Stability of each metric shown next to the metric      │
│  Promotion / role change · rookie score if applicable   │
│  (Phase 2: distribution, availability, age)             │
└─────────────────────────────────────────────────────────┘
┌─ Method ────────────────────────────────────────────────┐
│  Stability tables — the engine, explained               │
│  The two measured nulls, unchanged                      │
│  Rookie model validation · PAR & replacement level      │
│  Concentration over time · the strategy finding         │
│  Glossary                                               │
└─────────────────────────────────────────────────────────┘
```

**Rankings is the front door.** It is the only surface with a decision on it.

**Player is the research surface** and is the one that does not exist today —
current sections are organized by *analysis*, so answering "what do I think
about this guy" means visiting four tabs and joining by eye.

**Method is credibility, archived not deleted.** The audience for this repo is
Zach plus people evaluating his work, and the honest negative results are the
most credible thing in it. `docs/archive/DASHBOARD_SPEC_v2.md` made that call and it stands.

---

## 7. What this requires in code

### 7.1 `profiles.py` must reach the sidebar — it currently cannot

The multi-league pivot rides entirely on `src/profiles.py`, which already does
the hard part: it binds a roster format to **the ADP market that prices it**, as
one indivisible object, because the 2026 superflex bug is exactly what happens
when those two drift.

**But `app.py` does not import `profiles` at all.** The sidebar reads one league
from Sleeper and exposes only teams, points-per-reception, and passing-TD value.
There is no way to say "12-team, 1QB, one FLEX, full PPR" in the UI. Roster
shape — the thing that moves replacement level, which moves everything — is not
selectable.

Work: import `profiles`, add a selector, thread `as_settings()` through the
existing cache boundaries. The `sleeper_backed` flag already distinguishes live
from synthetic profiles, so this is wiring rather than design. Add profiles for
the leagues Zach is actually preparing for; `customize` covers one-off variants.

### 7.2 Split `board.py`, do not delete it

At 1,111 lines it is the largest module and it is two things stitched together.

| keep — format-general | cut — league-specific |
|---|---|
| `replacement`, `build`, `targets` | `kept_players`, `keeper_summary` |
| `cost_of_waiting` | `keeper_slots`, `keeper_adjusted_adp` |
| `attach_environment`, `attach_quality` | `keeper_match_report`, `compare_baselines` |
| `indistinguishable`, `tier_map`, `context_flags` | `draftable`, `drafted_players`, `picks` |

Roughly 600 lines survive as the ranking spine. **The spine is what Rankings
sits on — deleting `board.py` wholesale would take the good half with it.**

### 7.3 Test consequences

`tests/test_draft_day.py` is tied to the pick selector and goes wherever that
goes. **The pattern it encodes must not go with it.** It exists because driving
a widget forward, repeatedly, found a segfault that 270 unit tests missed, and
whatever the Rankings surface's primary control turns out to be needs the same
treatment. Port the file; do not delete it.

The pyarrow exclusion and both its guards are format-independent and stay.

---

## 8. Sequencing

The Shiva Bowl draft is **2026-08-22**. Nine days. A restructure of a
2,849-line app in that window is how you arrive at a draft with a broken tool,
so the order is deliberate.

### Phase 0 — before 08-22. Additive only, no restructure.

1. Confirm the roster on Sleeper actually still has `SUPER_FLEX` (§0).
2. Surface `peek.regression_candidates`, `peek.market_disagreement`,
   `peek.snap_trend`, `adp.movement` — new expander in the existing Research
   tab. Zero risk to Draft Day.
3. Add the profile selector (§7.1). This is the pivot's foundation and it is
   independent of the tab layout.
4. Draft.

### Phase 1 — after the draft. The restructure.

5. Split `board.py` (§7.2).
6. Cut `simulate.py`, comparables, concentration-over-time from the product.
7. Rebuild as three surfaces (§6). Port the driven-widget test.
8. Build the composite rank + disagreement column (§5.2).

### Phase 2 — the measurements, in value order.

9. **M1** — outcome spread. The open thread, and the one worth a real week.
10. **M2** — availability. Most likely to produce a useful null.
11. **M3** — `pts_over_exp` as a ranking input, gated on its own measurement.
12. **M4** — age. Cheapest of the four.
13. Cross-market price disagreement (§2.4), building on `adp.multi_format`.

---

## 9. Open questions for Zach

1. **Which leagues?** Team count, roster slots, scoring, keepers y/n for each.
   That is the entire input to a profile, and Phase 0 item 3 needs it.
2. **Is the Shiva Bowl roster actually changing**, or is it superflex-on-paper
   drafting like 1QB? These lead to opposite recommendations (§0).
3. **Your tab ideas.** This proposes three surfaces on first principles. You
   said you had "quite a few ideas on how to reshuffle" — those should be merged
   in here before anything is built, not proposed around.
4. **Does the composite need a stated weighting you agree with**, or should it
   ship showing all three inputs side by side with no blend at all? The second
   is more honest and slower to read on the clock.

---

*This is a spec, not a build. Per `CLAUDE.md`, dashboard work is a spec exercise
before it is a build exercise — nothing in `app.py` changes until this is
agreed. Record where it turns out to be wrong, inline, the way the two dashboard
specs did.*
