# ff-edge analysis spec

Design agreement for the three analysis tracks and the Streamlit app that fronts
them. Written 2026-07-27, after the data layer was built and verified.

## Decisions locked in interview

| Decision | Choice |
|---|---|
| Interface | Local Streamlit app; static artifact left open for later |
| Format | Parameterized, defaulting to Shiva Bowl |
| Strategy evidence | Simulation over NFL history, not the 210-matchup league sample |
| Player model | Clustering for explanation + supervised backtest for validation |
| Breakout label | Beat ADP by positional tier |
| Rookies | Separate model, kept visibly distinct |
| App layout | Four tabs: Landscape / Players / Strategy / Board |
| Strategy output | General format-level findings, not slot-specific |

**Shiva Bowl parameters** (pulled from the live league, not assumed): half-PPR
(0.5/rec), 10 teams, `QB/RB/RB/WR/WR/TE/FLEX/FLEX/K/DEF` + 6 bench, 14-week
regular season, playoffs week 15.

## Data constraints found by probing

Two findings that shape the build. Both were established by running against the
live APIs, not assumed.

**1. FFC has no 2025 ADP, at any format.** All twelve scoring×team-count
combinations return 0 rows. Coverage for half-PPR/10-team is:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 222 | 196 | 208 | 222 | 124 | 199 | 178 | **0** | 200 |

This is better than it first looked. The backtest doesn't need 2025 — it needs
*many* labeled seasons, and there are **seven** (2018-2024). That's roughly 1,400
labeled player-seasons, which is a workable n for a simple model. 2025 is dropped
as a label year and still used for current-state features.

**2. Feature availability splits into two tiers**, because `ff_opportunity`
only starts in 2022 while the backtest reaches back to 2018.

- **Tier A — deep history (2012+)**: box score production, snap share, target
  share, air yards share, draft capital, age, PFR advanced stats. Available for
  every label season.
- **Tier B — rich but shallow (2022+)**: `ff_opportunity` expected/diff, Next Gen
  Stats, FTN charting.

Design consequence: **the backtested model trains on Tier A only**, so it gets
seven honest seasons instead of two contaminated by missingness. Tier B feeds the
*current-year* clustering and the player detail views, where we only need 2025
data and depth doesn't matter. Mixing them would mean either imputing four years
of the most predictive columns or throwing away 70% of the training set.

## Track 1 — Landscape

Descriptive meta-analysis of how fantasy scoring has moved, 2018-2025, in Shiva
Bowl scoring. New module: `scoring.py` (league-parameterized point calculation
and replacement level) + `landscape.py`.

Metrics:

- **Points above replacement by position, by season.** Replacement level derived
  from actual roster slots × teams, not a convention. In a 10-team league with 2
  FLEX, replacement RB is roughly RB30-38 depending on FLEX allocation — this is
  computed, and the FLEX assumption is exposed as a slider because it genuinely
  moves the answer.
- **Concentration**: top-15 (and top-5, top-30) share of total positional points
  over time. Directly answers "are the top players taking a bigger slice."
- **Positional scarcity curves**: points vs positional rank, overlaid by season.
  The shape of the dropoff is the actual draft-strategy input; the aggregate PAR
  number hides it.
- **Cross-positional value**: where RB1-24 sits relative to WR1-36 in a given
  year, which is what determines whether early-RB is defensible in this format.

Everything is recomputed under the user's scoring, so a change in the scoring
sidebar propagates to every chart.

## Track 2 — Players

Three modules: `features.py`, `archetypes.py`, `breakout.py`, plus `rookies.py`.

**`features.py`** builds one row per player-season with Tier A and Tier B feature
sets clearly separated. Opportunity metrics are expressed as *shares and rates*,
not totals, so a player isn't flagged simply for having played 17 games.

**`archetypes.py`** — unsupervised clustering (k-means on standardized features,
with the cluster count chosen by silhouette and sanity-checked by eye) on the
current season's Tier B features, fit per position. Output is "this WR's usage
profile sits in the same cluster as these established WR1s," which is
interpretable and is the thing you asked for: mid-round players whose
*characteristics* resemble top-tier players.

Clusters describe. They do not predict, and the app will say so.

**`breakout.py`** — the validation half. Label: a player "beat ADP" if their
end-of-season positional finish exceeded their ADP-implied positional tier by a
threshold. Trained on 2018-2024 Tier A features (prior-season features → next-
season label), evaluated with **season-forward splits** — train on earlier years,
test on later ones. Never a random split; random splits leak in time-series data
and would produce a flattering, meaningless score.

Deliverable is honest calibration, not a leaderboard: if the model's top decile
beats ADP 30% of the time against a 20% base rate, that's the finding, and the
app reports it as such. If it doesn't clear base rate, that is also a finding and
will be shown rather than buried.

**`rookies.py`** — separate model on draft capital, combine, landing-spot
opportunity (vacated targets/carries on the new team), and age. Presented in its
own section, never merged into veteran clusters.

## Track 3 — Strategy

`simulate.py`. Monte Carlo over Shiva Bowl rules using real 2018-2025 player
scoring.

- **Draft sim**: nine opponents draft from FFC ADP with noise scaled by each
  player's observed `stdev` — reusing the `survival()` machinery already built,
  which calibrates realistically to how players actually fall. Your team follows
  a named strategy.
- **Season sim**: draw actual historical weekly scores for drafted players,
  set optimal-ish lineups, play a 14-week schedule, run the week-15 playoff.
- **Strategies tested**: zero-RB, hero-RB, balanced, robust-RB, early-TE, late-QB,
  and QB/WR stacking.
- **Output**: win rate, playoff rate, and title rate per strategy with confidence
  intervals across thousands of seasons. General format-level findings, per your
  answer — not slot-conditioned.

**Stated limitation up front**: this sims *drafting*, not in-season management.
Waivers and trades are a large share of real outcomes and are out of scope, so
these results describe how much a draft strategy is worth holding management
constant. That's a narrower claim than "what wins," and the app will phrase it
that way.

## App

`app.py`, four tabs, sidebar holds league parameters (scoring, teams, roster
slots) that propagate everywhere.

| Tab | Contents |
|---|---|
| Landscape | PAR by position over time, concentration, scarcity curves |
| Players | Cluster explorer, breakout scores w/ calibration, rookie board, player detail |
| Strategy | Sim results by strategy, with CIs and the management caveat |
| Board | Draft-day view: VOR, ADP survival, your queue, cut/exclude players |

The Board tab is where "make changes, cuts" lives — exclusions persist in session
state and propagate to the available-player pool and survival math.

New dependencies: `streamlit`, `scikit-learn`, `altair`. `scikit-learn` moves
from the notebook group into the main dependency set.

## Phasing

Each phase is independently useful and independently verifiable.

1. **`scoring.py` + `landscape.py` + app shell.** Foundational — replacement
   level feeds every other track. Ships tab 1.
2. **`features.py` + `archetypes.py`.** Ships the cluster explorer.
3. **`breakout.py` + `rookies.py`.** Ships the backtest and calibration.
4. **`simulate.py`.** Ships tab 3.
5. **Board tab** on top of the finished pieces.

## Risks

- **Seven label seasons is workable, not comfortable.** Expect modest signal.
  Guarding against overfit with season-forward splits and a deliberately simple
  model; will report base rates alongside every result.
- **Regime change.** 2018 fantasy football is not 2026 fantasy football. Will
  check whether the model's edge decays on recent test years, and say so if it
  does.
- **Sim realism.** Drawing historical weekly scores preserves each player's real
  volatility but assumes 2026 usage resembles the past. It is a strategy
  comparison engine, not a projection system.
- **Clustering is not causal.** A mid-round WR clustering with alphas means his
  usage profile rhymes with theirs, not that he will produce like them.
