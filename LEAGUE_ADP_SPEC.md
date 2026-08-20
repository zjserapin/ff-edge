# LEAGUE_ADP_SPEC.md — pricing players at what they cost *here*

Status: **measured, not built.** 2026-08-19. Written in response to "can we make
ADP league-specific?", and the short answer is *partly, and not in the way the
question assumes*. Read the Findings before the Design — the measurement changed
what the design should be.

---

## The question

The board prices every player off Fantasy Football Calculator's national ADP.
That is a market of strangers. The complaint is that it does not reflect what a
player actually costs in the Shiva Bowl, which makes PAR, tiers and
cost-of-waiting hard to trust.

## What is and is not possible

**Not possible: a per-player league ADP.** Sleeper gives us three prior Shiva
Bowl drafts (2023, 2024, 2025 — `sleeper.draft_history` already walks the
`previous_league_id` chain and `bootstrap` already caches them). That is at most
three observations per player, and for the players who matter in 2026 it is
usually zero: Trey McBride has never been drafted in this league at anything
resembling his 2026 price. **You cannot observe a 2026 league price for a
player from drafts that happened before he was worth this much.** Any module
claiming to do so is fitting noise.

**Possible, and what was actually asked for: a league-calibrated *cost*.**
Estimate how this league's draft order departs from the national one *as a
function of position*, then apply that departure to the 2026 national board.
450 picks across three drafts is enough to estimate a handful of per-position
offsets, and nothing more than that.

## Method

For each prior draft:

1. Split picks into keepers (`is_keeper`) and real selections.
2. Remove the kept players from that season's national ADP board, then rank
   what remains — that is the **expected selection index**, the order the
   national market says the pool goes in once keepers are gone.
3. Rank the real selections in pick order — the **actual selection index**.
4. `resid = actual - expected`. Negative means the league takes him *earlier*
   than the national market.

Working on selection *index* rather than pick number is deliberate: it isolates
league behaviour from the keeper mechanics that `board.keeper_adjusted_adp`
already handles. Mixing them would count the keeper adjustment twice, the same
way `adj_adp` and `exp_pick` come apart in that function.

Match rate to national ADP: 93% / 91% / 82% (2023/2024/2025).

## Findings

Residuals inside the top 100, centered on each draft's own median so two leagues
with different pool depths can be compared:

| pos | Shiva 2023 | Shiva 2024 | Shiva 2025 | 828 2024 | 828 2025 |
|---|---|---|---|---|---|
| WR | -1.5 | 0.0 | +1.5 | -0.5 | +0.5 |
| RB | +3.5 | +1.0 | -1.0 | +1.0 | -0.5 |
| **TE** | **-20.0** | **-9.5** | **-9.5** | **-16.0** | **-10.0** |
| **QB** | **+8.5** | **+11.5** | **+18.0** | 0.0 | +4.0 |

Three results, and they point in different directions.

**1. WR and RB are national. This is an honest null and it should stay one.**
Both sit inside ±3.5 picks in all five drafts, which is smaller than the
season-to-season wobble. There is no correction to make for two thirds of the
board, and inventing one would be the "don't invent an ordering the data doesn't
support" failure in `CLAUDE.md`.

**2. The tight-end effect is not league-specific — FFC is wrong about tight
ends.** TEs went ~10 picks earlier than national ADP in **all five drafts across
two unrelated leagues** with different scoring, different sizes and different
managers. Nothing about that is a Shiva Bowl habit. The honest reading is that
FFC's market and real Sleeper home leagues disagree about tight ends by about a
round, consistently — and since Zach drafts in a Sleeper home league, the home
leagues are the population that governs his cost.

The individual reaches are large and unmistakable: Evan Engram national 112 →
taken 48th; Kyle Pitts, three seasons running, at -31, -24, -33.

This joins the Derrick Henry / Josh Jacobs rows as a **third** independent
reason not to take an FFC number at face value. It is a market correction, not
a league correction, and it applies to the 828 board too.

**3. The one genuinely league-specific signal is the one 2026 destroys.** The
Shiva Bowl let quarterbacks slide relative to the national market — and by a
*growing* margin (+8.5 → +11.5 → +18), against 828 at roughly zero. That is a
real, stable, league-specific tendency.

**It must not be applied to the 2026 board.** Those three seasons were 1QB. 2026
is superflex, and 13 of the league's 18 declared keepers are quarterbacks. The
tendency was measured under a roster format that no longer exists, and the
format changed in exactly the direction that reverses it. A model that fit these
offsets naively would tell you quarterbacks fall 13 picks *later* than the market
says, in the one season this league will hunt them hardest.

This is the same class of error as `config.SIM_ROSTER_POSITIONS` being pinned to
the old two-FLEX format: a number that was right about a format, read as though
it were right about a league.

## The design point this uncovered: ADP is doing two jobs

`adp` is currently the input to both, and conflating them is most of the
confusion in the original question.

**ADP as value.** `expected.adp_curve` maps `(position, adp_pos_rank)` to
expected points, and `board.build` derives `par` from that. This is a
*projection*: the market's rank order is being used as an estimate of how good a
player is.

**ADP as cost.** `adj_adp`, `exp_pick`, `adp.survival`, `board.cost_of_waiting`
and every "will he last to my pick" question. This is a *price*.

A league (or market) correction belongs **only on the cost side.** Feeding a
shifted ADP into `adp_curve` would assert that this league reaching for tight
ends makes tight ends score more points, which is circular — the reach is
evidence about the drafters, not the player. Left uncorrected on the cost side,
the board keeps telling you McBride is available at pick 53 when the last five
comparable drafts say he goes around 40.

**Rule: correct the price, never the projection.** Any implementation that
changes `exp_points` or `par` as a side effect is wrong.

## Proposed design

New module `src/market.py` — separate from `adp.py`, which stays the FFC client.

- `market.draft_bias(league_ids, seasons_back=4) -> pl.DataFrame`
  Runs the measurement above and returns `position, n, median_resid, seasons`.
  Reports per-season values, not just the pooled median, because a correction
  whose sign flips across seasons should be visible rather than averaged away.

- `market.apply_bias(board, bias, cap_rank=110) -> pl.DataFrame`
  Shifts `adj_adp` and recomputes `exp_pick`, adding `league_adj_adp`,
  `league_exp_pick` and `bias_applied`. **Adds columns, never overwrites** —
  the uncorrected number has to stay readable next to the corrected one, the
  same reason `board.compare_baselines` runs both ways.

Where the numbers come from is a config decision, not a fitted one:

- `TE: -10` — applied. Five drafts, two leagues, consistent sign.
- `WR / RB: 0` — measured null, stated explicitly rather than omitted.
- `QB: 0` — **deliberately not applied in 2026**, with the format-break reason
  written into the config next to it so it is not "fixed" later.

The board then renders `exp_pick` from the corrected column, and
`adp.survival(adp_col="league_exp_pick")` prices availability off it.

## Traps in this measurement, for whoever touches it next

**The deep board is censored, and the censoring looks exactly like a finding.**
Past national index ~110 the residuals go to -21, -24, -47. That is not the
league reaching. Only 150 picks happen, so the deep national-ADP players you
*observe* are by construction the ones this league liked; the ones it passed on
contribute no row at all. Fitting anything past ~110 fits survivorship. Hence
`cap_rank=110`.

**The per-position medians rest on 7-12 tight ends per draft.** The result
survives because the sign is the same in five independent drafts, not because
any one draft is well-powered. Report the per-season spread, always.

**Kickers move and it does not matter.** K ran -30, -11, -3.5 across the three
Shiva drafts — a fading habit, on a position that carries no PAR. Do not model
it.

**Centering is required to compare two leagues.** Raw residuals carry a global
offset from how deep each league's pool runs relative to FFC's board. The
uncentered 828 2024 numbers read WR +4.5, RB +6.0 — that offset is an artifact
of pool depth, not six positions' worth of reaching.

## What this changes on the 2026 board, concretely

Three tight ends sit inside the top 100. With `TE: -10` on the cost side:

| player | national adp | adj_adp now | adj_adp corrected |
|---|---|---|---|
| Trey McBride | 65.8 | 47.8 | 37.8 |
| Colston Loveland | 81.8 | 63.8 | 53.8 |
| Tyler Warren | 89.2 | 71.2 | 61.2 |

`par`, `tier` and `board_rank` are untouched by construction. What moves is the
answer to "can I wait" — McBride goes from *probably there at 53* to *plan on
40*, which is the difference between a third-round decision and a fourth-round
regret.
