"""League-parameterized scoring and replacement level.

Every number in this project is denominated in *your* league's points. That
sounds obvious and is routinely gotten wrong: public rankings quote full PPR,
nflverse ships `fantasy_points` (standard) and `fantasy_points_ppr` (full), and
neither is the Shiva Bowl's half point per reception. A half-point swing on 100
receptions is 50 points, which is several rounds of draft capital at WR.

Two things live here.

**Scoring.** `points_expr()` compiles a Sleeper `scoring_settings` dict into a
single polars expression over `weekly_stats` columns. That mapping is the crux
of the module — Sleeper names things like `rec_yd` and nflverse names them
`receiving_yards`, and nothing in either project connects the two.

**Replacement level.** The value of a player is not his points, it's his points
above what you could have had for free at the same position. That baseline falls
out of roster slots times teams, not out of convention: in a 10-team league with
a FLEX and a SUPER_FLEX, roughly 60 of RB/WR/TE plus a second wave of
quarterbacks are starting in any given week, and *which* ones depends on how
those slots actually get allocated. That allocation is computed from the
season's real scoring rather than assumed, and exposed as a parameter because it
genuinely moves the answer.

The superflex slot added for 2026 is the single largest rules change this league
has made for valuation purposes — it roughly doubles league-wide QB demand and
drops replacement quarterback about ten positional ranks, which reprices every
quarterback in the draft. `starter_demand` derives that rather than special-
casing it.
"""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from src import nflverse as nv
from src import sleeper
from src.cache import frame
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    FANTASY_POSITIONS,
    FLEX_SLOTS,
    LEAGUE_ID,
    NON_STARTING_SLOTS,
    REGULAR_SEASON_WEEKS,
)

# --- The Sleeper -> nflverse mapping ----------------------------------------

# Each Sleeper scoring key maps to the weekly_stats columns that feed it. A key
# mapping to several columns is summed before the weight is applied.
#
# Three of these are easy to get wrong and all three were checked against the
# real 145-column schema:
#
#   fum_lost  The three components, not `fumbles_lost_total`. They disagree on
#             125 of 24,576 skill player-weeks — the total counts fumbles that
#             nflverse's own scoring excludes. Using the components makes the
#             identity check in the module test exact.
#   st_td     Live in this league at 6.0, and the column exists. Worth 6 points
#             on ~1% of player-weeks, which is invisible until it isn't.
#   fgm_50p   Sleeper has one bucket above 50; nflverse splits 50-59 and 60+.
SCORING_COLUMNS: dict[str, tuple[str, ...]] = {
    # passing
    "pass_yd": ("passing_yards",),
    "pass_td": ("passing_tds",),
    "pass_int": ("passing_interceptions",),
    "pass_2pt": ("passing_2pt_conversions",),
    # rushing
    "rush_yd": ("rushing_yards",),
    "rush_td": ("rushing_tds",),
    "rush_2pt": ("rushing_2pt_conversions",),
    # receiving
    "rec": ("receptions",),
    "rec_yd": ("receiving_yards",),
    "rec_td": ("receiving_tds",),
    "rec_2pt": ("receiving_2pt_conversions",),
    # turnovers and returns
    "fum_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "st_td": ("special_teams_tds",),
    # kicking — weekly_stats carries every bucket, so K is exactly scoreable
    "xpm": ("pat_made",),
    "xpmiss": ("pat_missed",),
    "fgm_0_19": ("fg_made_0_19",),
    "fgm_20_29": ("fg_made_20_29",),
    "fgm_30_39": ("fg_made_30_39",),
    "fgm_40_49": ("fg_made_40_49",),
    "fgm_50p": ("fg_made_50_59", "fg_made_60_"),
    "fgmiss": ("fg_missed",),
}

# ff_opportunity's expected-production columns, same key space. See
# expected_points_expr for why this exists at all.
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "pass_yd": ("pass_yards_gained_exp",),
    "pass_td": ("pass_touchdown_exp",),
    "pass_int": ("pass_interception_exp",),
    "pass_2pt": ("pass_two_point_conv_exp",),
    "rush_yd": ("rush_yards_gained_exp",),
    "rush_td": ("rush_touchdown_exp",),
    "rush_2pt": ("rush_two_point_conv_exp",),
    "rec": ("receptions_exp",),
    "rec_yd": ("rec_yards_gained_exp",),
    "rec_td": ("rec_touchdown_exp",),
    "rec_2pt": ("rec_two_point_conv_exp",),
}


def _weighted_sum(scoring: Mapping[str, float], mapping: Mapping[str, tuple[str, ...]]) -> pl.Expr:
    """Compile weight x column-sum terms into one expression, skipping dead keys."""
    terms = [
        pl.lit(float(weight))
        * pl.sum_horizontal([pl.col(c).fill_null(0) for c in mapping[key]])
        for key, weight in scoring.items()
        if key in mapping and weight
    ]
    return pl.sum_horizontal(terms) if terms else pl.lit(0.0)


def points_expr(scoring: Mapping[str, float] | None = None) -> pl.Expr:
    """Fantasy points under `scoring`, as an expression over weekly_stats columns.

    Works unchanged on `weekly_stats` and `season_stats` because they share a
    schema. This is the primitive the rest of the project reuses — anywhere you
    see points in this repo, they came from here.

    Scoring is by stat line, not by roster slot, which is how Sleeper actually
    works. That is not a pedantic distinction: in 2023 week 9 the Texans lost
    their kicker and running back Dare Ogunbowale made a 29-yard field goal. He
    scores 3 here. nflverse's own `fantasy_points_ppr` gives him 0, because it
    only scores kicking for players listed at K.
    """
    scoring = scoring or DEFAULT_SCORING
    return _weighted_sum(scoring, SCORING_COLUMNS).alias("fantasy_points")


def expected_points_expr(scoring: Mapping[str, float] | None = None) -> pl.Expr:
    """Expected fantasy points under `scoring`, over ff_opportunity columns.

    This exists because `ff_opportunity.total_fantasy_points_exp` is hardcoded to
    **full PPR** and there is no parameter to change it. Verified: a 6-reception,
    77-yard, 0-TD game is scored 13.7, which is 6x1.0 + 77x0.1. Under half-PPR
    every `*_exp` and `*_diff` column in that table is wrong by 0.5 x
    receptions_exp — about 1.5 points per game for a target hog, which is exactly
    the population you'd be using expected points to evaluate.

    Scoring is linear and the table ships every component, so rebuild rather than
    correct. One documented omission: there is no `*_exp` for fumbles lost, worth
    roughly 0.1 points per game.
    """
    scoring = scoring or DEFAULT_SCORING
    return _weighted_sum(scoring, EXPECTED_COLUMNS).alias("expected_points")


def unmapped_keys(scoring: Mapping[str, float] | None = None) -> list[str]:
    """Scoring keys with a live weight and nowhere to get the stat from.

    Always the DST and IDP keys, because `weekly_stats` has no team-defense rows
    at all — points allowed, sacks, and defensive touchdowns are simply not in
    this data layer. Surfaced rather than silently dropped so the app can say
    which parts of your league it isn't modeling.
    """
    scoring = scoring or DEFAULT_SCORING
    return sorted(k for k, w in scoring.items() if w and k not in SCORING_COLUMNS)


# --- League settings --------------------------------------------------------


def resolve_league_id(league_id: str | None = None, force: bool = False) -> str:
    """The league to read, from the argument, the environment, or discovery.

    `LEAGUE_ID` is empty unless FF_EDGE_LEAGUE_ID is exported, so an unset
    environment falls through to asking Sleeper which leagues the configured
    username is in and taking the first. Returns "" when neither is available,
    which callers treat as "use the saved defaults".
    """
    if league_id:
        return league_id
    if LEAGUE_ID:
        return LEAGUE_ID
    try:
        mine = sleeper.my_leagues(force=force)
    except Exception:  # noqa: BLE001 — no network or no username configured
        return ""
    if mine.height and "league_id" in mine.columns:
        return str(mine.get_column("league_id")[0])
    return ""


def league_settings(league_id: str | None = None, force: bool = False) -> dict[str, Any]:
    """Live league parameters, falling back to config defaults when offline.

    The fallback matters more than it looks: the app should open and render on a
    plane, and it should open for someone who cloned this repo and has no league
    configured at all. Defaults are the real 2026 Shiva Bowl values, so the
    offline path is correct rather than merely functional.
    """
    resolved = resolve_league_id(league_id, force=force)
    try:
        meta = sleeper.league(resolved, force=force) if resolved else None
    except Exception:  # noqa: BLE001 — offline is a supported state, not an error
        meta = None

    if not meta:
        return {
            "season": None,
            "teams": DEFAULT_TEAMS,
            "scoring": dict(DEFAULT_SCORING),
            "roster_positions": list(DEFAULT_ROSTER_POSITIONS),
            "playoff_week_start": 15,
            "playoff_teams": 6,
            "source": "config defaults (Sleeper unreachable)",
        }

    settings = meta.get("settings") or {}
    return {
        "season": meta.get("season"),
        "teams": meta.get("total_rosters") or DEFAULT_TEAMS,
        "scoring": meta.get("scoring_settings") or dict(DEFAULT_SCORING),
        "roster_positions": meta.get("roster_positions") or list(DEFAULT_ROSTER_POSITIONS),
        "playoff_week_start": settings.get("playoff_week_start", 15),
        "playoff_teams": settings.get("playoff_teams", 6),
        "source": "sleeper",
    }


def scoring_history(league_id: str | None = None, force: bool = False) -> pl.DataFrame:
    """How this league's own rules have changed, season by season.

    Worth a caption on any chart that plots history under current scoring,
    because the Shiva Bowl is not the league it was: 2023 ran full PPR with one
    FLEX and -1 per interception; 2024-2025 was half-PPR with two FLEX and -2;
    2026 turned one of those FLEX slots into a SUPER_FLEX. Recomputing history
    under today's rules is the right call — you draft into today's rules — but a
    reader should know the past wasn't played that way, and the superflex change
    means past quarterback value in particular is not comparable.

    `flex_slots` counts every multi-position slot; `super_flex_slots` breaks out
    the QB-eligible ones, because that is the count that repriced the position.

    Returns: season, teams, rec, pass_int, flex_slots, super_flex_slots,
    playoff_teams, playoff_week_start.
    """
    resolved = resolve_league_id(league_id, force=force)
    if not resolved:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    for lid, season in sleeper.league_chain(resolved, force=force):
        meta = sleeper.league(lid, force=force)
        if not meta:
            continue
        sc = meta.get("scoring_settings") or {}
        slots = meta.get("roster_positions") or []
        settings = meta.get("settings") or {}
        rows.append(
            {
                "season": int(season) if season else None,
                "teams": meta.get("total_rosters"),
                "rec": sc.get("rec"),
                "pass_int": sc.get("pass_int"),
                "flex_slots": sum(1 for s in slots if s in FLEX_SLOTS),
                "super_flex_slots": sum(
                    1 for s in slots if "QB" in FLEX_SLOTS.get(s, ())
                ),
                "playoff_teams": settings.get("playoff_teams"),
                "playoff_week_start": settings.get("playoff_week_start"),
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("season")


# --- Scoring the data -------------------------------------------------------


def score_weekly(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
    season_type: str = "REG",
    force: bool = False,
) -> pl.DataFrame:
    """One row per player-week, scored under `scoring`.

    Positions are filtered because `weekly_stats` is every rostered player in the
    NFL — there are 10,052 linebacker rows in the 2022-25 file alone, all
    scoring zero, all of which would otherwise land in your positional ranks.

    Returns: player_id, player_display_name, position, team, season, week,
    season_type, receptions, fantasy_points.
    """
    scoring = scoring or DEFAULT_SCORING
    df = nv.weekly_stats(seasons, force=force)

    return (
        df.filter(
            pl.col("position").is_in(list(positions))
            & (pl.col("season_type") == season_type)
        )
        .with_columns(points_expr(scoring))
        .select(
            "player_id",
            "player_display_name",
            "position",
            "team",
            "season",
            "week",
            "season_type",
            # Carried so the half-PPR identity check can run without re-reading
            # the source frame, and so the app can show a PPR/half-PPR delta.
            pl.col("receptions").fill_null(0),
            pl.col("fantasy_points").round(2),
        )
    )


def score_season(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
    weeks: tuple[int, int] = (1, REGULAR_SEASON_WEEKS),
    season_type: str = "REG",
    force: bool = False,
) -> pl.DataFrame:
    """Season totals and positional finish, over the weeks that decide the league.

    `weeks` defaults to (1, 14), not (1, 18). A player's fantasy season ends when
    his manager's does; weeks 15-17 are the playoff and week 18 is starters
    resting. Ranking on an 18-week total credits production that arrived after
    the roster it was on had already been eliminated. Parameterized so the choice
    is arguable rather than buried.

    Returns: season, player_id, player_display_name, position, team, games,
    fantasy_points, ppg, pos_rank.
    """
    lo, hi = weeks
    weekly = score_weekly(seasons, scoring, positions, season_type, force=force).filter(
        pl.col("week").is_between(lo, hi)
    )

    return (
        weekly.group_by(["season", "player_id", "player_display_name", "position"])
        .agg(
            pl.col("team").last(),
            pl.len().alias("games"),
            pl.col("fantasy_points").sum().round(2),
        )
        .with_columns((pl.col("fantasy_points") / pl.col("games")).round(2).alias("ppg"))
        .with_columns(
            pl.col("fantasy_points")
            .rank("ordinal", descending=True)
            .over(["season", "position"])
            .cast(pl.Int32)
            .alias("pos_rank")
        )
        .sort(["season", "position", "pos_rank"])
    )


# --- Replacement level ------------------------------------------------------


def _dedicated_slots(roster_positions: list[str], teams: int) -> dict[str, float]:
    """Starters demanded by name-matched slots alone, ignoring every flex type."""
    counts: dict[str, float] = {}
    for slot in roster_positions:
        if slot in NON_STARTING_SLOTS or slot in FLEX_SLOTS:
            continue
        counts[slot] = counts.get(slot, 0.0) + teams
    return counts


def _flex_counts(roster_positions: list[str], teams: int) -> dict[str, int]:
    """League-wide count of each flex slot type on the roster."""
    counts: dict[str, int] = {}
    for slot in roster_positions:
        if slot in FLEX_SLOTS:
            counts[slot] = counts.get(slot, 0) + teams
    return counts


def _greedy_flex(
    season_points: pl.DataFrame,
    dedicated: dict[str, float],
    flex_counts: dict[str, int],
    flex_eligible: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Allocate every flex slot to whichever eligible position is best at its
    next open rank.

    A convention ("FLEX is mostly RB") is a guess. This is the same question
    asked of the data: with RB20 and WR20 already starting, is RB21 or WR21 worth
    more? Take that one, then ask again. Repeating it once per flex slot is
    exactly how the marginal starter gets chosen in a real lineup, and it means
    the replacement baseline reflects the season's actual shape rather than a
    received opinion about what flex spots are for.

    **Slot types are filled most-restrictive-first**, and that ordering is what
    makes the result exactly optimal rather than merely reasonable. A permissive
    slot can always take what a restrictive one could have taken, so letting the
    restrictive slot choose first never costs anything, while the reverse can:
    if SUPER_FLEX grabbed WR21 before FLEX chose, FLEX would be left picking
    from a strictly smaller set. The argument holds whenever the eligibility
    sets nest or are disjoint, which covers dedicated ⊂ FLEX ⊂ SUPER_FLEX — this
    league's shape. A roster carrying both REC_FLEX and WRRB_FLEX would have two
    sets that overlap without nesting, and there this is a good heuristic rather
    than a guarantee.

    Superflex is where this stops being an accounting detail. A QB-eligible slot
    pulls quarterbacks into the marginal-starter comparison, and in half-PPR the
    QB20 the greedy is weighing is worth far more than the RB21 or WR21 it is
    weighed against — so a superflex slot goes to a quarterback essentially
    every time, league-wide QB demand roughly doubles, and replacement
    quarterback falls from about QB11 to about QB21. Nothing about that is
    assumed here; it falls out of the same comparison the FLEX slots go through.
    """
    positions = sorted({p for slot in flex_counts for p in FLEX_SLOTS[slot]})
    if flex_eligible is not None:
        # An explicit override replaces the plain FLEX slot's eligibility only;
        # a SUPER_FLEX still gets to consider quarterbacks.
        positions = sorted(set(positions) | set(flex_eligible))

    ranked = {
        pos: season_points.filter(pl.col("position") == pos)
        .sort("pos_rank")
        .get_column("fantasy_points")
        .to_list()
        for pos in positions
    }
    filled = {pos: int(round(dedicated.get(pos, 0))) for pos in positions}

    for slot in sorted(flex_counts, key=lambda s: len(FLEX_SLOTS[s])):
        eligible = (
            flex_eligible
            if (slot == "FLEX" and flex_eligible is not None)
            else FLEX_SLOTS[slot]
        )
        for _ in range(flex_counts[slot]):
            best, best_val = None, float("-inf")
            for pos in eligible:
                idx = filled.get(pos, 0)
                val = ranked[pos][idx] if idx < len(ranked.get(pos, [])) else float("-inf")
                if val > best_val:
                    best, best_val = pos, val
            if best is None:
                break
            filled[best] += 1

    return {pos: float(filled[pos] - round(dedicated.get(pos, 0))) for pos in positions}


def starter_demand(
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    season_points: pl.DataFrame | None = None,
    flex_split: Mapping[str, float] | None = None,
    flex_eligible: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """How many of each position are starting league-wide in a given week.

    Dedicated slots are arithmetic: two RB slots across ten teams is twenty
    starting backs. The flex slots are the interesting part, and there are two
    ways to settle them:

      flex_split given      Allocate proportionally. This is the app's slider.
      season_points given   Allocate greedily from the real data (the default,
                            and the honest one).
      neither               Split evenly across eligible positions — a last
                            resort that only applies when no data was passed.

    **`flex_split` only governs the slots it can describe.** The slider names
    shares for RB/WR/TE, which is enough to express a FLEX slot and not enough
    to express a SUPER_FLEX, whose whole question is whether the slot goes to a
    quarterback. So any slot type whose eligible positions are all named in the
    split is allocated proportionally, and the rest are computed greedily even
    when a split was passed. Letting the slider silently decide superflex would
    hard-code the answer to the one question that slot exists to ask.

    The answer moves the entire analysis: replacement RB is somewhere around
    RB27-30 in this format, and a naive "RB is half the flex" assumption can miss
    that by five ranks, which is a full round of draft capital. Adding a
    superflex slot moves replacement quarterback about ten ranks, which is
    larger than any other single rules change this league has made.
    """
    roster_positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    demand = _dedicated_slots(roster_positions, teams)
    flex_counts = _flex_counts(roster_positions, teams)
    if not flex_counts:
        return demand

    split_slots: dict[str, int] = {}
    if flex_split:
        named = {pos for pos, w in flex_split.items()}
        split_slots = {
            slot: n
            for slot, n in flex_counts.items()
            if set(FLEX_SLOTS[slot]) <= named
        }

    total = sum(flex_split.values()) if flex_split else 0.0
    for slot, n in split_slots.items():
        for pos, w in (flex_split or {}).items():
            demand[pos] = demand.get(pos, 0.0) + n * (w / (total or 1.0))

    remaining = {s: n for s, n in flex_counts.items() if s not in split_slots}
    if not remaining:
        return demand

    if season_points is not None and season_points.height:
        extra = _greedy_flex(season_points, demand, remaining, flex_eligible)
    else:
        extra = {}
        for slot, n in remaining.items():
            eligible = (
                flex_eligible
                if (slot == "FLEX" and flex_eligible is not None)
                else FLEX_SLOTS[slot]
            )
            for pos in eligible:
                extra[pos] = extra.get(pos, 0.0) + n / len(eligible)

    for pos, add in extra.items():
        demand[pos] = demand.get(pos, 0.0) + add
    return demand


def replacement_level(
    season_points: pl.DataFrame,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
    min_games: int = 8,
) -> pl.DataFrame:
    """The last startable player at each position, per season.

    Replacement is the first player *past* the starting pool — if 28 backs start,
    replacement is RB29, because that's who you could have had instead of
    drafting one. Computed per season so the baseline moves with the era rather
    than being pinned to whatever this year happens to look like.

    The two baselines are ranked on different columns on purpose, and this is not
    a detail. Rank by season total and read that player's per-game average and
    you get whoever happened to land on the slot — in 2025 that was Tucker Kraft,
    TE11 on total points in 8 games, whose 12.65 ppg is 60% above the actual
    streaming baseline and would have depressed every tight end's PAR in the
    league. Season totals answer "who did you draft"; per-game answers "what can
    you stream", so per-game ranks on per-game among players who were available
    often enough to stream (`min_games`).

    Returns: season, position, demand, replacement_rank, replacement_points,
    replacement_ppg_rank, replacement_ppg.
    """
    rows: list[dict[str, Any]] = []
    for season in season_points.get_column("season").unique().sort().to_list():
        year = season_points.filter(pl.col("season") == season)
        demand = starter_demand(roster_positions, teams, year, flex_split)

        for position, n in demand.items():
            pool = year.filter(pl.col("position") == position)
            if not pool.height:
                continue

            by_total = pool.sort("pos_rank")
            idx = min(int(round(n)), by_total.height - 1)
            total_row = by_total.row(idx, named=True)

            by_ppg = pool.filter(pl.col("games") >= min_games).sort("ppg", descending=True)
            if not by_ppg.height:
                by_ppg = by_total
            pidx = min(int(round(n)), by_ppg.height - 1)
            ppg_row = by_ppg.row(pidx, named=True)

            rows.append(
                {
                    "season": season,
                    "position": position,
                    "demand": round(float(n), 1),
                    "replacement_rank": int(total_row["pos_rank"]),
                    "replacement_points": float(total_row["fantasy_points"]),
                    "replacement_ppg_rank": pidx + 1,
                    "replacement_ppg": float(ppg_row["ppg"]),
                }
            )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def points_above_replacement(
    season_points: pl.DataFrame,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    """Attach PAR to a season-points frame.

    PAR is the only number in this project that compares across positions. Raw
    points say a QB is worth more than a running back; PAR says whether he's
    worth more *than the QB you'd have gotten anyway*, which is the question the
    draft actually poses.

    Adds: demand, replacement_rank, replacement_points, replacement_ppg, par,
    par_ppg.
    """
    repl = replacement_level(season_points, roster_positions, teams, flex_split)
    if not repl.height:
        return season_points

    return (
        season_points.join(repl, on=["season", "position"], how="left")
        .with_columns(
            (pl.col("fantasy_points") - pl.col("replacement_points")).round(2).alias("par"),
            (pl.col("ppg") - pl.col("replacement_ppg")).round(2).alias("par_ppg"),
        )
    )


def kicker_baseline(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Weekly kicker scoring distribution, per season.

    Feeds the simulator, which does not draft named kickers. Cheap to compute
    because every field-goal bucket and PAT column is already in `weekly_stats`,
    so unlike DST this is real data rather than an assumed constant.

    Returns: season, kickers, mean_weekly, sd_weekly.
    """
    weekly = score_weekly(seasons, scoring, positions=("K",), force=force)
    if not weekly.height:
        return pl.DataFrame()

    return (
        weekly.group_by("season")
        .agg(
            pl.col("player_id").n_unique().alias("kickers"),
            pl.col("fantasy_points").mean().round(3).alias("mean_weekly"),
            pl.col("fantasy_points").std().round(3).alias("sd_weekly"),
        )
        .sort("season")
    )


def cached_season_points(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """`score_season` under the league's default scoring, memoized to parquet.

    Only the default scoring is cached. A user twiddling the sidebar gets a live
    recompute, which is fast enough (a few hundred ms) not to be worth a cache
    key per scoring permutation.
    """
    if scoring is not None and dict(scoring) != DEFAULT_SCORING:
        return score_season(seasons, scoring, force=force)

    seasons = seasons or None
    key = "all" if seasons is None else f"{min(seasons)}-{max(seasons)}"
    return frame(
        f"season_points_{key}",
        "weekly",
        lambda: score_season(seasons, DEFAULT_SCORING, force=force),
        force,
    )
