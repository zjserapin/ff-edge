"""How positional value has moved, 2018-2025, in your league's scoring.

This is the descriptive track. It does not predict anything; it establishes the
shape of the format you are drafting into, which is the thing most draft advice
assumes rather than measures.

Four questions, four functions:

  par_by_position         Is a position getting more or less valuable over time?
  concentration           Are the top players taking a bigger slice?
  scarcity_curve          What does the dropoff actually look like? (the real
                          draft input — an aggregate PAR number hides the shape)
  cross_positional_value  Where does RB24 sit against WR36 in a given year?
                          This is the early-RB question, stated numerically.

Every frame comes back long/tidy — one row per mark — so the app's chart specs
stay one-liners and the same frame can be re-encoded without reshaping.

All of it is recomputed under whatever scoring is passed in, so a change in the
sidebar propagates to every chart.
"""

from __future__ import annotations

from typing import Mapping

import polars as pl

from src import scoring as sc
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_TEAMS,
    FANTASY_POSITIONS,
)

# Denominator for the concentration metric, per position. Roughly 3x starter
# demand — see `concentration` for why an unbounded denominator is misleading.
DEFAULT_POOL_SIZE: dict[str, int] = {"QB": 32, "RB": 60, "WR": 60, "TE": 32}


def _season_points(
    seasons: list[int] | None,
    scoring: Mapping[str, float] | None,
    season_points: pl.DataFrame | None,
) -> pl.DataFrame:
    """Accept a precomputed frame or build one. Lets the app score history once."""
    if season_points is not None:
        return season_points if seasons is None else season_points.filter(
            pl.col("season").is_in(seasons)
        )
    return sc.score_season(seasons, scoring)


def par_by_position(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
    season_points: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Value above replacement by position and season.

    `par_mean_starter` is the headline: the average edge a starting player at
    this position gives you over the waiver wire. It is the number that answers
    "should I spend early picks here," because it is denominated in the same
    units across positions.

    `par_total` sums positive PAR across the starting pool — a position can have
    a modest mean and still be worth attacking early if the top of it is steep,
    which is what `scarcity_curve` shows and this deliberately does not.

    Returns: season, position, demand, replacement_rank, replacement_ppg,
    par_total, par_mean_starter, par_top3_ppg.
    """
    pts = _season_points(seasons, scoring, season_points)
    par = sc.points_above_replacement(pts, roster_positions, teams, flex_split)
    if not par.height:
        return pl.DataFrame()

    starters = par.filter(pl.col("pos_rank") <= pl.col("demand"))

    agg = starters.group_by(["season", "position"]).agg(
        pl.col("demand").first(),
        pl.col("replacement_rank").first(),
        pl.col("replacement_ppg").first(),
        pl.col("par").clip(lower_bound=0).sum().round(1).alias("par_total"),
        pl.col("par_ppg").mean().round(2).alias("par_mean_starter"),
        pl.col("par_ppg").top_k(3).mean().round(2).alias("par_top3_ppg"),
    )
    return agg.sort(["season", "position"])


def concentration(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    shares: tuple[int, ...] = (5, 15, 30),
    pool_size: Mapping[str, int] | None = None,
    season_points: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """What fraction of positional points the top N players account for.

    The denominator is the entire argument. Counting every player who caught a
    pass makes this metric a function of how many bodies the league cycled
    through — the top-15 share "falls" in a year with more garbage-time
    receivers, which has nothing to do with stars concentrating. Capping the pool
    at roughly three times starter demand asks the question you actually mean:
    among players anyone would consider rostering, how top-heavy is it?

    Returns long: season, position, top_n, share, pool_size, pool_points.
    """
    pts = _season_points(seasons, scoring, season_points)
    if not pts.height:
        return pl.DataFrame()

    pool_size = pool_size or DEFAULT_POOL_SIZE
    rows: list[dict[str, object]] = []

    for (season, position), grp in pts.group_by(["season", "position"], maintain_order=True):
        cap = pool_size.get(str(position))
        pool = grp.sort("pos_rank")
        if cap:
            pool = pool.head(cap)
        total = float(pool.get_column("fantasy_points").sum())
        if total <= 0:
            continue
        for n in shares:
            top = float(pool.head(n).get_column("fantasy_points").sum())
            rows.append(
                {
                    "season": season,
                    "position": position,
                    "top_n": n,
                    "share": round(top / total, 4),
                    "pool_size": pool.height,
                    "pool_points": round(total, 1),
                }
            )

    return pl.DataFrame(rows).sort(["position", "top_n", "season"]) if rows else pl.DataFrame()


def scarcity_curve(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    max_rank: int = 48,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
    basis: str = "ppg",
    min_games: int = 8,
    season_points: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Points versus positional rank — the dropoff curve, by season.

    This is the actual draft-strategy input and the reason the aggregate PAR
    number isn't enough. Two positions can have identical mean PAR while one is
    a cliff at rank 6 and the other a gentle slope to rank 30. The cliff is worth
    reaching for; the slope is worth waiting on. You cannot tell them apart from
    a single number, only from this shape.

    `basis` decides what "rank" means, and the curve is ranked on the same
    quantity it plots — otherwise it isn't a curve, it's noise. Rank by season
    total and plot per-game and you get a sawtooth: the RB27 who missed six games
    outscores the RB24 who played every week on a per-game basis, so the line
    jumps backwards at every injured player. Both bases are monotone by
    construction:

        "total"  season points, ranked on season points. Availability counts as
                 value, which is the right frame for "who should I draft".
        "ppg"    points per game among players with `min_games`, ranked on the
                 same. The right frame for comparing seasons of different length
                 and for asking what a healthy player at this tier is worth.

    Also the frame the Board tab reads to turn an ADP positional rank into a
    market-implied points estimate.

    Returns: season, position, pos_rank, player_display_name, games,
    fantasy_points, ppg, par_ppg.
    """
    if basis not in ("total", "ppg"):
        raise ValueError(f"basis must be 'total' or 'ppg', got {basis!r}")

    pts = _season_points(seasons, scoring, season_points)
    par = sc.points_above_replacement(pts, roster_positions, teams, flex_split)
    if not par.height:
        return pl.DataFrame()

    if basis == "ppg":
        par = par.filter(pl.col("games") >= min_games).with_columns(
            pl.col("ppg")
            .rank("ordinal", descending=True)
            .over(["season", "position"])
            .cast(pl.Int32)
            .alias("pos_rank")
        )

    return (
        par.filter(pl.col("pos_rank") <= max_rank)
        .select(
            "season",
            "position",
            "pos_rank",
            "player_display_name",
            "games",
            "fantasy_points",
            "ppg",
            "par_ppg",
        )
        .sort(["position", "season", "pos_rank"])
    )


def cross_positional_value(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
    top_n: int = 100,
    season_points: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every position on one axis, ranked together by value over replacement.

    PAR per game is the only unit that makes a quarterback and a tight end
    comparable, and once everyone is on it the early-RB argument becomes an
    empirical question with a countable answer: of the top 24 slots in 2024, how
    many were running backs? The app renders exactly that as a stacked bar.

    Returns: season, overall_par_rank, position, pos_rank, player_display_name,
    ppg, par_ppg.
    """
    pts = _season_points(seasons, scoring, season_points)
    par = sc.points_above_replacement(pts, roster_positions, teams, flex_split)
    if not par.height:
        return pl.DataFrame()

    return (
        par.with_columns(
            pl.col("par_ppg")
            .rank("ordinal", descending=True)
            .over("season")
            .cast(pl.Int32)
            .alias("overall_par_rank")
        )
        .filter(pl.col("overall_par_rank") <= top_n)
        .select(
            "season",
            "overall_par_rank",
            "position",
            "pos_rank",
            "player_display_name",
            "ppg",
            "par_ppg",
        )
        .sort(["season", "overall_par_rank"])
    )


def positional_mix(cross: pl.DataFrame, cutoffs: tuple[int, ...] = (12, 24, 36, 48)) -> pl.DataFrame:
    """Position counts within the top-N slots of the combined PAR ranking.

    The summary of `cross_positional_value` that fits in a sentence: "of the top
    24 players by value over replacement in 2024, nine were backs."

    Returns long: season, cutoff, position, n, share.
    """
    if not cross.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for season in cross.get_column("season").unique().sort().to_list():
        year = cross.filter(pl.col("season") == season)
        for cutoff in cutoffs:
            head = year.filter(pl.col("overall_par_rank") <= cutoff)
            if not head.height:
                continue
            counts = head.group_by("position").agg(pl.len().alias("n"))
            for row in counts.iter_rows(named=True):
                rows.append(
                    {
                        "season": season,
                        "cutoff": cutoff,
                        "position": row["position"],
                        "n": row["n"],
                        "share": round(row["n"] / head.height, 4),
                    }
                )

    return pl.DataFrame(rows).sort(["season", "cutoff", "position"]) if rows else pl.DataFrame()


def crossover_table(
    cross: pl.DataFrame,
    anchors: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """For a given player, the rank at every other position worth the same.

    "I'm on the clock and RB24 is there — what receiver is that equivalent to?"
    Read off the shared PAR-per-game axis: find the anchor's par_ppg, then find
    where that value falls in each other position's curve.

    Returns: season, anchor_position, anchor_rank, anchor_par_ppg,
    other_position, equivalent_rank.
    """
    if not cross.height:
        return pl.DataFrame()

    anchors = anchors or {"RB": 24, "WR": 36, "TE": 12, "QB": 12}
    rows: list[dict[str, object]] = []

    for season in cross.get_column("season").unique().sort().to_list():
        year = cross.filter(pl.col("season") == season)
        for anchor_pos, anchor_rank in anchors.items():
            anchor = year.filter(
                (pl.col("position") == anchor_pos) & (pl.col("pos_rank") == anchor_rank)
            )
            if not anchor.height:
                continue
            value = float(anchor.get_column("par_ppg")[0])

            for other in FANTASY_POSITIONS:
                if other == anchor_pos:
                    continue
                # The last player at `other` still worth at least as much.
                better = year.filter(
                    (pl.col("position") == other) & (pl.col("par_ppg") >= value)
                )
                rows.append(
                    {
                        "season": season,
                        "anchor_position": anchor_pos,
                        "anchor_rank": anchor_rank,
                        "anchor_par_ppg": round(value, 2),
                        "other_position": other,
                        "equivalent_rank": int(better.get_column("pos_rank").max())
                        if better.height
                        else 0,
                    }
                )

    return pl.DataFrame(rows).sort(["season", "anchor_position", "other_position"]) if rows else pl.DataFrame()


def market_implied_value(
    adp_board: pl.DataFrame,
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    flex_split: Mapping[str, float] | None = None,
    season_points: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """What a player's draft slot has historically been worth. Not a projection.

    The Board wants value over replacement, which normally needs a points
    projection — and this project deliberately has none, because building one
    honestly is a bigger job than everything else here combined and building one
    dishonestly is worse than having none.

    So invert the question. Rather than "how many points will this player score",
    ask "what has the *slot he is being drafted at* returned historically". Take
    his 2026 ADP positional rank, look up the median points at that rank across
    the seasons in the window, and subtract replacement. It answers: if this
    player performs like the typical player drafted here, what is he worth?

    That is a real and useful number, and it is emphatically not a forecast about
    him. It knows nothing about the player — two backs at RB14 get the same
    figure. Its value is as a *baseline to disagree with*: the interesting column
    on the Board is the gap between this and what you believe.

    Returns the input frame plus market_points, market_ppg, replacement_ppg,
    market_var (value above replacement), and rank_seasons (how many seasons
    supported the estimate).
    """
    if not adp_board.height:
        return adp_board

    curve = scarcity_curve(
        seasons,
        scoring,
        max_rank=200,
        roster_positions=roster_positions,
        teams=teams,
        flex_split=flex_split,
        basis="total",
        season_points=season_points,
    )
    if not curve.height:
        return adp_board

    # Median across seasons, not mean: one wrecked year at a rank should not
    # drag the estimate for that slot.
    by_rank = curve.group_by(["position", "pos_rank"]).agg(
        pl.col("fantasy_points").median().alias("market_points"),
        pl.col("ppg").median().alias("market_ppg"),
        pl.len().alias("rank_seasons"),
    )

    repl = (
        sc.replacement_level(
            _season_points(seasons, scoring, season_points),
            roster_positions,
            teams,
            flex_split,
        )
        .group_by("position")
        .agg(pl.col("replacement_ppg").median())
    )

    return (
        adp_board.join(
            by_rank,
            left_on=["position", "adp_pos_rank"],
            right_on=["position", "pos_rank"],
            how="left",
        )
        .join(repl, on="position", how="left")
        .with_columns(
            (pl.col("market_ppg") - pl.col("replacement_ppg")).round(2).alias("market_var")
        )
    )
