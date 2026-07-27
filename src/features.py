"""One row per player-season: what a player's role was, not what he scored.

The whole point is to describe *opportunity*, because opportunity is what
carries forward. Efficiency — catch rate, yards per target, touchdown rate — is
mostly noise year to year; volume and role are stickier, which is why a receiver
who saw 28% of his team's targets is a better bet than one who scored the same
points on 18%.

Everything here is a share or a rate, never a total. A total quietly encodes
games played, so a model trained on totals learns "stayed healthy" and reports
it as insight. `games` is carried separately, where it can be reasoned about.

Four sources, four ways to join them wrong, all of which fail silently:

  ff_opportunity   `season` is a String and `week` a Float64; 7% of rows have a
                   null player_id (unattributed team plays); and it carries
                   weeks 19-22 with no season_type column to filter on.
  snap_counts      no gsis_id at all — only pfr_player_id — and its game_type
                   values are REG/WC/DIV/CON/SB, so filtering on "POST" silently
                   keeps everything.
  nextgen          week == 0 rows are the season aggregate. Averaging the weekly
                   rows instead gives an unweighted mean of per-game averages.
  players          birth_date is a String.

Each is handled explicitly below and asserted in the tests.
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

import polars as pl

from src import ids
from src import nflverse as nv
from src import scoring as sc
from src.cache import frame
from src.config import DEFAULT_SCORING, FANTASY_POSITIONS, FEATURE_SEASONS

# Features grouped by what they describe. `feature_columns()` reads these, and
# so does the app's cluster explorer, so a column added here shows up everywhere.
ROLE_FEATURES = [
    "snap_pct",
    "target_share",
    "air_yards_share",
    "rush_share",
    "tgt_per_game",
    "carry_per_game",
    "adot",
    "wopr",
]
EFFICIENCY_FEATURES = ["catch_rate", "ypt", "ypc", "yac_per_rec"]
EXPECTED_FEATURES = ["exp_ppg", "pts_over_exp_per_game", "exp_pts_share"]
NGS_FEATURES = ["avg_separation", "avg_cushion", "avg_yac_above_expectation"]
CONTEXT_FEATURES = ["age", "seasons_exp", "draft_round", "draft_pick", "undrafted"]

# Carried for labeling and display. Never features — `fantasy_points` is the
# thing being predicted, and feeding it back in is the classic leak.
OUTCOME_COLUMNS = ["games", "fantasy_points", "ppg", "pos_rank"]


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Ratio that yields null rather than inf when the denominator is zero."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def _opportunity(
    seasons: list[int],
    scoring: Mapping[str, float],
    force: bool = False,
) -> pl.DataFrame:
    """Season usage shares and expected points, from ff_opportunity.

    Shares use this table's own `*_team` mirror columns as denominators, which
    are the same team's same weeks by construction. That is both more accurate
    than joining team_stats and cheaper — and it gives a true season share
    (sum of player over sum of team) rather than an average of weekly ratios,
    which would weight a 2-target game the same as a 12-target one.
    """
    opp = nv.ff_opportunity(seasons, stat_type="weekly", force=force).filter(
        pl.col("player_id").is_not_null()
        # No season_type column exists here, and weeks 19-22 are in the file.
        # Without this, playoff production lands in "season usage".
        & (pl.col("week") <= 18)
        & pl.col("position").is_in(list(FANTASY_POSITIONS))
    )
    if not opp.height:
        return pl.DataFrame()

    opp = opp.with_columns(
        pl.col("season").cast(pl.Int32),  # String in the source
        pl.col("week").cast(pl.Int32),  # Float64 in the source
        sc.expected_points_expr(scoring),
    )

    agg = opp.group_by(["season", "player_id"]).agg(
        pl.col("full_name").last().alias("player_name"),
        pl.col("position").last(),
        pl.col("posteam").last().alias("team"),
        pl.len().alias("opp_games"),
        # volume
        pl.col("rec_attempt").sum().alias("targets"),
        pl.col("rush_attempt").sum().alias("carries"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("rec_air_yards").sum().alias("air_yards"),
        pl.col("rec_yards_gained").sum().alias("rec_yards"),
        pl.col("rush_yards_gained").sum().alias("rush_yards"),
        # team denominators, summed over the same weeks the player appears in
        pl.col("rec_attempt_team").sum().alias("team_targets"),
        pl.col("rush_attempt_team").sum().alias("team_carries"),
        pl.col("rec_air_yards_team").sum().alias("team_air_yards"),
        pl.col("total_fantasy_points_exp_team").sum().alias("team_exp_pts"),
        # expected, rebuilt under league scoring
        pl.col("expected_points").sum().alias("exp_pts"),
        pl.col("total_fantasy_points").sum().alias("opp_act_pts"),
    )

    return agg.with_columns(
        _safe_div(pl.col("targets"), pl.col("team_targets")).alias("target_share"),
        _safe_div(pl.col("carries"), pl.col("team_carries")).alias("rush_share"),
        # Signed, unlike the other shares, and correctly so: air yards are
        # measured from the line of scrimmage, so a screen thrown two yards
        # behind it contributes -2. A back targeted exclusively on checkdowns
        # ends the season with negative air yards and therefore a negative
        # share. 347 of 2,610 player-seasons are negative here and every one is
        # a running back with a negative aDOT (Bucky Irving at -3.1, Jaylen
        # Warren at -2.5). Clipping this at zero would erase the distinction
        # between a checkdown back and one who is never targeted at all.
        _safe_div(pl.col("air_yards"), pl.col("team_air_yards")).alias("air_yards_share"),
        _safe_div(pl.col("exp_pts"), pl.col("team_exp_pts")).alias("exp_pts_share"),
        (pl.col("targets") / pl.col("opp_games")).alias("tgt_per_game"),
        (pl.col("carries") / pl.col("opp_games")).alias("carry_per_game"),
        (pl.col("exp_pts") / pl.col("opp_games")).alias("exp_ppg"),
        ((pl.col("opp_act_pts") - pl.col("exp_pts")) / pl.col("opp_games")).alias(
            "pts_over_exp_per_game"
        ),
        _safe_div(pl.col("air_yards"), pl.col("targets")).alias("adot"),
        _safe_div(pl.col("receptions"), pl.col("targets")).alias("catch_rate"),
        _safe_div(pl.col("rec_yards"), pl.col("targets")).alias("ypt"),
        _safe_div(pl.col("rush_yards"), pl.col("carries")).alias("ypc"),
    ).with_columns(
        # WOPR — the standard blend of target share and air-yards share. The
        # weights are convention, not fitted; it is here because it is the one
        # composite the fantasy literature agrees on.
        (1.5 * pl.col("target_share").fill_null(0) + 0.7 * pl.col("air_yards_share").fill_null(0))
        .alias("wopr")
    )


def _snaps(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Season snap share, bridged from pfr_player_id to gsis_id.

    snap_counts is a Pro Football Reference feed and carries no nflverse ID.
    The crosswalk is the only way across, and the match rate is worth printing
    rather than assuming — see `coverage_report`.
    """
    snaps = nv.snap_counts(seasons, force=force).filter(
        # Values are REG/WC/DIV/CON/SB. There is no "POST".
        (pl.col("game_type") == "REG")
        & pl.col("position").is_in(list(FANTASY_POSITIONS))
    )
    if not snaps.height:
        return pl.DataFrame()

    bridge = ids.crosswalk().select("pfr_id", "gsis_id").drop_nulls()

    return (
        snaps.with_columns(pl.col("season").cast(pl.Int32))
        .join(bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .group_by(["season", "gsis_id"])
        .agg(
            pl.col("offense_pct").mean().alias("snap_pct"),
            pl.col("offense_snaps").sum().alias("offense_snaps"),
        )
        .rename({"gsis_id": "player_id"})
    )


def _nextgen(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """NGS receiving separation and YAC-over-expected, season level.

    Uses the `week == 0` rows, which are NGS's own properly weighted season
    aggregate. Coverage is qualified receivers only — roughly 120 per season —
    so these columns are legitimately null for most of the pool and must be
    imputed downstream rather than treated as missing data to drop.
    """
    ng = nv.nextgen("receiving", seasons, force=force).filter(
        (pl.col("week") == 0) & (pl.col("season_type") == "REG")
    )
    if not ng.height:
        return pl.DataFrame()

    return (
        ng.with_columns(pl.col("season").cast(pl.Int32))
        .group_by(["season", "player_gsis_id"])
        .agg(
            pl.col("avg_separation").mean(),
            pl.col("avg_cushion").mean(),
            pl.col("avg_yac_above_expectation").mean(),
            pl.col("percent_share_of_intended_air_yards").mean().alias("intended_ay_share"),
        )
        .rename({"player_gsis_id": "player_id"})
        .drop_nulls("player_id")
    )


def _bio(force: bool = False) -> pl.DataFrame:
    """Draft capital and birth date. The context a usage profile is read against.

    A 26% target share means something different for a 23-year-old second-round
    pick than for a 30-year-old on his third team.
    """
    players = nv.players(force=force).select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("birth_date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
        pl.col("draft_round").cast(pl.Int32, strict=False),
        pl.col("draft_pick").cast(pl.Int32, strict=False),
        pl.col("rookie_season").cast(pl.Int32, strict=False),
    )
    return players.drop_nulls("player_id").unique(subset=["player_id"], keep="first")


def player_seasons(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    min_games: int = 4,
    force: bool = False,
) -> pl.DataFrame:
    """The feature table: one row per player-season.

    `min_games` exists because a two-game sample produces share values that look
    like signal and are not. Four is low enough to keep genuine partial seasons
    (a rookie who took over in week 10) and high enough to drop noise.
    """
    seasons = seasons or FEATURE_SEASONS
    scoring = scoring or DEFAULT_SCORING

    opp = _opportunity(seasons, scoring, force=force)
    if not opp.height:
        return pl.DataFrame()

    outcomes = sc.score_season(seasons, scoring, force=force).select(
        "season",
        "player_id",
        "games",
        "fantasy_points",
        "ppg",
        "pos_rank",
    )

    df = (
        opp.join(outcomes, on=["season", "player_id"], how="inner")
        .join(_snaps(seasons, force=force), on=["season", "player_id"], how="left")
        .join(_nextgen(seasons, force=force), on=["season", "player_id"], how="left")
        .join(_bio(force=force), on="player_id", how="left")
    )

    df = df.with_columns(
        (
            (
                pl.date(pl.col("season"), 9, 1).cast(pl.Date)
                - pl.col("birth_date")
            ).dt.total_days()
            / 365.25
        ).alias("age"),
        (pl.col("season") - pl.col("rookie_season") + 1).alias("seasons_exp"),
        # yards after catch per reception, from the two columns we already have
        _safe_div(
            pl.col("rec_yards") - pl.col("air_yards") * pl.col("catch_rate"),
            pl.col("receptions"),
        ).alias("yac_per_rec"),
        # A null draft_round on a player whose bio we found is not missing data
        # — it means undrafted, and going undrafted is one of the strongest
        # negative priors there is. Verified: the nulls are Taysom Hill, James
        # Robinson, J.D. McKissic and company, all genuine UDFAs. Left as null
        # they would be imputed to the median (~round 4), which would hand every
        # undrafted player a mid-round pedigree he never had.
        (pl.col("rookie_season").is_not_null() & pl.col("draft_round").is_null()).alias(
            "undrafted"
        ),
    ).with_columns(
        # Computed in a second pass so these read the `undrafted` flag above
        # rather than the draft_round they are about to replace.
        pl.when(pl.col("undrafted"))
        .then(pl.lit(8, dtype=pl.Int32))
        .otherwise(pl.col("draft_round"))
        .alias("draft_round"),
        pl.when(pl.col("undrafted"))
        .then(pl.lit(262, dtype=pl.Int32))  # one past the last pick of round 7
        .otherwise(pl.col("draft_pick"))
        .alias("draft_pick"),
    )

    return df.filter(pl.col("games") >= min_games).sort(
        ["season", "position", "pos_rank"]
    )


def cluster_feature_columns(position: str | None = None) -> list[str]:
    """The subset describing *role*, for clustering. Two exclusions, both load-bearing.

    **No production.** `exp_ppg` and `pts_over_exp_per_game` are points. Cluster
    on points and the clusters are scoring tiers wearing a usage costume — you
    get "good players" and "bad players" and then discover, remarkably, that the
    good cluster outscores the bad one. The question this module exists to answer
    is which cheap players are used *like* expensive ones, which is only
    answerable if price and production stay out of the distance metric.

    **No booleans.** `undrafted` standardizes to a large spike on a handful of
    rows, and k-means minimizes squared distance, so it spends its first split
    isolating them. That is exactly what happened: QB's k=2 solution was Taysom
    Hill in one cluster and the other 31 quarterbacks in the other.

    `exp_pts_share` stays. It is a share of one's own offense — a role, not a
    scoring rate.
    """
    role = ROLE_FEATURES + EFFICIENCY_FEATURES + ["exp_pts_share"]
    if position == "QB":
        return [c for c in role if c not in
                ("target_share", "air_yards_share", "adot", "wopr", "catch_rate", "ypt", "yac_per_rec")]
    if position == "RB":
        return [c for c in role if c not in ("air_yards_share", "adot")]
    if position in ("WR", "TE"):
        return [c for c in role if c not in ("rush_share", "ypc")] + NGS_FEATURES
    return role


def feature_columns(position: str | None = None) -> list[str]:
    """The model-facing columns. Position-aware because the sets barely overlap.

    Receiving air-yards features are meaningless for a running back's usage
    profile and rushing share is meaningless for a receiver; including both
    everywhere would mean imputing half of every row.
    """
    base = ROLE_FEATURES + EFFICIENCY_FEATURES + EXPECTED_FEATURES + CONTEXT_FEATURES
    if position is None:
        return base + NGS_FEATURES
    if position == "QB":
        return [
            c
            for c in base
            if c not in ("target_share", "air_yards_share", "adot", "wopr", "catch_rate", "ypt", "yac_per_rec")
        ]
    if position == "RB":
        return [c for c in base if c not in ("air_yards_share", "adot")]
    return [c for c in base if c not in ("rush_share", "ypc")] + NGS_FEATURES


def coverage_report(df: pl.DataFrame) -> pl.DataFrame:
    """Non-null rate per feature, overall and by position.

    Meant to be read, not just run. NGS columns sitting near 35% is expected
    (qualified receivers only); snap_pct below ~90% means the pfr_id bridge
    broke and every role feature downstream is quietly thinner than it looks.
    """
    if not df.height:
        return pl.DataFrame()

    rows = []
    cols = [c for c in feature_columns() if c in df.columns]
    for position in [None, *sorted(df.get_column("position").unique().to_list())]:
        scope = df if position is None else df.filter(pl.col("position") == position)
        for col in cols:
            non_null = int(scope.get_column(col).is_not_null().sum())
            rows.append(
                {
                    "scope": position or "ALL",
                    "column": col,
                    "n": scope.height,
                    "non_null": non_null,
                    "pct": round(non_null / scope.height, 3) if scope.height else 0.0,
                }
            )
    return pl.DataFrame(rows).sort(["scope", "pct"])


def build(force: bool = False) -> pl.DataFrame:
    """The cached feature table under league default scoring."""
    lo, hi = min(FEATURE_SEASONS), max(FEATURE_SEASONS)
    return frame(
        f"features_player_season_{lo}-{hi}",
        "weekly",
        lambda: player_seasons(force=force),
        force,
    )
