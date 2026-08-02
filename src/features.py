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

from src import context as cx
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
        pl.col("pass_attempt").sum().alias("pass_attempts"),
        pl.col("pass_yards_gained").sum().alias("pass_yards"),
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


def _routes(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Routes run per player-season, from play participation.

    A route is counted when the player was on the field for a *dropback*. That
    denominator matters more than any single metric added here: snap share tells
    you a player was on the field, but a tight end who blocks on half his snaps
    and a receiver split wide have very different route counts from identical
    snap shares.

    Dropbacks are identified by `number_of_pass_rushers > 0`, which correctly
    includes sacks and scrambles — routes are still run on those. It yields
    22,002 plays in 2025 against a true league dropback count of roughly 21,000
    (19k attempts + 1.2k sacks + 0.8k scrambles). The obvious alternatives are
    wrong: `was_pressure` is populated on every row from 2023 on and gives
    45,175, roughly double.

    Yards per route run is the metric this exists to enable. It is the cleanest
    public separator of "good but blocked by a teammate" from "not good", which
    is precisely the distinction a target-share-based view cannot make.

    Returns: season, player_id, routes, team_dropbacks, route_share.
    """
    part = nv.participation(seasons, force=force)
    if not part.height:
        return pl.DataFrame()

    dropbacks = part.with_columns(
        pl.col("nflverse_game_id").str.slice(0, 4).cast(pl.Int32).alias("season")
    ).filter(
        pl.col("season").is_in(seasons)
        & (pl.col("number_of_pass_rushers") > 0)
        & (pl.col("offense_players").str.len_chars() > 0)
    )
    if not dropbacks.height:
        return pl.DataFrame()

    per_player = (
        dropbacks.select(
            "season",
            "possession_team",
            pl.col("offense_players").str.split(";").alias("player_id"),
        )
        .explode("player_id")
        .filter(pl.col("player_id").is_not_null() & (pl.col("player_id") != ""))
        .group_by(["season", "player_id"])
        .agg(
            pl.len().alias("routes"),
            pl.col("possession_team").mode().first().alias("_team"),
        )
    )

    team_totals = (
        dropbacks.group_by(["season", "possession_team"])
        .agg(pl.len().alias("team_dropbacks"))
        .rename({"possession_team": "_team"})
    )

    return (
        per_player.join(team_totals, on=["season", "_team"], how="left")
        .with_columns(
            _safe_div(pl.col("routes"), pl.col("team_dropbacks")).alias("route_share")
        )
        .drop("_team")
    )


def _pfr_quality(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Charted per-touch quality from Pro Football Reference.

    These are the columns that describe a player independent of how much work he
    gets: does he drop passes, does he break tackles, does he gain yards after
    contact. A back on a bad offensive line has poor yards per carry and can
    still be excellent after contact — that gap is the whole point.

    Bridged from `pfr_player_id` through the crosswalk, since this feed carries
    no nflverse id.

    Returns: season, player_id, drop_rate, rec_broken_tackles, receiving_rat,
    rush_broken_tackles, yards_after_contact_per_att.
    """
    bridge = ids.crosswalk().select("pfr_id", "gsis_id").drop_nulls()

    rec = nv.pfr_advstats("rec", seasons, force=force)
    rush = nv.pfr_advstats("rush", seasons, force=force)
    if not rec.height and not rush.height:
        return pl.DataFrame()

    parts = []
    if rec.height:
        parts.append(
            rec.filter(pl.col("game_type") == "REG")
            .group_by(["season", "pfr_player_id"])
            .agg(
                pl.col("receiving_drop").sum().alias("_drops"),
                pl.col("receiving_drop_pct").mean().alias("drop_rate"),
                pl.col("receiving_broken_tackles").sum().alias("rec_broken_tackles"),
                pl.col("receiving_rat").mean().alias("receiving_rat"),
            )
        )
    if rush.height:
        parts.append(
            rush.filter(pl.col("game_type") == "REG")
            .group_by(["season", "pfr_player_id"])
            .agg(
                pl.col("rushing_broken_tackles").sum().alias("rush_broken_tackles"),
                pl.col("rushing_yards_after_contact_avg")
                .mean()
                .alias("yards_after_contact_per_att"),
            )
        )

    merged = parts[0]
    for extra in parts[1:]:
        merged = merged.join(extra, on=["season", "pfr_player_id"], how="full", coalesce=True)

    return (
        merged.with_columns(pl.col("season").cast(pl.Int32))
        .join(bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .drop("pfr_player_id")
        .rename({"gsis_id": "player_id"})
    )


def _competition(seasons: list[int], base: pl.DataFrame) -> pl.DataFrame:
    """How much of his own offense is spoken for by teammates.

    This is what makes a good player cheap, and what makes him a buy when the
    teammate leaves. A receiver with a 15% target share behind a 30% alpha is a
    different proposition from one with 15% on a team where nobody clears 18% —
    the first is capped by someone else, the second is simply not winning.

    Returns: season, player_id, teammate_top_share, teammate_share,
    team_target_hhi.
    """
    if not base.height:
        return pl.DataFrame()

    receivers = base.filter(pl.col("position").is_in(["WR", "TE", "RB"])).select(
        "season", "team", "player_id", "target_share"
    )

    team = receivers.group_by(["season", "team"]).agg(
        pl.col("target_share").sum().alias("_team_share"),
        (pl.col("target_share") ** 2).sum().alias("team_target_hhi"),
        pl.col("target_share").top_k(2).alias("_top2"),
    ).with_columns(
        pl.col("_top2").list.get(0, null_on_oob=True).alias("_max_share"),
        pl.col("_top2").list.get(1, null_on_oob=True).alias("_second_share"),
    )

    joined = receivers.join(team, on=["season", "team"], how="left")

    return joined.with_columns(
        # The biggest target share on the roster *other than his own*. For a
        # team's alpha that is the second-largest share, not null — leaving it
        # null would mean every alpha gets median-imputed into looking like he
        # is competing with an average teammate, when the real answer is that he
        # is the one being competed with. This is the column that separates
        # "good but capped by someone better" from "the best option here".
        pl.when(pl.col("target_share") >= pl.col("_max_share"))
        .then(pl.col("_second_share"))
        .otherwise(pl.col("_max_share"))
        .alias("teammate_top_share"),
        (pl.col("_team_share") - pl.col("target_share")).alias("teammate_share"),
        (pl.col("target_share") >= pl.col("_max_share")).alias("is_team_alpha"),
    ).select(
        "season", "player_id", "teammate_top_share", "teammate_share",
        "team_target_hhi", "is_team_alpha",
    )


def _vacated(seasons: list[int], base: pl.DataFrame, force: bool = False) -> pl.DataFrame:
    """Opportunity leaving a player's team, per season.

    The situation half of the question. A player whose team just lost 25% of its
    targets is in a materially different spot from an identical player whose
    depth chart is unchanged, and neither his usage nor his efficiency says so.

    Uses the roster of the *following* season to decide who is gone, so for the
    current season this reflects the roster you are actually drafting into.

    Returns: season, team, vacated_target_share_next, vacated_carry_share_next.
    """
    rows = []
    for season in seasons:
        roster = nv.rosters(season + 1, force=force)
        if not roster.height:
            continue
        staying = set(roster.get_column("gsis_id").drop_nulls().to_list())

        year = base.filter(pl.col("season") == season)
        if not year.height:
            continue
        gone = year.with_columns(
            (~pl.col("player_id").is_in(list(staying))).alias("gone")
        )
        rows.append(
            gone.group_by(["season", "team"])
            .agg(
                pl.col("target_share").filter(pl.col("gone")).sum().alias("vacated_target_share_next"),
                pl.col("rush_share").filter(pl.col("gone")).sum().alias("vacated_carry_share_next"),
            )
        )

    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


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
        .join(_routes(seasons, force=force), on=["season", "player_id"], how="left")
        .join(_pfr_quality(seasons, force=force), on=["season", "player_id"], how="left")
        .join(cx.play_context(seasons, force=force), on=["season", "player_id"], how="left")
        .join(_bio(force=force), on="player_id", how="left")
    )

    df = df.with_columns(
        # The two metrics routes exist to enable. Yards per route run is the
        # public standard for receiver efficiency; targets per route run is the
        # purest measure of whether the offense looks for him when he is out
        # there, stripped of how often he is out there.
        _safe_div(pl.col("rec_yards"), pl.col("routes")).alias("yprr"),
        _safe_div(pl.col("targets"), pl.col("routes")).alias("tprr"),
        # Per-touch quality, so a back on a bad line is not punished for the
        # line: yards after contact and broken tackles are his, the blocking
        # in front of him is not.
        _safe_div(pl.col("rush_broken_tackles"), pl.col("carries")).alias(
            "rush_broken_tackles_per_att"
        ),
        _safe_div(pl.col("pass_yards"), pl.col("pass_attempts")).alias("ypa"),
        _safe_div(
            pl.col("opp_act_pts") - pl.col("exp_pts"), pl.col("pass_attempts")
        ).alias("pts_over_exp_per_att"),
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

    df = df.filter(pl.col("games") >= min_games)

    # Context that depends on the assembled frame rather than a raw source.
    df = (
        df.join(_competition(seasons, df), on=["season", "player_id"], how="left")
        .join(_vacated(seasons, df, force=force), on=["season", "team"], how="left")
    )

    return df.sort(["season", "position", "pos_rank"])


# --- The three axes ---------------------------------------------------------
#
# The split that makes under/overvaluation detectable at all.
#
# QUALITY is per-opportunity: how good is he *when* he plays. Nothing here scales
# with how much work he gets, which is the point — volume is what ADP already
# knows, so clustering on it rediscovers the market instead of disagreeing with
# it. (It also explains why the old clustering always chose two groups: volume
# dominated the variance and k-means split on "featured vs not" every time.)
#
# OPPORTUNITY is volume: how much work he got. Read as a separate axis, not
# mixed into the distance metric.
#
# SITUATION is what is about to change: who else is competing for his touches
# and how much opportunity his team just lost.

# Every column here cleared `stability.NOISE_FLOOR` at its own position — a
# year-over-year percentile correlation of at least 0.20. That gate removed six
# columns that had been in these sets on reputation, and the casualties are
# worth naming because they are all things the fantasy literature treats as
# skill:
#
#   contested_catch_rate  WR 0.061, TE -0.118   charted by hand, and still a
#                                               coin flip on a small denominator
#   drop_rate             WR 0.096, TE  0.183   drops are rare enough that a
#                                               season is mostly sampling error
#   catch_rate            RB 0.046, TE  0.142   at these positions it measures
#                                               how far the throws travel, and
#                                               `adot` already says that
#   ypt                   RB 0.174              same, one step removed
#
# `catch_rate` and `catch_rate_on_catchable` survive at receiver (0.468, 0.354)
# and only there. The FTN split does what it was added to do at the position
# with enough targets to measure it, and nothing at the position without.
QUALITY_FEATURES: dict[str, list[str]] = {
    # `catch_rate_on_catchable` is here rather than beside `catch_rate` for a
    # reason: catch rate charges a receiver for every ball thrown nowhere near
    # him, so it is partly a measurement of his quarterback. Splitting on whether
    # the ball was catchable moves the passer's contribution out to
    # `catchable_rate` in SITUATION and leaves the receiver's hands here. It
    # begins in 2022 — see `CHARTED_FEATURES`.
    "WR": ["yprr", "tprr", "ypt", "yac_per_rec", "catch_rate", "adot",
           "avg_separation", "avg_yac_above_expectation", "receiving_rat",
           "catch_rate_on_catchable"],
    "TE": ["yprr", "tprr", "ypt", "yac_per_rec", "adot",
           "avg_separation", "avg_yac_above_expectation", "receiving_rat"],
    # `ryoe_per_att` was added expecting it to be the running back's yards per
    # route run — it models the blocking and the box and charges the back only
    # with the difference, which is exactly what yards per carry fails to do. It
    # persists at 0.202, barely above the floor and well under the receiving
    # columns at the same position. Kept, because it is still the best available
    # measure of a back carrying the ball; demoted from the claim that it would
    # carry this position. What actually persists about a running back is
    # whether his offense throws to him.
    "RB": ["ypc", "yards_after_contact_per_att", "rush_broken_tackles_per_att",
           "yprr", "tprr", "yac_per_rec",
           "ryoe_per_att", "rush_efficiency"],
    # Thin by comparison, and deliberately so: the metrics that make this work
    # for skill players (yards per route, separation, yards after contact) have
    # no quarterback analogue in this data. Rushing efficiency is included
    # because it is the axis that actually separates fantasy quarterbacks.
    "QB": ["ypa", "pts_over_exp_per_att", "ypc"],
}

# Volume, plus *where* the volume happened. Red-zone and end-zone shares sit on
# this axis rather than the quality one because they describe the work a player
# is handed, not what he does with it — but they are the half of the work that
# raw target share is blindest to. Two receivers on 25% of targets are not the
# same asset when one of them is the goal-line look and the other is not, and
# `exp_td_share` is the single column that says so: the share of his offense's
# modeled touchdown chances, pass and rush combined, that run through him.
OPPORTUNITY_FEATURES: dict[str, list[str]] = {
    "WR": ["route_share", "target_share", "air_yards_share", "snap_pct", "tgt_per_game",
           "rz_target_share", "ez_target_share", "neutral_target_share", "exp_td_share"],
    "TE": ["route_share", "target_share", "air_yards_share", "snap_pct", "tgt_per_game",
           "rz_target_share", "ez_target_share", "neutral_target_share", "exp_td_share"],
    "RB": ["snap_pct", "rush_share", "target_share", "carry_per_game", "route_share",
           "rz_carry_share", "gz_carry_share", "neutral_rush_share", "exp_td_share"],
    "QB": ["snap_pct", "rush_share", "gz_carry_share", "exp_td_share"],
}

SITUATION_FEATURES = [
    "teammate_top_share",
    "teammate_share",
    "team_target_hhi",
    "vacated_target_share_next",
    "vacated_carry_share_next",
    # What his usage was made of rather than how much of it there was. A high
    # screen rate is a discount, not a virtue: those yards were manufactured by
    # the play-caller and are the first thing a new coordinator takes away. A
    # low `catchable_rate` is the opposite — it means his catch rate is being
    # dragged down by the man throwing to him.
    "screen_target_rate",
    "catchable_rate",
    "contested_rate",
    "stacked_box_rate",
    "time_to_los",
    "exp_td_per_touch",
]

# The context columns that begin in 2022 rather than 2018, because FTN charting
# does. They are legitimate for the current-season board and for clustering,
# where every row is 2025 and fully covered. They are excluded from the backtest,
# where four of eight feature seasons would be null and median imputation would
# hand half the training set an invented value — see `breakout.model_features`.
CHARTED_FEATURES = [
    "catchable_rate",
    "catch_rate_on_catchable",
    "contested_rate",
    "contested_catch_rate",
    "screen_target_rate",
]


def quality_features(position: str | None = None) -> list[str]:
    """Per-opportunity quality columns — the axis ADP is worst at pricing."""
    if position is None:
        seen: list[str] = []
        for cols in QUALITY_FEATURES.values():
            seen += [c for c in cols if c not in seen]
        return seen
    return list(QUALITY_FEATURES.get(position, QUALITY_FEATURES["WR"]))


def opportunity_features(position: str | None = None) -> list[str]:
    """Volume columns — how much work he gets, which the market prices well."""
    if position is None:
        seen: list[str] = []
        for cols in OPPORTUNITY_FEATURES.values():
            seen += [c for c in cols if c not in seen]
        return seen
    return list(OPPORTUNITY_FEATURES.get(position, OPPORTUNITY_FEATURES["WR"]))


def play_context_features(position: str | None = None, charted: bool = False) -> list[str]:
    """Context columns from `context.py`, split by whether they cover the window.

    `charted=False` — the default — returns only what exists back to 2018, which
    is the set a season-forward backtest may use. Passing `charted=True` adds the
    FTN columns, which start in 2022 and belong to the current-season board.
    """
    pool = quality_features(position) + opportunity_features(position) + SITUATION_FEATURES
    cols = [c for c in cx.CONTEXT_COLUMNS if c in pool]
    if charted:
        return cols
    return [c for c in cols if c not in CHARTED_FEATURES]


def feature_columns(position: str | None = None) -> list[str]:
    """The model-facing columns. Position-aware because the sets barely overlap.

    Receiving air-yards features are meaningless for a running back's usage
    profile and rushing share is meaningless for a receiver; including both
    everywhere would mean imputing half of every row.

    Play-context columns are appended from `play_context_features`, which drops
    the FTN-charted ones by default so this stays usable back to 2018.
    """
    base = ROLE_FEATURES + EFFICIENCY_FEATURES + EXPECTED_FEATURES + CONTEXT_FEATURES
    extra = play_context_features(position)
    if position is None:
        return base + NGS_FEATURES + extra
    if position == "QB":
        return [
            c
            for c in base
            if c not in ("target_share", "air_yards_share", "adot", "wopr", "catch_rate", "ypt", "yac_per_rec")
        ] + extra
    if position == "RB":
        return [c for c in base if c not in ("air_yards_share", "adot")] + extra
    return [c for c in base if c not in ("rush_share", "ypc")] + NGS_FEATURES + extra


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
