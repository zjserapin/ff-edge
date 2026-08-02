"""Who the market has wrong — quality against price, with a path to volume.

The whole project points here. Everything else measures something; this compares
two of those measurements and names the disagreement.

The argument in three steps:

1. **Quality and volume are different things, and ADP conflates them.** Draft
   price is built almost entirely on last season's production, which is volume
   times efficiency. A receiver who was efficient on few targets and one who was
   inefficient on many can post the same points and carry the same price.

2. **Volume is the market's strength; quality is its blind spot.** Everyone can
   see a 28% target share. Far fewer are pricing 2.4 yards per route run on 300
   routes. So the exploitable disagreement is on the quality axis.

3. **Quality alone is not a buy signal.** A good player permanently stuck behind
   a better one stays stuck. Quality has to be paired with a *path*: opportunity
   he does not yet have, competition that just left, or volume his team is about
   to hand out. That is what `path_score` is for.

The output is deliberately not a projection or a ranking. It is a disagreement
score — where this project's read of a player differs from the draft market's —
and disagreement is a reason to look closer, not a reason to be right.

**Percentiles, not raw scores, and always within position.** A quality score is
a mean of standardized metrics and its scale means nothing on its own; a
percentile against the other players at his position is directly comparable to
a percentile of draft price, which is what makes the subtraction legitimate.
"""

from __future__ import annotations

from typing import Mapping

import polars as pl

from src import archetypes as ar
from src import breakout as bo
from src import features as ft
from src.config import CURRENT_SEASON, LEAGUE_ADP_SCORING, LEAGUE_ADP_TEAMS, SEASON

# The positions this module is for. Quarterbacks are excluded by default: one
# starts, the position is shallow, and the quality metrics that make this work
# (yards per route run, separation, yards after contact) have no quarterback
# analogue worth the name.
SKILL_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")

# How far a player's quality percentile must sit above or below his price
# percentile before it is worth calling a disagreement. Twenty points is roughly
# two tiers at a position and is comfortably outside the noise on a single
# season of quality metrics.
GAP_THRESHOLD = 20.0


def _pct(col: str, bigger_is_better: bool = True) -> pl.Expr:
    """Percentile within position, 0-100, where 100 is always the good end.

    Named for the semantics rather than for polars' `descending` flag, which
    reads backwards here: `rank(descending=True)` assigns rank 1 to the largest
    value, so passing it for a "high is good" column silently inverts the scale.
    That bug put Christian McCaffrey and Justin Jefferson at the top of the
    undervalued list — the most expensive players in the league, scored as
    though they were free.

    For a column where *small* is good (a draft rank, a drop rate), pass
    `bigger_is_better=False` and 100 still means the desirable end.
    """
    return (
        pl.col(col).rank("average", descending=not bigger_is_better).over("position")
        / pl.len().over("position")
        * 100
    )


def board(
    feature_season: int = CURRENT_SEASON,
    adp_season: int = SEASON,
    positions: tuple[str, ...] = SKILL_POSITIONS,
    min_games: int = 8,
    min_routes: int = 100,
    scoring: str = LEAGUE_ADP_SCORING,
    teams: int = LEAGUE_ADP_TEAMS,
    df: pl.DataFrame | None = None,
    clusters: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every drafted skill player, with quality, opportunity, path and price.

    `min_routes` matters more than it looks. Yards per route run on 40 routes is
    not a measurement, it is a rumour — a single long catch moves it half a yard.
    The default drops players whose efficiency could not be estimated, which is
    the right trade even though it hides a few genuine deep-bench fliers.

    Returns: gsis_id, name, position, team, adp, adp_pos_rank, quality_score,
    quality_pct, opportunity_score, opportunity_pct, market_pct, value_gap,
    path_score, verdict, plus the underlying quality and situation columns.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    fits = (
        clusters
        if clusters is not None
        else ar.scores(feature_season, positions=positions, min_games=min_games, df=base)
    )
    if not fits.height:
        return pl.DataFrame()

    market = bo.adp_board(adp_season, scoring, teams)
    if not market.height:
        return pl.DataFrame()

    season_features = base.filter(pl.col("season") == feature_season)
    # Carried for reading the row, not for scoring it — the verdict comes from
    # the percentiles below. `drop_rate` used to sit here and was removed for the
    # same reason it left the quality set: it repeats year to year at 0.10, so
    # putting it in front of a drafter invites a decision on noise. The
    # touchdown-equity and goal-line columns replace it because they persist
    # (0.36-0.56) and because they explain the gap the board is pointing at.
    keep = [
        c
        for c in (
            "player_id", "routes", "yprr", "tprr", "target_share", "route_share",
            "snap_pct", "adot", "avg_separation", "ypc",
            "yards_after_contact_per_att", "ryoe_per_att",
            "exp_td_share", "ez_target_share", "gz_carry_share",
            "neutral_target_share", "teammate_top_share", "is_team_alpha",
            "vacated_target_share_next", "vacated_carry_share_next", "age",
        )
        if c in season_features.columns
    ]

    joined = (
        market.filter(pl.col("position").is_in(list(positions)))
        .join(
            fits.select(
                pl.col("player_id").alias("gsis_id"),
                "quality_score", "opportunity_score",
                pl.col("ppg").alias("prior_ppg"),
                pl.col("pos_rank").alias("prior_pos_rank"),
            ),
            on="gsis_id",
            how="inner",
        )
        .join(
            season_features.select([pl.col("player_id").alias("gsis_id"), *keep[1:]]),
            on="gsis_id",
            how="left",
        )
    )
    if not joined.height:
        return pl.DataFrame()

    if "routes" in joined.columns:
        joined = joined.filter(pl.col("routes").fill_null(0) >= min_routes)
    if not joined.height:
        return pl.DataFrame()

    scored = joined.with_columns(
        _pct("quality_score").round(1).alias("quality_pct"),
        _pct("opportunity_score").round(1).alias("opportunity_pct"),
        # Draft price on the same 0-100 scale: 100 is the most expensive player
        # at the position, so a small ADP positional rank scores high.
        _pct("adp_pos_rank", bigger_is_better=False).round(1).alias("market_pct"),
    ).with_columns(
        (pl.col("quality_pct") - pl.col("market_pct")).round(1).alias("value_gap")
    )

    # Path to volume: room he has left to grow, plus opportunity his team is
    # about to hand out, minus the teammate standing in front of him. Each term
    # is a percentile where 100 is the favourable end, so they average cleanly.
    path_terms = [
        (100 - pl.col("opportunity_pct")),  # room to grow
    ]
    if "vacated_target_share_next" in scored.columns:
        path_terms.append(_pct("vacated_target_share_next"))
    if "teammate_top_share" in scored.columns:
        # A *smaller* teammate share is the good end — nobody blocking him.
        path_terms.append(_pct("teammate_top_share", bigger_is_better=False))

    scored = scored.with_columns(
        (sum(path_terms) / len(path_terms)).round(1).alias("path_score")
    )

    return scored.with_columns(
        pl.when(pl.col("value_gap") >= GAP_THRESHOLD)
        .then(pl.lit("undervalued"))
        .when(pl.col("value_gap") <= -GAP_THRESHOLD)
        .then(pl.lit("overvalued"))
        .otherwise(pl.lit("fairly priced"))
        .alias("verdict")
    ).sort("value_gap", descending=True)


def undervalued(
    valued: pl.DataFrame, min_path: float = 50.0, limit: int = 25
) -> pl.DataFrame:
    """Players whose quality outruns their price *and* who have somewhere to go.

    The `min_path` filter is what separates this from a list of good players on
    bad depth charts. A receiver in the 90th percentile of quality who is already
    the alpha on his team has no room left — his price is high because he earned
    it, and the quality gap is telling you he is good, not that he is cheap.
    """
    if not valued.height:
        return pl.DataFrame()
    return (
        valued.filter(
            (pl.col("verdict") == "undervalued") & (pl.col("path_score") >= min_path)
        )
        .sort("value_gap", descending=True)
        .head(limit)
    )


def overvalued(valued: pl.DataFrame, limit: int = 25) -> pl.DataFrame:
    """Players priced well above their per-opportunity quality.

    Read this more cautiously than the undervalued list. Expensive players are
    expensive partly because they command volume, and volume is genuinely
    valuable — a mediocre efficiency profile on 30% of the targets still scores
    points. This flags where the price is paying for last year's production
    rather than for the player.
    """
    if not valued.height:
        return pl.DataFrame()
    return valued.filter(pl.col("verdict") == "overvalued").sort("value_gap").head(limit)


def comparables(
    valued: pl.DataFrame,
    gsis_id: str,
    n: int = 8,
    df: pl.DataFrame | None = None,
    season: int = CURRENT_SEASON,
) -> pl.DataFrame:
    """Players with the most similar quality profile, and what they cost.

    This is the "does his profile look like an elite receiver's" question stated
    so it can be answered: same position, nearest in standardized quality space,
    shown next to their draft price. If the neighbours are all going four rounds
    earlier, that is the finding.

    Returns: name, position, team, distance, quality_pct, opportunity_pct,
    market_pct, adp, prior_ppg.
    """
    if not valued.height:
        return pl.DataFrame()

    target = valued.filter(pl.col("gsis_id") == gsis_id)
    if not target.height:
        return pl.DataFrame()
    position = str(target.get_column("position")[0])

    base = df if df is not None else ft.build()
    pool = valued.filter(pl.col("position") == position)

    feats = base.filter(pl.col("season") == season).select(
        ["player_id", *[c for c in ft.quality_features(position) if c in base.columns]]
    )
    grp = pool.join(feats, left_on="gsis_id", right_on="player_id", how="left")

    x, used = ar._matrix(grp, ft.quality_features(position))
    if not used:
        return pl.DataFrame()

    ids_list = grp.get_column("gsis_id").to_list()
    if gsis_id not in ids_list:
        return pl.DataFrame()
    idx = ids_list.index(gsis_id)

    import numpy as np

    distance = np.linalg.norm(x - x[idx], axis=1)

    return (
        grp.select(
            "gsis_id", "name", "position", "team", "adp", "adp_pos_rank",
            "quality_pct", "opportunity_pct", "market_pct", "prior_ppg", "verdict",
        )
        .with_columns(pl.Series("distance", distance).round(3))
        .filter(pl.col("gsis_id") != gsis_id)
        .sort("distance")
        .head(n)
    )


def summary(valued: pl.DataFrame) -> pl.DataFrame:
    """Counts by position and verdict, so the board's shape is visible at a glance."""
    if not valued.height:
        return pl.DataFrame()
    return (
        valued.group_by(["position", "verdict"])
        .agg(pl.len().alias("n"), pl.col("value_gap").mean().round(1).alias("mean_gap"))
        .sort(["position", "verdict"])
    )
