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
from src import profiles as pf
from src.config import CURRENT_SEASON, SEASON
from src.profiles import LeagueProfile

# The positions this module scores. Named for what it is rather than
# `SKILL_POSITIONS`, which stopped being true when quarterbacks were added.
#
# **Quarterbacks were excluded until 2026-08-10, and the reason was half right.**
# The old note said the metrics that carry this — yards per route run,
# separation, yards after contact — have no quarterback analogue. True of those
# metrics, and false of the position: QB has its own quality set (`ypa`,
# `pts_over_exp_per_att`, `ypc`) which `archetypes.scores` has always computed
# and this module simply threw away. In a superflex league that left the board
# blank at exactly the position where the roster format makes the edge.
#
# It was gated on a measurement before being switched on. Season-forward, the
# stability-weighted QB quality score at season t against PPG at t+1:
#
#     QB   n=163   rho +0.367   95% CI [+0.213, +0.493]   (3 features)
#     RB   n=377   rho +0.330   95% CI [+0.236, +0.419]   (8 features)
#     WR   n=610   rho +0.502   95% CI [+0.439, +0.564]   (10 features)
#
# So QB carries about as much signal as RB, off three columns instead of eight.
# Two caveats that belong next to any QB number this produces, because neither
# is visible in the pooled figure:
#
#   - **It is a thinner measurement.** Three metrics, and the stability weights
#     make it mostly a rushing score (`ypc` repeats at 0.601, `ypa` at 0.360).
#     Defensible for fantasy quarterbacks, but it is a rushing read wearing a
#     general-quality label.
#   - **The recent window does not clear zero on its own.** 2018-2022 gives
#     +0.420 [+0.267, +0.567]; 2023-2025 gives +0.118 [-0.208, +0.416] on 43
#     pairs. Those intervals overlap, so this is not established decay — a
#     single season's interval here is ~0.9 wide — but it is not nothing either.
VALUED_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE", "QB")

# The volume floor under which a per-opportunity metric is a rumour rather than a
# measurement, keyed by position because the denominators are different things.
#
# **A quarterback's `routes` value is dropbacks, not routes run.** Josh Allen
# shows 665. So the old single `routes >= 100` gate did not drop quarterbacks —
# it waved them through on a column that means something else at their position,
# which is the more dangerous of the two failure modes. Taysom Hill is the case
# that makes it concrete: 6 pass attempts, 52 carries, a `ypa` computed off the
# 6, and a quality percentile of 25 built on it.
MIN_VOLUME: dict[str, tuple[str, float]] = {
    "WR": ("routes", 100.0),
    "RB": ("routes", 100.0),
    "TE": ("routes", 100.0),
    # Roughly a half-season of starter volume. Below it `ypa` and
    # `pts_over_exp_per_att` are both small-sample artifacts.
    "QB": ("pass_attempts", 150.0),
}

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
    positions: tuple[str, ...] = VALUED_POSITIONS,
    min_games: int = 8,
    min_routes: int = 100,
    scoring: str | None = None,
    teams: int | None = None,
    df: pl.DataFrame | None = None,
    clusters: pl.DataFrame | None = None,
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """Every drafted skill player, with quality, opportunity, path and price.

    `min_routes` matters more than it looks. Yards per route run on 40 routes is
    not a measurement, it is a rumour — a single long catch moves it half a yard.
    The default drops players whose efficiency could not be estimated, which is
    the right trade even though it hides a few genuine deep-bench fliers. It
    applies to the routes-based positions only; quarterbacks are gated on pass
    attempts instead. See `MIN_VOLUME` for why that distinction is not cosmetic.

    **Quarterbacks carry a `quality_pct` but not a `path_score`.** The path terms
    are vacated target share and the teammate share standing in front of him,
    neither of which describes a quarterback's route to volume — a starting QB
    already has all the volume there is. `path_score` therefore comes back null
    for QB, and null means *not applicable* rather than *no path*. Read
    `quality_pct` and `value_gap` for quarterbacks and ignore the path column.

    **The price side has to come from the profile's market**, and defaulting it
    to half-PPR was wrong here for the same reason it was wrong in
    `expected.adp_curve`: `market_pct` is a percentile of *draft price*, so
    reading it off a market the league does not play in makes the subtraction
    against quality compare two different drafts. In a superflex league the 1QB
    board misprices by a round or more before the comparison even starts.

    Returns: gsis_id, name, position, team, adp, adp_pos_rank, quality_score,
    quality_pct, opportunity_score, opportunity_pct, market_pct, value_gap,
    path_score, verdict, plus the underlying quality and situation columns.
    """
    profile = profile or pf.resolve()
    scoring = scoring or profile.adp_scoring
    teams = teams or profile.adp_teams

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
            "player_id", "routes", "pass_attempts", "yprr", "tprr",
            "target_share", "route_share",
            "snap_pct", "adot", "avg_separation", "ypc",
            "yards_after_contact_per_att", "ryoe_per_att",
            "exp_td_share", "ez_target_share", "gz_carry_share",
            "neutral_target_share", "teammate_top_share", "is_team_alpha",
            "vacated_target_share_next", "vacated_carry_share_next", "age",
            # Carried so the board can be plotted against the metrics that
            # actually persist — `stability.sticky_features` picks the list and
            # every position's top six has to be reachable from here. Without
            # these, quarterbacks lost `rush_share` (the single stickiest metric
            # measured anywhere, 0.82) and running backs lost four of their six.
            # The panel degraded silently to whichever columns happened to be
            # present, which is a chart quietly answering a smaller question.
            "air_yards_share", "tgt_per_game", "carry_per_game",
            "rush_share", "neutral_rush_share", "rz_carry_share",
            "rz_target_share",
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

    # Volume floor, one denominator per position — see `MIN_VOLUME`. Written as
    # an allow-list rather than a series of exclusions so that a position with no
    # floor defined drops out instead of being waved through ungated, which is
    # the direction that fails safe.
    # `min_routes` stays the caller-facing knob it always was, so it overrides the
    # routes-based floors; QB's attempt floor has no such knob and comes from the
    # table.
    floors = {
        position: (col, float(min_routes) if col == "routes" else floor)
        for position, (col, floor) in MIN_VOLUME.items()
    }
    keep_gate = pl.lit(False)
    for position, (volume_col, floor) in floors.items():
        if position not in positions or volume_col not in joined.columns:
            continue
        keep_gate = keep_gate | (
            (pl.col("position") == position)
            & (pl.col(volume_col).fill_null(0) >= floor)
        )
    joined = joined.filter(keep_gate)
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
    #
    # **Known limitation, measured rather than assumed.** For a team alpha the
    # first and third terms cancel: he has no room left to grow *and* nobody
    # standing in front of him, and both are true at once. So `path_score`
    # separates alphas from the field by 0.3 points on a 0-100 scale while
    # `opportunity_pct` separates them by 33. The composite is close to silent
    # on room, and `test_path_score_does_not_claim_to_measure_room` pins that so
    # a reweighting has to be deliberate. Read `opportunity_pct` directly when
    # room to grow is the question.
    path_terms = [
        (100 - pl.col("opportunity_pct")),  # room to grow
    ]
    if "vacated_target_share_next" in scored.columns:
        path_terms.append(_pct("vacated_target_share_next"))
    if "teammate_top_share" in scored.columns:
        # A *smaller* teammate share is the good end — nobody blocking him.
        path_terms.append(_pct("teammate_top_share", bigger_is_better=False))

    # Nulled at QB rather than left to compute. Every term above is either about
    # earning targets or about the teammate taking them, so for a starting
    # quarterback the composite would silently reduce to "room to grow" alone and
    # read as a real path score. A blank column says *not applicable*; a plausible
    # number says something false.
    scored = scored.with_columns(
        pl.when(pl.col("position") == "QB")
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise((sum(path_terms) / len(path_terms)).round(1))
        .alias("path_score")
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

    **Quarterbacks bypass the path filter rather than failing it.** Their
    `path_score` is null by construction (see `board`), and a null fails a `>=`
    comparison silently, so without this every underpriced quarterback would
    vanish from the list for a reason that has nothing to do with his price. The
    path question is not asked of quarterbacks, so it cannot disqualify one.
    """
    if not valued.height:
        return pl.DataFrame()
    return (
        valued.filter(
            (pl.col("verdict") == "undervalued")
            & (
                (pl.col("path_score") >= min_path)
                | (pl.col("position") == "QB")
            )
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

    # Weighted by how much each metric repeats, the same way `quality_score` is.
    # This was a bare `np.linalg.norm` until 2026-08-14 — every column an equal
    # vote — while the score built from this exact matrix weighted them. Two
    # backs could come out "similar" on a fluky efficiency season neither would
    # repeat. See `archetypes._distance`.
    distance = ar._distance(x, used, position, ar.stability_weights(base), idx)

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
