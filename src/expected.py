"""Expected fantasy points, and the tiers that follow from how uncertain they are.

The project already measured that prior-season usage fed to a *fitted* model
does not beat ADP (-0.000, CI [-0.010, +0.011]), and that the binding
constraint is label seasons rather than features — at 2.5-6.5 events per
variable another column costs more variance than it buys. Nothing here
contradicts that, because nothing here fits a coefficient.

This module multiplies measured quantities instead of estimating them:

    expected points = team environment x player's share of it

`team_environment` is priced by the betting market, which is a *different
market* from fantasy ADP rather than another feature derived from last
season — it is forward-looking and already contains the offseason news that
prior-season usage cannot. `player share` is the opportunity block, which the
stability module found to be the most persistent thing in the project (r ~ 0.5,
against ~0.28 for running back quality). Neither term consumes a degree of
freedom, so the events-per-variable ceiling that closed the modelling door does
not apply.

**What the data actually supports, measured here rather than assumed:**

*The ADP curve is real but not smooth.* Rank-to-points correlations run -0.44
at QB to -0.59 at RB, so price carries genuine information. But the spread
around it swamps the gaps between neighbours: the first five running backs
average 168.3 points with a standard deviation of 74.9, and the next six
average *169.4*. Those two groups are indistinguishable. Receivers are
non-monotone in places — ranks 24-29 average 103.9 and ranks 30-35 average
115.4.

That is the argument for tiering, and it is an empirical one rather than a
presentational convenience: a rank-ordered board implies a precision the data
does not contain, and agonising over RB4 against RB9 is optimising inside the
noise. `tiers()` groups players until the smoothed curve has dropped by more
than a stated amount, so a tier means "these are the same player as far as
anyone can tell."

**Coverage, honestly.** nflverse ships game lines free and complete back to
2022 (`spread_line`, `total_line`), but sportsbooks only post them a few weeks
out — as of August 2026 exactly 51 of 272 games have lines, covering weeks 1-4.
So the Vegas layer is an in-season instrument that switches on progressively,
not a preseason one. Preseason, `expected_points` runs on the ADP curve alone
and says so in its `basis` column. Season win totals would fill that gap and
are not in nflverse; 32 hand-entered numbers once a year is the cheap fix if
it is ever worth it.

Season-long player props ("10+ TDs", "4000 passing yards") are deliberately not
used. They are not free at any useful coverage, they exist only for the stars
who are already the easiest players to rank, and a single threshold gives one
point on a distribution rather than an expectation.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src import breakout as bo
from src import nflverse as nv
from src import scoring as sc
from src.config import FANTASY_POSITIONS, LABEL_SEASONS, SEASON

# Half a point per game over a 14-week fantasy season. The unit a tier break
# has to clear: below this, two players are the same asset in practice.
TIER_GAP_POINTS = 7.0

# Ranks either side of a player pooled when smoothing the ADP curve. Wide
# enough to survive one bust landing on a slot, narrow enough to keep the
# shape — the curve is estimated from ~6 seasons, so a single blown pick at
# WR14 should not carve a notch into the board.
SMOOTH_WINDOW = 3


def _labeled(seasons: list[int] | None = None) -> pl.DataFrame:
    """Each season's draft board joined to what those players actually scored.

    A drafted player who never posted a season scores zero rather than being
    dropped — missing the year is an outcome, and excluding it would inflate
    every expectation on the board by conditioning on health.
    """
    seasons = seasons or LABEL_SEASONS
    parts = []
    for season in seasons:
        board = bo.adp_board(season)
        if not board.height:
            continue
        pts = sc.score_season([season]).select(
            pl.col("player_id").alias("gsis_id"), "fantasy_points", "games"
        )
        parts.append(
            board.join(pts, on="gsis_id", how="left").with_columns(
                pl.col("fantasy_points").fill_null(0.0),
                pl.col("games").fill_null(0),
            )
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def adp_curve(
    seasons: list[int] | None = None,
    window: int = SMOOTH_WINDOW,
    labeled: pl.DataFrame | None = None,
    monotone: bool = True,
) -> pl.DataFrame:
    """Expected points by positional ADP rank, pooled across seasons.

    Two estimates per rank, and the difference between them is the point.

    `raw_points` is a centred window mean over neighbouring ranks, because a
    single rank is six observations wide. It is *not* monotone, and taking it
    at face value produces nonsense: it makes the 7th running back off the
    board worth more than the 1st. With ~35 observations behind each point and
    a standard deviation near 70, the standard error is 9-16 points, so an
    11-point inversion is noise wearing a decimal place.

    `exp_points` is that curve pushed through isotonic regression, which finds
    the closest non-increasing fit. The prior it encodes is the one thing about
    ADP nobody disputes — a player drafted earlier is, in expectation, worth
    more — and where the data contradicts it the algorithm pools the offending
    ranks into one flat value rather than inventing an ordering. Those pools
    are informative in their own right: a flat stretch is the data saying it
    cannot tell those players apart, which is precisely a tier.

    `sd` and `se` come from the unsmoothed window so the uncertainty reported
    is the real one, not the residual left after fitting. Mean and median are
    both carried because they diverge where the bust tail is fat (quarterbacks
    drafted 6th-11th average 209.5 against a median of 215.2), and that gap is
    itself a risk signal rather than a rounding artifact.

    Returns: position, adp_pos_rank, n, exp_points, raw_points, med_points,
    sd, se.
    """
    labeled = labeled if labeled is not None else _labeled(seasons)
    if not labeled.height:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for position in FANTASY_POSITIONS:
        sub = labeled.filter(pl.col("position") == position)
        if not sub.height:
            continue
        ranks = sub.get_column("adp_pos_rank").to_numpy()
        pts = sub.get_column("fantasy_points").to_numpy()

        rows: list[dict[str, object]] = []
        for rank in range(1, int(ranks.max()) + 1):
            hit = np.abs(ranks - rank) <= window
            n = int(hit.sum())
            if n < 5:
                continue
            vals = pts[hit]
            sd = float(vals.std(ddof=1)) if n > 1 else 0.0
            rows.append(
                {
                    "position": position,
                    "adp_pos_rank": rank,
                    "n": n,
                    "raw_points": round(float(vals.mean()), 1),
                    "med_points": round(float(np.median(vals)), 1),
                    "sd": round(sd, 1),
                    "se": round(sd / np.sqrt(n), 1) if n else 0.0,
                }
            )
        if not rows:
            continue

        frame = pl.DataFrame(rows)
        if monotone and frame.height > 1:
            from sklearn.isotonic import IsotonicRegression

            fit = IsotonicRegression(increasing=False).fit_transform(
                frame.get_column("adp_pos_rank").to_numpy(),
                frame.get_column("raw_points").to_numpy(),
            )
            frame = frame.with_columns(
                pl.Series("exp_points", np.round(fit, 1))
            )
        else:
            frame = frame.with_columns(pl.col("raw_points").alias("exp_points"))
        frames.append(frame)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").select(
        "position", "adp_pos_rank", "n", "exp_points", "raw_points",
        "med_points", "sd", "se",
    ).sort(["position", "adp_pos_rank"])


def expected_points(
    season: int = SEASON,
    curve: pl.DataFrame | None = None,
    board: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """This season's draft board with an expected-points estimate attached.

    `basis` records what the number came from, so a downstream consumer never
    has to guess how much to trust it. Today that is always `adp_curve`;
    once game lines populate, a Vegas-adjusted basis joins it and the column
    is how the two are told apart.

    Returns: gsis_id, name, position, team, adp, adp_pos_rank, exp_points,
    med_points, se, basis.
    """
    curve = curve if curve is not None else adp_curve()
    board = board if board is not None else bo.adp_board(season)
    if not board.height or not curve.height:
        return pl.DataFrame()

    return (
        board.join(curve, on=["position", "adp_pos_rank"], how="left")
        .with_columns(pl.lit("adp_curve").alias("basis"))
        .select(
            "gsis_id", "name", "position", "team", "adp", "adp_pos_rank",
            "exp_points", "med_points", "se", "basis",
        )
        .sort("adp")
    )


def tiers(
    scored: pl.DataFrame,
    value_col: str = "exp_points",
    gap: float = TIER_GAP_POINTS,
) -> pl.DataFrame:
    """Group players until the expected-points curve has dropped by `gap`.

    Walks each position from the top and opens a new tier when the running
    leader is more than `gap` points better than the current player. Comparing
    against the *tier leader* rather than the previous player is what stops a
    long shallow slope from being sliced into singleton tiers — the question a
    tier answers is "is this the same asset as the best one in the group",
    not "is this the same as the man one slot up".

    The default gap is half a point per game across a 14-week season. Every
    band this produces is far narrower than the spread around the curve it is
    cut from, so a tier is a floor on indifference, not a claim of precision.

    Adds: tier, tier_rank.
    """
    if not scored.height:
        return scored

    out: list[pl.DataFrame] = []
    for position in scored.get_column("position").unique().sort().to_list():
        sub = (
            scored.filter(pl.col("position") == position)
            .filter(pl.col(value_col).is_not_null())
            .sort(value_col, descending=True)
        )
        if not sub.height:
            continue

        assigned: list[int] = []
        tier = 1
        leader = float(sub.get_column(value_col)[0])
        for value in sub.get_column(value_col).to_list():
            if leader - float(value) > gap:
                tier += 1
                leader = float(value)
            assigned.append(tier)

        out.append(
            sub.with_columns(
                pl.Series("tier", assigned, dtype=pl.Int32),
            ).with_columns(
                pl.col(value_col).rank("ordinal", descending=True).over("tier")
                .cast(pl.Int32).alias("tier_rank")
            )
        )

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def team_environment(
    seasons: list[int] | None = None, force: bool = False
) -> pl.DataFrame:
    """Vegas implied team totals per team-week, from free nflverse game lines.

    A game total and a spread pin both teams' expected scoring exactly:
    the home side is priced at `total/2 + spread_line/2` and the away side at
    `total/2 - spread_line/2`. Verified against outcomes — implied totals
    average 22.91 against an actual 23.64, and correlate 0.39 with the score of
    a single game, which is about as tight as one game can be.

    **nflverse's `spread_line` runs opposite to the betting convention**, and
    it is worth stating plainly because the sign is silent when wrong: there a
    *positive* number means the home team is favoured, whereas a sportsbook
    quotes a favourite at a *negative* number. The `spread` column returned
    here is flipped into the familiar form — negative means this team is
    favoured, and a favourite always carries the higher implied total, which
    `test_favourite_is_priced_above_the_underdog` pins down. Read `spread`
    from this function, never `spread_line` from the raw schedule, unless you
    have re-derived which way it points.

    Aggregated to a team-season the same number correlates **0.87** with that
    team's skill-position fantasy production, which is what makes this the
    right backbone for the environment term. Read that 0.87 as concurrent
    rather than predictive: a season's mean line absorbs in-season movement,
    so it reflects an offence Vegas tracked correctly as much as one it called
    in August. For the weekly use it is put to — who to start, who to trade
    for — concurrent is exactly the question being asked.

    Returns: season, week, team, opponent, is_home, implied_total,
    game_total, spread.
    """
    seasons = seasons or [SEASON]
    games = nv.schedules(seasons, force=force).filter(
        pl.col("total_line").is_not_null() & pl.col("spread_line").is_not_null()
    )
    if not games.height:
        return pl.DataFrame()

    home = games.select(
        "season", "week",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("implied_total"),
        pl.col("total_line").alias("game_total"),
        (-pl.col("spread_line")).alias("spread"),
    )
    away = games.select(
        "season", "week",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("implied_total"),
        pl.col("total_line").alias("game_total"),
        pl.col("spread_line").alias("spread"),
    )
    return pl.concat([home, away]).sort(["season", "week", "team"])


def line_coverage(season: int = SEASON) -> pl.DataFrame:
    """How much of a season the sportsbooks have actually priced yet.

    The Vegas layer is only usable where lines exist, and preseason they mostly
    do not — this is the check that keeps that fact visible instead of letting
    a half-priced slate quietly produce half-informed rankings.

    Returns: week, games, lined, pct_lined.
    """
    games = nv.schedules([season])
    if not games.height:
        return pl.DataFrame()
    return (
        games.group_by("week")
        .agg(
            pl.len().alias("games"),
            pl.col("total_line").is_not_null().sum().alias("lined"),
        )
        .with_columns(
            (pl.col("lined") / pl.col("games") * 100).round(0).alias("pct_lined")
        )
        .sort("week")
    )
