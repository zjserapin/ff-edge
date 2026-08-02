"""Per-position quality and opportunity scores, and nearest-neighbour comparables.

**The k-means clustering that used to live here has been removed.** It was
measured and it did not earn its place:

  - Silhouette peaked at 0.19 to 0.29 depending on position. That is a partition
    of a continuum, not a set of groups. Presenting it as archetypes implied
    structure the data does not contain.
  - Cluster membership added nothing downstream. Fed to the projection model as
    Gaussian-mixture soft memberships — the generous version, since it carries
    the uncertainty rather than a hard label — it scored 0.022 *below* ADP alone.
  - PCA-whitening the block first, the standard fix for the correlated features
    the quality set has, made the resulting score worse at ranking next season
    (0.464 to 0.340 at receiver). Un-whitened PCA is an orthogonal rotation and
    cannot change a Euclidean distance at all, so there was no version of that
    idea left to try.

What survived the same testing, and is what remains here:

**The quality/opportunity split.** Two axes, computed separately and never mixed
into one distance. Volume is what ADP already prices; per-opportunity quality is
the axis it prices worst. Opportunity persists year to year at 0.47-0.55 and
quality at 0.28-0.44 (see `stability`), so they are genuinely different
measurements rather than two views of one thing.

**Stability weighting.** Each standardized metric is weighted by how well it
repeats rather than counted equally, which moved the quality score's correlation
with next season's points from 0.464 to 0.502 at receiver.

**Comparables.** `neighbors` — nearest players in the same standardized space —
is what the clustering was a lossy summary of. "Who does this player look like
and what do they cost" is answerable directly; going through a hard group label
first only threw information away.

Everything is per position, because the metrics barely overlap: a back's yards
after contact and a receiver's separation are not the same axis.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src import features as ft
from src import stability as st
from src.config import CURRENT_SEASON, FANTASY_POSITIONS


def _matrix(
    df: pl.DataFrame, cols: list[str], winsorize: float = 0.02
) -> tuple[np.ndarray, list[str]]:
    """Standardized feature matrix, median-imputed and winsorized within position.

    Imputation is within-position on purpose: a tight end's median separation is
    not a receiver's, and imputing across positions would pull every missing TE
    toward receiver norms and invent a difference that isn't there.

    Winsorizing matters more than it sounds. Every quality metric here is a rate,
    and a rate on a small denominator is unbounded — a quarterback with twenty
    attempts can post a yards-per-attempt figure no starter approaches. k-means
    minimizes squared distance, so one such value drags a cluster centre onto
    itself and returns a group of one with a flattering silhouette. That is
    exactly what happened at quarterback: every k from 2 to 5 isolated a single
    player, and k=2 scored 0.71 for doing it. Clipping each column to its 2nd
    and 98th percentile keeps the outlier in the data as an extreme value
    without letting it define an archetype.
    """
    usable = [c for c in cols if c in df.columns and df.get_column(c).is_not_null().any()]
    if not usable:
        return np.empty((df.height, 0)), []

    filled = df.select(
        [pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).median()) for c in usable]
    )
    # A column that is entirely null has no median; drop rather than propagate.
    keep = [c for c in usable if filled.get_column(c).is_not_null().all()]
    if not keep:
        return np.empty((df.height, 0)), []

    x = filled.select(keep).to_numpy()
    n = x.shape[0]
    if winsorize > 0 and n >= 20:
        # Clip at least one and a half observations from each tail regardless of
        # pool size. A flat 2% does nothing to a 32-player position — it trims
        # 0.64 of a player — which is how Taysom Hill's 9.5 yards per attempt on
        # six throws survived to isolate itself as a quarterback archetype at
        # every k from 2 to 5.
        frac = max(winsorize, 1.5 / n)
        lo = np.quantile(x, frac, axis=0)
        hi = np.quantile(x, 1 - frac, axis=0)
        x = np.clip(x, lo, hi)
    return StandardScaler().fit_transform(x), keep


def stability_weights(df: pl.DataFrame | None = None) -> dict[tuple[str, str], float]:
    """Year-over-year persistence per (position, metric), for weighting the scores.

    Computed rather than hardcoded so a wider window re-derives it instead of
    inheriting a constant nobody remembers setting.
    """
    table = st.year_over_year(df)
    if not table.height:
        return {}
    return {
        (row["position"], row["metric"]): max(float(row["r_yoy"]), 0.0)
        for row in table.iter_rows(named=True)
    }


def _weighted(
    x: np.ndarray,
    used: list[str],
    position: str,
    weights: dict[tuple[str, str], float] | None,
) -> np.ndarray:
    """Mean of standardized features, weighted by how much each one persists.

    An unweighted mean asserts that every column deserves an equal vote, and the
    stability table says plainly that they do not: at receiver, `adot` repeats at
    0.638 and `receiving_rat` at 0.279, so averaging them flat gives half the
    weight to a column that is half signal. Weighting by year-over-year
    correlation moved the score's rank correlation with *next* season's points
    from 0.464 to 0.502 at WR and 0.455 to 0.492 at TE, with QB up and RB flat.

    Uses no outcome data — the weights come from whether a metric repeats, not
    from whether it predicts fantasy points — so this is not the search-until-
    something-scores trap. Falls back to the flat mean if no weight is known.
    """
    if not weights:
        return x.mean(axis=1)
    w = np.array([weights.get((position, c), 0.0) for c in used])
    if w.sum() <= 0:
        return x.mean(axis=1)
    return (x * w).sum(axis=1) / w.sum()


def scores(
    season: int = CURRENT_SEASON,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
    min_games: int = 8,
    df: pl.DataFrame | None = None,
    weights: dict[tuple[str, str], float] | None = None,
) -> pl.DataFrame:
    """Every qualified player's quality and opportunity score, within his position.

    Replaces the old `cluster()`. Same two scores, same construction, without the
    k-means step that produced a label nothing downstream could use.

    Both are weighted means of standardized features, so they are relative to
    this position and season: a quality score of +1 means one standard deviation
    better per opportunity than other players at his position that year, not an
    absolute. `quality_pct` and `opportunity_pct` are the same thing as 0-100
    percentiles, which is what `valuation.py` crosses against price.

    `min_games` is 8 rather than the feature table's 4 — a per-opportunity
    profile from four games is mostly a description of which four games.

    Returns: season, player_id, player_name, position, team, games, ppg,
    pos_rank, quality_score, opportunity_score, quality_pct, opportunity_pct.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    # Derived from the full history, not the season being scored: how much a
    # metric repeats is a property of the metric, and estimating it from one
    # season of one position would be noisier than the thing it is correcting.
    if weights is None:
        weights = stability_weights(base)

    pool = base.filter((pl.col("season") == season) & (pl.col("games") >= min_games))
    out: list[pl.DataFrame] = []

    for position in positions:
        grp = pool.filter(pl.col("position") == position)
        if grp.height < 12:
            continue

        qx, qused = _matrix(grp, ft.quality_features(position))
        if not qused:
            continue
        ox, oused = _matrix(grp, ft.opportunity_features(position))

        out.append(
            grp.select(
                "season", "player_id", "player_name", "position", "team",
                "games", "ppg", "pos_rank",
            ).with_columns(
                pl.Series("quality_score", _weighted(qx, qused, position, weights)).round(4),
                pl.Series(
                    "opportunity_score",
                    _weighted(ox, oused, position, weights) if oused else np.zeros(grp.height),
                ).round(4),
            )
        )

    if not out:
        return pl.DataFrame()

    return pl.concat(out, how="diagonal_relaxed").with_columns(
        (
            pl.col("quality_score").rank("average").over("position")
            / pl.len().over("position") * 100
        ).round(1).alias("quality_pct"),
        (
            pl.col("opportunity_score").rank("average").over("position")
            / pl.len().over("position") * 100
        ).round(1).alias("opportunity_pct"),
    )


def neighbors(
    player_id: str,
    clusters: pl.DataFrame,
    df: pl.DataFrame | None = None,
    n: int = 8,
    season: int | None = None,
) -> pl.DataFrame:
    """Players whose per-opportunity profile sits closest to this one, same position.

    Euclidean distance in the standardized quality space. This is the output the
    module exists for now that the clustering is gone: "who does this player look
    like, and what do they cost" answered directly, rather than through a hard
    group label that discarded the distances on the way.

    `clusters` is any frame carrying player_id, player_name, position, team and
    the outcome columns — `scores()` is the intended source, and the parameter
    keeps its name so existing callers do not break.

    Returns: player_id, player_name, position, team, distance, ppg, pos_rank,
    games.
    """
    if not clusters.height:
        return pl.DataFrame()

    base = df if df is not None else ft.build()
    season = season or int(clusters.get_column("season")[0])

    target = clusters.filter(pl.col("player_id") == player_id)
    if not target.height:
        return pl.DataFrame()
    position = str(target.get_column("position")[0])

    pool = clusters.filter(pl.col("position") == position)
    feats = base.filter(pl.col("season") == season).select(
        ["player_id", *[c for c in ft.quality_features(position) if c in base.columns]]
    )
    grp = pool.join(feats, on="player_id", how="left")

    x, used = _matrix(grp, ft.quality_features(position))
    if not used:
        return pl.DataFrame()

    idx = grp.get_column("player_id").to_list().index(player_id)
    dist = np.linalg.norm(x - x[idx], axis=1)

    return (
        grp.select("player_id", "player_name", "position", "team", "games", "ppg", "pos_rank")
        .with_columns(pl.Series("distance", dist).round(3))
        .filter(pl.col("player_id") != player_id)
        .sort("distance")
        .head(n)
    )
