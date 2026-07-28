"""Quality archetypes — grouping players by how good they are per opportunity.

The question this answers: which cheap players have the *per-snap profile* of
expensive ones? A receiver with elite yards per route run and a 12% target share
is a different bet from one with average efficiency and the same share, and no
volume-based view can tell them apart.

**Clustering runs on quality only — volume is deliberately excluded**, and that
is the change that makes this module useful. Target share, snap share and
air-yards share are what ADP already prices; clustering on them rediscovers the
market rather than disagreeing with it. It also swamped the distance metric:
volume carried most of the variance, so k-means split on "featured vs not" every
single time and reported two groups for every position. With volume removed, the
groups describe how a player performs rather than how often he is used.

Opportunity is still computed and returned — as a *separate axis*, not folded
into the distance. `valuation.py` crosses the two, because the interesting
player is the one who is high on quality and low on opportunity and price.

Clusters describe. They do not predict. A receiver landing with three alphas
means his efficiency profile rhymes with theirs; whether he ever gets their
volume depends on his depth chart, which is what the situation columns are for.

Fit per position, because the metrics barely overlap: a back's yards after
contact and a receiver's separation are not the same axis.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import features as ft
from src.config import CURRENT_SEASON, FANTASY_POSITIONS

# Positions with a small pool cannot support many clusters. QB and TE are
# ~45 and ~95 players a season; asking for eight groups from 45 players yields
# groups of five, which is a partition, not a pattern.
K_CEILING: dict[str, int] = {"QB": 5, "RB": 6, "TE": 5, "WR": 7}


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


def choose_k(
    x: np.ndarray,
    k_range: tuple[int, int] = (2, 7),
    seed: int = 0,
    min_cluster: int = 4,
) -> pl.DataFrame:
    """Silhouette and inertia at each k. Returned so the app can show the curve.

    k is picked by max silhouette *among viable solutions*. A solution is viable
    only if its smallest cluster has `min_cluster` members: k-means minimizes
    squared distance, so given an outlier it will happily spend a split
    quarantining one player and collect a high silhouette for doing it. A cluster
    of one is not an archetype, and "Taysom Hill" is not a usage pattern.

    The curve is worth exposing whole. If it is monotone with no interior peak,
    the position genuinely does not cluster, and saying so is more useful than
    drawing whatever k won by 0.003.

    Returns: k, silhouette, inertia, smallest, viable.
    """
    lo, hi = k_range
    hi = min(hi, max(lo, x.shape[0] - 1))
    rows = []
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(x)
        smallest = int(np.bincount(km.labels_, minlength=k).min())
        rows.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(x, km.labels_)), 4),
                "inertia": round(float(km.inertia_), 2),
                "smallest": smallest,
                "viable": smallest >= min_cluster,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def cluster(
    season: int = CURRENT_SEASON,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
    min_games: int = 8,
    k_range: tuple[int, int] = (2, 7),
    seed: int = 0,
    k: int | None = None,
    df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Assign every qualified player a usage cluster within his position.

    `min_games` is 8 rather than the feature table's 4 — a usage *profile* from
    four games is mostly a description of which four games.

    **What this actually found, and it is worth stating plainly:** silhouette
    peaks at k=2 for all four positions and declines monotonically after. NFL
    usage has one dominant axis — how much of his offense a player commands —
    and the honest two-group answer is "featured" and "not". The one genuinely
    interesting split is at quarterback, where the two groups are rushing and
    pocket, not good and bad.

    `k` overrides the silhouette choice. Finer archetypes are legitimate to
    explore, but they are the analyst asserting structure rather than the data
    offering it, so the app shows the silhouette cost when you turn the dial.

    Returns: season, player_id, player_name, position, team, cluster, k,
    silhouette, dist_to_center, ppg, pos_rank, games.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    pool = base.filter((pl.col("season") == season) & (pl.col("games") >= min_games))
    out: list[pl.DataFrame] = []

    for position in positions:
        grp = pool.filter(pl.col("position") == position)
        if grp.height < 12:
            continue

        x, used = _matrix(grp, ft.quality_features(position))
        if not used:
            continue

        scores = choose_k(x, (k_range[0], min(k_range[1], K_CEILING.get(position, 6))), seed)
        if not scores.height:
            continue
        if k is not None:
            chosen = min(k, grp.height - 1)
        else:
            viable = scores.filter(pl.col("viable"))
            if viable.height:
                chosen = int(viable.sort("silhouette", descending=True).get_column("k")[0])
            else:
                # Nothing clears the minimum group size — the position resists
                # clustering at every k. Fall back to the *smallest* k rather
                # than the best silhouette: fewer groups is the safer failure,
                # and taking the highest score here is how a one-player cluster
                # gets shipped as an archetype. Happens at quarterback, whose
                # quality set is thin enough that one outlier dominates.
                chosen = int(scores.sort(["smallest", "silhouette"], descending=[True, True]).get_column("k")[0])

        km = KMeans(n_clusters=chosen, n_init=25, random_state=seed).fit(x)
        centers = km.cluster_centers_[km.labels_]
        dist = np.linalg.norm(x - centers, axis=1)

        # A single interpretable number per axis, so the two can be crossed
        # without re-deriving them everywhere. Both are mean standardized
        # scores, which means they are relative to this position and season —
        # a quality score of +1 is "one standard deviation better per
        # opportunity than other players at this position", not an absolute.
        quality = x.mean(axis=1)
        opp_x, opp_used = _matrix(grp, ft.opportunity_features(position))
        opportunity = opp_x.mean(axis=1) if opp_used else np.zeros(grp.height)

        frame_ = grp.select(
            "season", "player_id", "player_name", "position", "team",
            "games", "ppg", "pos_rank",
        ).with_columns(
            pl.Series("cluster", km.labels_.astype(np.int32)),
            pl.lit(chosen, dtype=pl.Int32).alias("k"),
            pl.Series("silhouette", silhouette_samples(x, km.labels_)).round(4),
            pl.Series("dist_to_center", dist).round(4),
            pl.Series("quality_score", quality).round(4),
            pl.Series("opportunity_score", opportunity).round(4),
        )

        # Rank clusters by mean quality so tier 1 is always the best group,
        # rather than whatever integer k-means happened to assign.
        order = (
            frame_.group_by("cluster")
            .agg(pl.col("quality_score").mean().alias("_m"))
            .sort("_m", descending=True)
            .with_row_index("quality_tier", offset=1)
            .select("cluster", pl.col("quality_tier").cast(pl.Int32))
        )
        out.append(frame_.join(order, on="cluster", how="left"))

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def cluster_profiles(
    clusters: pl.DataFrame,
    df: pl.DataFrame | None = None,
    season: int | None = None,
) -> pl.DataFrame:
    """What each cluster actually is, in feature terms, with a generated label.

    The label comes from the two features furthest from the positional mean in
    standard deviations ("high target_share / low adot"). Generated rather than
    hand-written because hand-written labels drift the moment the data updates
    and nobody notices.

    Returns: position, cluster, n, mean_ppg, label, plus z_<feature> columns.
    """
    if not clusters.height:
        return pl.DataFrame()

    base = df if df is not None else ft.build()
    season = season or int(clusters.get_column("season")[0])
    joined = clusters.join(
        base.filter(pl.col("season") == season).drop(
            [c for c in ("player_name", "position", "team", "games", "ppg", "pos_rank") if c in base.columns]
        ),
        on=["season", "player_id"],
        how="left",
    )

    rows: list[dict[str, object]] = []
    for position in joined.get_column("position").unique().sort().to_list():
        grp = joined.filter(pl.col("position") == position)
        cols = [c for c in ft.quality_features(position) if c in grp.columns]
        stats = {
            c: (grp.get_column(c).mean(), grp.get_column(c).std() or 1.0) for c in cols
        }

        for cid in sorted(grp.get_column("cluster").unique().to_list()):
            sub = grp.filter(pl.col("cluster") == cid)
            row: dict[str, object] = {
                "position": position,
                "cluster": int(cid),
                "n": sub.height,
                "mean_ppg": round(float(sub.get_column("ppg").mean() or 0.0), 2),
            }
            zs: list[tuple[str, float]] = []
            for c in cols:
                mean, sd = stats[c]
                val = sub.get_column(c).mean()
                if val is None or mean is None:
                    continue
                z = (float(val) - float(mean)) / float(sd or 1.0)
                row[f"z_{c}"] = round(z, 2)
                zs.append((c, z))

            zs.sort(key=lambda t: abs(t[1]), reverse=True)
            row["label"] = " / ".join(
                f"{'high' if z > 0 else 'low'} {name}" for name, z in zs[:2]
            ) or "undifferentiated"
            rows.append(row)

    return pl.DataFrame(rows, infer_schema_length=None).sort(["position", "mean_ppg"], descending=[False, True])


def neighbors(
    player_id: str,
    clusters: pl.DataFrame,
    df: pl.DataFrame | None = None,
    n: int = 8,
    season: int | None = None,
) -> pl.DataFrame:
    """Players whose usage profile sits closest to this one, same position.

    Euclidean distance in the standardized feature space — the same space the
    clustering used, so "nearest neighbor" and "same cluster" tell a consistent
    story. This is the output the whole module exists for: the mid-round back
    whose profile sits next to three players going four rounds earlier.

    Returns: player_id, player_name, position, team, cluster, distance, ppg,
    pos_rank, games.
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
        grp.select("player_id", "player_name", "position", "team", "cluster", "games", "ppg", "pos_rank")
        .with_columns(pl.Series("distance", dist).round(3))
        .filter(pl.col("player_id") != player_id)
        .sort("distance")
        .head(n)
    )
