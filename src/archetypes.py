"""Usage archetypes — grouping players by how they are used, not how they scored.

The question this answers: which mid-round players have a *usage profile* that
looks like the profiles of the players going in round two? Not "who will break
out" — that is `breakout.py`, and it is a different and much harder question.
This is pattern description, and it is useful precisely because it is honest
about being description.

Clusters describe. They do not predict. A receiver landing in the same cluster
as three alphas means his target share, air-yards share, and route role rhyme
with theirs — it does not mean he will produce like them, because the thing that
separates him from them may be talent, and talent is not in this feature set.
The app repeats that claim wherever it shows a cluster.

Fit per position, because the features barely overlap: a back's rushing share
and a receiver's air-yards share are not the same axis and putting them in one
distance metric produces four clusters that are really just "is a running back".
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
    df: pl.DataFrame, cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Standardized feature matrix, median-imputed within the position.

    Imputation is within-position on purpose: a tight end's median separation is
    not a receiver's, and imputing across positions would pull every missing TE
    toward receiver norms and invent a difference that isn't there.
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

        x, used = _matrix(grp, ft.cluster_feature_columns(position))
        if not used:
            continue

        scores = choose_k(x, (k_range[0], min(k_range[1], K_CEILING.get(position, 6))), seed)
        if not scores.height:
            continue
        if k is not None:
            chosen = min(k, grp.height - 1)
        else:
            viable = scores.filter(pl.col("viable"))
            # If nothing is viable the position resists clustering entirely; fall
            # back to the best available rather than dropping it silently, and
            # let the silhouette curve in the app explain why it looks thin.
            pick_from = viable if viable.height else scores
            chosen = int(pick_from.sort("silhouette", descending=True).get_column("k")[0])

        km = KMeans(n_clusters=chosen, n_init=25, random_state=seed).fit(x)
        centers = km.cluster_centers_[km.labels_]
        dist = np.linalg.norm(x - centers, axis=1)

        out.append(
            grp.select(
                "season", "player_id", "player_name", "position", "team",
                "games", "ppg", "pos_rank",
            ).with_columns(
                pl.Series("cluster", km.labels_.astype(np.int32)),
                pl.lit(chosen, dtype=pl.Int32).alias("k"),
                pl.Series("silhouette", silhouette_samples(x, km.labels_)).round(4),
                pl.Series("dist_to_center", dist).round(4),
            )
        )

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
        cols = [c for c in ft.cluster_feature_columns(position) if c in grp.columns]
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
        ["player_id", *[c for c in ft.cluster_feature_columns(position) if c in base.columns]]
    )
    grp = pool.join(feats, on="player_id", how="left")

    x, used = _matrix(grp, ft.cluster_feature_columns(position))
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
