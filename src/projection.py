"""Same question as `breakout.py`, asked in a way this sample can answer.

`breakout` predicts a yes/no: did a player finish inside 60% of his draft price.
That is the decision a drafter actually makes, and it is also an expensive way to
spend 540 rows. Thresholding a continuous outcome throws away everything except
which side of the line a player landed on — RB4 and RB40 are the same "no" if you
paid for RB3, and the model is told nothing about the difference.

This module keeps the outcome continuous. The target is a player's *percentile*
of positional finish within his season, so first at the position is 1.0 and the
worst drafted player is near 0.0, and a season missed entirely is the bottom
rather than a dropped row. Ridge, because with four features and thirty-odd
training rows per position-fold the choice is between a shrunk linear fit and
noise wearing a decision boundary.

Scoring is Spearman rather than AUC, for the same reason the target changed: rank
correlation uses every pair, not just the concordant ones spanning a threshold.

**The distinction that makes the numbers readable.** Predicting next season's
finish is not hard — usage persists, so `opportunity` alone ranks it at roughly
0.70 (see `stability`). Beating *ADP* is hard, because the draft market has
already read the same box scores. Both are reported below, and they are different
questions:

    predicting the finish        skill against nothing
    beating ADP at the finish    skill against the market

Only the second one is worth money, and it is the one that stays small.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import breakout as bk
from src import features as ft
from src.config import FANTASY_POSITIONS

# Ridge penalty. Set from the sample size (four to six features on ~35 rows per
# position-fold), not by sweeping values against the test seasons.
ALPHA = 10.0


def target(df: pl.DataFrame) -> pl.DataFrame:
    """Add `finish_pct`: positional finish as a percentile within season.

    Reversed so that bigger is better, matching every other score in the project.
    Players who missed the season carry `MISSED_SEASON_RANK`, which ranks them
    last — correctly, since missing the year is the outcome and dropping them
    would score the model only on players who stayed healthy.

    The percentile is taken over drafted players at that position in that season,
    which is the pool a drafter is actually choosing between.
    """
    if not df.height:
        return df
    return df.with_columns(
        (
            1.0
            - (
                pl.col("finish_pos_rank").rank("average").over(["label_season", "position"])
                / pl.len().over(["label_season", "position"])
            )
        ).alias("finish_pct")
    )


def _fit_fold(
    train: pl.DataFrame, test: pl.DataFrame, cols: list[str], alpha: float
) -> pl.DataFrame | None:
    """One season-forward fold: the model, plus price alone fit the same way."""
    if not test.height or train.height < 20:
        return None

    x_tr, used = bk._design(train, cols)
    if not used:
        return None
    x_te, _ = bk._design(test.select(used), used)
    if x_te.shape[1] != len(used):
        return None

    y_tr = train.get_column("finish_pct").to_numpy()
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=alpha).fit(scaler.transform(x_tr), y_tr)
    pred = model.predict(scaler.transform(x_te))

    j = used.index("adp_pos_rank") if "adp_pos_rank" in used else None
    if j is None:
        price = np.full(test.height, float(y_tr.mean()))
    else:
        s2 = StandardScaler().fit(x_tr[:, [j]])
        m2 = Ridge(alpha=alpha).fit(s2.transform(x_tr[:, [j]]), y_tr)
        price = m2.predict(s2.transform(x_te[:, [j]]))

    return test.select(
        "label_season", "gsis_id", "name", "position", "adp", "adp_pos_rank",
        "finish_pos_rank", "finish_pct",
    ).with_columns(
        pl.Series("pred", pred).round(4),
        pl.Series("pred_adp_only", price).round(4),
        pl.lit(train.height, dtype=pl.Int32).alias("n_train"),
        pl.lit(len(used), dtype=pl.Int32).alias("n_features"),
    )


def fit_predict(
    df: pl.DataFrame,
    feature_cols: dict[str, list[str]] | list[str] | None = None,
    min_train_seasons: int = 2,
    by_position: bool = True,
    alpha: float = ALPHA,
) -> pl.DataFrame:
    """Out-of-sample projections from expanding-window season-forward folds.

    Every row returned was predicted by a model that never saw its season.

    Returns: label_season, gsis_id, name, position, adp, adp_pos_rank,
    finish_pos_rank, finish_pct, pred, pred_adp_only, n_train, n_features, model.
    """
    if not df.height:
        return pl.DataFrame()
    df = target(df)

    out: list[pl.DataFrame] = []
    for train_seasons, test_season in bk.season_forward_splits(df, min_train_seasons):
        train_all = df.filter(pl.col("label_season").is_in(train_seasons))
        test_all = df.filter(pl.col("label_season") == test_season)

        for position in (sorted(df.get_column("position").unique().to_list()) if by_position else [None]):
            if position is None:
                train, test, tag = train_all, test_all, "pooled"
                cols = feature_cols if isinstance(feature_cols, list) else bk.model_features()
            else:
                train = train_all.filter(pl.col("position") == position)
                test = test_all.filter(pl.col("position") == position)
                tag = position
                if isinstance(feature_cols, dict):
                    cols = feature_cols.get(position, bk.model_features(position))
                elif isinstance(feature_cols, list):
                    cols = feature_cols
                else:
                    cols = bk.model_features(position)

            fold = _fit_fold(train, test, cols, alpha)
            if fold is not None:
                out.append(fold.with_columns(pl.lit(tag).alias("model")))

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed directly so there is no scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = pl.Series(a).rank("average").to_numpy()
    rb = pl.Series(b).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def skill(
    preds: pl.DataFrame, n_boot: int = 2000, seed: int = 0
) -> pl.DataFrame:
    """Rank correlation with the actual finish, for the model and for price alone.

    `delta` is the number that answers the real question. Bootstrapping resamples
    *seasons*, not rows, because two receivers in the same season share a labour
    market, a schedule, and a set of injuries — treating their errors as
    independent would report an interval several times too narrow.

    Returns: scope, n, spearman, spearman_adp_only, delta, delta_lo, delta_hi.
    """
    if not preds.height:
        return pl.DataFrame()

    rng = np.random.default_rng(seed)
    seasons = sorted(preds.get_column("label_season").unique().to_list())
    rows: list[dict[str, object]] = []

    def add(scope: str, sub: pl.DataFrame) -> None:
        if sub.height < 10:
            return
        y = sub.get_column("finish_pct").to_numpy()
        model = _spearman(sub.get_column("pred").to_numpy(), y)
        price = _spearman(sub.get_column("pred_adp_only").to_numpy(), y)

        deltas = []
        by_season = {s: sub.filter(pl.col("label_season") == s) for s in seasons}
        for _ in range(n_boot):
            picked = rng.choice(seasons, size=len(seasons), replace=True)
            frames = [by_season[s] for s in picked if by_season[s].height]
            if not frames:
                continue
            boot = pl.concat(frames, how="diagonal_relaxed")
            yy = boot.get_column("finish_pct").to_numpy()
            d = _spearman(boot.get_column("pred").to_numpy(), yy) - _spearman(
                boot.get_column("pred_adp_only").to_numpy(), yy
            )
            if not np.isnan(d):
                deltas.append(d)

        lo, hi = (
            (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
            if deltas
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "scope": scope,
                "n": sub.height,
                "spearman": round(model, 4),
                "spearman_adp_only": round(price, 4),
                "delta": round(model - price, 4),
                "delta_lo": round(lo, 4),
                "delta_hi": round(hi, 4),
            }
        )

    add("all", preds)
    for position in FANTASY_POSITIONS:
        add(position, preds.filter(pl.col("position") == position))
    for season in seasons:
        add(f"test {season}", preds.filter(pl.col("label_season") == season))

    return pl.DataFrame(rows)


def coefficients(
    df: pl.DataFrame,
    feature_cols: dict[str, list[str]] | None = None,
    alpha: float = ALPHA,
) -> pl.DataFrame:
    """Standardized Ridge coefficients per position, fit on everything.

    In-sample and for reading only — the out-of-sample number is `skill`. Useful
    for checking that a feature enters with the sign the theory says it should,
    which is a weaker claim than "it predicts" and a much easier one to falsify.

    Returns: position, feature, coefficient, n.
    """
    if not df.height:
        return pl.DataFrame()
    df = target(df)

    rows = []
    for position in sorted(df.get_column("position").unique().to_list()):
        sub = df.filter(pl.col("position") == position)
        cols = (feature_cols or {}).get(position) or bk.model_features(position)
        x, used = bk._design(sub, cols)
        if not used or sub.height < 20:
            continue
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=alpha).fit(scaler.transform(x), sub.get_column("finish_pct").to_numpy())
        for name, coef in zip(used, model.coef_):
            rows.append(
                {
                    "position": position,
                    "feature": name,
                    "coefficient": round(float(coef), 4),
                    "n": sub.height,
                }
            )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["position", "coefficient"], descending=[False, True])


def build(
    label_seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    feature_cols: dict[str, list[str]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience: the out-of-sample predictions and their skill table."""
    df = bk.training_frame(label_seasons, scoring)
    if not df.height:
        return pl.DataFrame(), pl.DataFrame()
    preds = fit_predict(df, feature_cols)
    return preds, skill(preds)
