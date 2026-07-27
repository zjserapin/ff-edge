"""Did prior-season usage predict beating ADP? Backtested honestly.

This is the validation half of the player track. `archetypes.py` describes; this
one is asked to predict, and is set up so that failing is a reportable result
rather than something to tune away.

**The label.** The design interview settled on "beat ADP by a positional tier",
and measuring it killed it. Across 2021-2024, the tier rule's base rate by ADP
tier is:

    tier 1  0.000 (n=160)   tier 4  0.312
    tier 2  0.207           tier 5  0.412
    tier 3  0.323           tier 6  0.451

A tier-1 player must finish above rank zero to qualify, so a quarter of the
sample is structurally incapable of a positive label, and the rate then climbs
monotonically with how late you were drafted. That is a measure of price, not of
performance, and a model trained on it learns `adp_pos_rank` and stops. The
ratio rule — finish at or inside 60% of your ADP positional rank — is flat
across tiers at roughly 0.22 and defined everywhere. It is the default here;
`adp_tier` and `tier_delta` are still carried for display, and `beat_ratio` is a
parameter, so the tier framing is one argument away.

**The honest size.** 629 labeled player-seasons across four label years, ~140
positives, and season-forward validation leaves exactly two test folds. Every
design choice below follows from that: a linear model rather than a boosted one,
four calibration bins rather than ten, and an ADP-only baseline reported beside
every score.

**What it found: nothing usable, and that is the result.**

    pooled out-of-sample AUC   0.401   95% CI [0.327, 0.476]
    ADP-only baseline AUC      0.472
    difference                -0.071   95% CI [-0.173, 0.028]
    base rate                  0.224   (629 labeled player-seasons)

The model does not beat price alone — the difference interval covers zero — and
its own AUC lands below 0.5. Calibration is inverted: the lowest-probability
quartile hit 28% and the highest hit 14%.

Three things were checked before accepting that, because a sub-0.5 AUC usually
means a sign error:

  Shuffling the labels within season gives AUC 0.497 across twelve seeds
  (range 0.412-0.582), so the pipeline is wired correctly.

  In-sample AUC is 0.630 against 0.438 out of sample, so the fit is finding
  structure in the training seasons.

  Sweeping regularization from C=1.0 down to C=0.001 makes out-of-sample AUC
  *worse* (0.438 -> 0.395), not better. Overfitting noise would improve under
  shrinkage. This does not, which means the relationship genuinely reverses
  between the training seasons and the test seasons.

The plausible reading is mean reversion against an efficient market: ADP already
prices last season's usage, so the players whose usage most impressed the market
are priced past what they repeat. That is a coherent story and it is *not*
established here — two test folds cannot separate it from regime noise.

Which is why nothing in this module inverts the model and calls it a signal. An
anti-predictive model on 284 out-of-sample rows is a reason to distrust the
features, not a reason to bet the other way.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import adp as adp_mod
from src import features as ft
from src import ids
from src import scoring as sc
from src import uncertainty as unc
from src.config import (
    ADP_MISSING_YEARS,
    CURRENT_SEASON,
    DEFAULT_SCORING,
    FANTASY_POSITIONS,
    LABEL_SEASONS,
    LEAGUE_ADP_SCORING,
    LEAGUE_ADP_TEAMS,
    REGULAR_SEASON_WEEKS,
    SEASON,
)

# Rank assigned to a drafted player who did not post a qualifying season. Not a
# null: missing the year is the outcome, and dropping those rows would score the
# model only on players who stayed healthy enough to be scored.
MISSED_SEASON_RANK = 999


def adp_board(
    season: int,
    scoring: str = LEAGUE_ADP_SCORING,
    teams: int = LEAGUE_ADP_TEAMS,
    force: bool = False,
) -> pl.DataFrame:
    """That season's draft market, joined to nflverse ids.

    FFC ships no player ids, so this is a normalized-name join. On skill
    positions it lands at 97-99%; the ~82% figure quoted for the raw feed is the
    all-positions rate, dragged down by kickers and defenses that have no
    gsis_id to match and are dropped here anyway.

    Returns: season, gsis_id, name, position, team, adp, stdev, adp_pos_rank,
    adp_tier.
    """
    board = adp_mod.fetch(scoring, teams, season, force=force)
    if not board.height:
        return pl.DataFrame()

    matched = ids.match_by_name(board, "name", "position").filter(
        pl.col("position").is_in(list(FANTASY_POSITIONS)) & pl.col("gsis_id").is_not_null()
    )
    if not matched.height:
        return pl.DataFrame()

    return matched.select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        "gsis_id",
        "name",
        "position",
        "team",
        "adp",
        "stdev",
    ).with_columns(
        pl.col("adp").rank("ordinal").over("position").cast(pl.Int32).alias("adp_pos_rank"),
        pl.col("adp").rank("ordinal").cast(pl.Int32).alias("adp_overall_rank"),
    ).with_columns(
        (((pl.col("adp_pos_rank") - 1) // 10) + 1).cast(pl.Int32).alias("adp_tier")
    )


def labels(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    beat_ratio: float = 0.6,
    tier_size: int = 10,
    weeks: tuple[int, int] = (1, REGULAR_SEASON_WEEKS),
    force: bool = False,
) -> pl.DataFrame:
    """Who beat their draft price, per season.

    Positional finish is computed against *every* player at the position, not
    only the drafted ones — a waiver pickup who finishes RB8 genuinely pushes
    everyone below him down, and pretending the undrafted don't exist would
    inflate every label.

    Returns: season, gsis_id, name, position, adp, stdev, adp_pos_rank,
    adp_tier, finish_pos_rank, finish_tier, tier_delta, rank_ratio, beat_adp,
    beat_adp_tier, games.
    """
    seasons = [s for s in (seasons or LABEL_SEASONS) if s not in ADP_MISSING_YEARS]
    scoring = scoring or DEFAULT_SCORING

    finishes = sc.score_season(seasons, scoring, weeks=weeks, force=force).select(
        "season",
        pl.col("player_id").alias("gsis_id"),
        pl.col("pos_rank").alias("finish_pos_rank"),
        "games",
        "ppg",
    )

    boards = [adp_board(s, force=force) for s in seasons]
    boards = [b for b in boards if b.height]
    if not boards:
        return pl.DataFrame()
    board = pl.concat(boards, how="diagonal_relaxed")

    joined = board.join(finishes, on=["season", "gsis_id"], how="left").with_columns(
        pl.col("finish_pos_rank").fill_null(MISSED_SEASON_RANK),
        pl.col("games").fill_null(0),
    )

    return joined.with_columns(
        (((pl.col("finish_pos_rank") - 1) // tier_size) + 1).cast(pl.Int32).alias("finish_tier"),
        (pl.col("finish_pos_rank") / pl.col("adp_pos_rank")).alias("rank_ratio"),
    ).with_columns(
        (pl.col("adp_tier") - pl.col("finish_tier")).alias("tier_delta"),
        # Primary label: finished at or inside `beat_ratio` of your price.
        (pl.col("finish_pos_rank") <= (pl.col("adp_pos_rank") * beat_ratio).ceil())
        .alias("beat_adp"),
        # The interview's original framing, kept for display and comparison.
        (pl.col("finish_tier") <= pl.col("adp_tier") - 1).alias("beat_adp_tier"),
    ).sort(["season", "position", "adp_pos_rank"])


def base_rates(labeled: pl.DataFrame, label_col: str = "beat_adp") -> pl.DataFrame:
    """How often the label fires, with Wilson intervals. The app leads with this.

    Reported overall, by season, by position and by ADP tier, because "the model
    hits 28%" means nothing until you know whether 28% is above or below what
    guessing would give you.

    Returns: scope, value, n, hits, rate, ci_lo, ci_hi.
    """
    if not labeled.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []

    def add(scope: str, value: object, sub: pl.DataFrame) -> None:
        n = sub.height
        hits = int(sub.get_column(label_col).sum())
        lo, hi = unc.wilson_interval(hits, n)
        rows.append(
            {
                "scope": scope,
                "value": str(value),
                "n": n,
                "hits": hits,
                "rate": round(hits / n, 4) if n else 0.0,
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
        )

    add("overall", "all", labeled)
    for col, scope in (("season", "season"), ("position", "position"), ("adp_tier", "adp_tier")):
        for value in labeled.get_column(col).unique().sort().to_list():
            add(scope, value, labeled.filter(pl.col(col) == value))

    return pl.DataFrame(rows)


def training_frame(
    label_seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    beat_ratio: float = 0.6,
    min_prior_games: int = 4,
    features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Prior-season features joined to next-season labels.

    The year offset is the whole design: 2023 usage against the 2024 label. Any
    feature from the label season itself would be leakage of the most trivial
    kind.

    `adp_pos_rank` is deliberately included as a feature. Withholding the price
    would let the model rediscover "good players are good" and score well for it;
    including it forces the question that matters, which is whether usage says
    anything the market has not already priced.
    """
    label_seasons = [s for s in (label_seasons or LABEL_SEASONS) if s not in ADP_MISSING_YEARS]
    feats = features if features is not None else ft.build()
    if not feats.height:
        return pl.DataFrame()

    lab = labels(label_seasons, scoring, beat_ratio=beat_ratio)
    if not lab.height:
        return pl.DataFrame()

    prior = feats.filter(pl.col("games") >= min_prior_games).with_columns(
        (pl.col("season") + 1).alias("label_season")
    )

    return (
        lab.rename({"season": "label_season"})
        .join(
            prior.rename({"player_id": "gsis_id"}).drop(
                [c for c in ("position", "team", "player_name") if c in prior.columns]
            ),
            on=["label_season", "gsis_id"],
            how="inner",
        )
        .sort(["label_season", "position", "adp_pos_rank"])
    )


def season_forward_splits(
    df: pl.DataFrame, min_train_seasons: int = 2
) -> list[tuple[list[int], int]]:
    """Expanding-window splits: train on earlier seasons, test on the next one.

    Never a random split. Player-seasons are not exchangeable across time —
    a random fold would train on 2024 and test on 2022, letting the model learn
    from the future and report a flattering number that would not survive
    contact with a real draft.
    """
    seasons = sorted(df.get_column("label_season").unique().to_list())
    return [
        (seasons[:i], seasons[i])
        for i in range(min_train_seasons, len(seasons))
    ]


def _design(
    df: pl.DataFrame, cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Median-imputed, standardization-ready matrix over `cols`."""
    usable = [c for c in cols if c in df.columns and df.get_column(c).is_not_null().any()]
    if not usable:
        return np.empty((df.height, 0)), []
    filled = df.select(
        [pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).median()) for c in usable]
    )
    keep = [c for c in usable if filled.get_column(c).is_not_null().all()]
    return (filled.select(keep).to_numpy() if keep else np.empty((df.height, 0))), keep


def model_features() -> list[str]:
    """Features the backtest is allowed to see, price included.

    Kept small on purpose. With ~400 training rows, every additional column buys
    variance more readily than signal.

    `seasons_exp` is excluded despite being available: it correlates 0.964 with
    `age`, and with both in the fit the model split them into a large positive
    and a large negative coefficient (+0.55 / -0.64) and keyed on the noisy
    difference between two measurements of the same thing.
    """
    return [
        "adp_pos_rank",
        "snap_pct",
        "target_share",
        "rush_share",
        "exp_pts_share",
        "tgt_per_game",
        "carry_per_game",
        "pts_over_exp_per_game",
        "age",
        "draft_round",
    ]


def fit_predict(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    seed: int = 0,
    min_train_seasons: int = 2,
) -> pl.DataFrame:
    """Out-of-sample predictions from season-forward folds.

    Every row returned was predicted by a model that never saw its season. The
    `p_adp_only` column is the null hypothesis fit the same way on price alone —
    without it, an AUC of 0.62 is uninterpretable, because ADP by itself is
    already a decent predictor of beating ADP.

    Returns: label_season, gsis_id, name, position, adp, adp_pos_rank,
    p_breakout, p_adp_only, beat_adp.
    """
    if not df.height:
        return pl.DataFrame()

    cols = feature_cols or model_features()
    out: list[pl.DataFrame] = []

    for train_seasons, test_season in season_forward_splits(df, min_train_seasons):
        train = df.filter(pl.col("label_season").is_in(train_seasons))
        test = df.filter(pl.col("label_season") == test_season)
        if train.height < 50 or not test.height:
            continue
        if train.get_column("beat_adp").n_unique() < 2:
            continue

        x_tr, used = _design(train, cols)
        if not used:
            continue
        x_te, _ = _design(test.select(used), used)
        y_tr = train.get_column("beat_adp").to_numpy().astype(int)

        scaler = StandardScaler().fit(x_tr)
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        model.fit(scaler.transform(x_tr), y_tr)
        p_full = model.predict_proba(scaler.transform(x_te))[:, 1]

        # The baseline: price alone, same family, same fold.
        j = used.index("adp_pos_rank") if "adp_pos_rank" in used else None
        if j is None:
            p_adp = np.full(test.height, float(y_tr.mean()))
        else:
            s2 = StandardScaler().fit(x_tr[:, [j]])
            m2 = LogisticRegression(max_iter=2000, random_state=seed).fit(
                s2.transform(x_tr[:, [j]]), y_tr
            )
            p_adp = m2.predict_proba(s2.transform(x_te[:, [j]]))[:, 1]

        out.append(
            test.select(
                "label_season", "gsis_id", "name", "position", "adp", "adp_pos_rank", "beat_adp"
            ).with_columns(
                pl.Series("p_breakout", p_full).round(4),
                pl.Series("p_adp_only", p_adp).round(4),
            )
        )

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def discrimination(
    preds: pl.DataFrame, n_boot: int = 2000, seed: int = 0
) -> pl.DataFrame:
    """AUC for the model and for price alone, with the gap between them.

    `delta_auc` is the number that answers the actual question. If its interval
    covers zero, prior-season usage adds nothing to what the draft market
    already knows, and that is the finding.

    Returns: scope, n, positives, auc, auc_lo, auc_hi, auc_adp_only, delta_auc,
    delta_lo, delta_hi.
    """
    if not preds.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []

    def add(scope: str, sub: pl.DataFrame) -> None:
        y = sub.get_column("beat_adp").to_numpy().astype(int)
        p = sub.get_column("p_breakout").to_numpy()
        a = sub.get_column("p_adp_only").to_numpy()
        if len(np.unique(y)) < 2:
            return

        stacked = np.column_stack([y, p, a])
        lo, hi = unc.bootstrap_ci(
            stacked, lambda v: unc.auc(v[:, 0], v[:, 1]), n_boot=n_boot, seed=seed
        )
        d_lo, d_hi = unc.bootstrap_ci(
            stacked,
            lambda v: unc.auc(v[:, 0], v[:, 1]) - unc.auc(v[:, 0], v[:, 2]),
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "scope": scope,
                "n": sub.height,
                "positives": int(y.sum()),
                "auc": round(unc.auc(y, p), 4),
                "auc_lo": round(lo, 4),
                "auc_hi": round(hi, 4),
                "auc_adp_only": round(unc.auc(y, a), 4),
                "delta_auc": round(unc.auc(y, p) - unc.auc(y, a), 4),
                "delta_lo": round(d_lo, 4),
                "delta_hi": round(d_hi, 4),
            }
        )

    add("pooled", preds)
    for season in preds.get_column("label_season").unique().sort().to_list():
        add(f"test {season}", preds.filter(pl.col("label_season") == season))

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def calibration(preds: pl.DataFrame, bins: int = 4) -> pl.DataFrame:
    """Predicted probability versus what actually happened, in `bins` buckets.

    Four bins, not ten, and the arithmetic is the reason. Two test folds give
    roughly 320 out-of-sample rows; a top decile is 32 players, and at a 22% base
    rate the standard error on that cell is about +/-7 points — wide enough that
    30% and 20% are the same number. Four bins of ~80 halve that. Still wide, but
    reportable, and the interval is shown so nobody reads more into it.

    Returns: bin, p_lo, p_hi, n, mean_predicted, actual_rate, ci_lo, ci_hi,
    base_rate, lift.
    """
    if not preds.height:
        return pl.DataFrame()

    base = float(preds.get_column("beat_adp").mean())
    ranked = preds.with_columns(
        pl.col("p_breakout").rank("ordinal").alias("_r")
    ).with_columns(
        ((pl.col("_r") - 1) * bins // preds.height).clip(0, bins - 1).alias("bin")
    )

    rows: list[dict[str, object]] = []
    for b in range(bins):
        sub = ranked.filter(pl.col("bin") == b)
        if not sub.height:
            continue
        hits = int(sub.get_column("beat_adp").sum())
        lo, hi = unc.wilson_interval(hits, sub.height)
        rows.append(
            {
                "bin": b + 1,
                "p_lo": round(float(sub.get_column("p_breakout").min()), 4),
                "p_hi": round(float(sub.get_column("p_breakout").max()), 4),
                "n": sub.height,
                "mean_predicted": round(float(sub.get_column("p_breakout").mean()), 4),
                "actual_rate": round(hits / sub.height, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "base_rate": round(base, 4),
                "lift": round((hits / sub.height) / base, 3) if base else None,
            }
        )

    return pl.DataFrame(rows)


def score_current(
    feature_season: int = CURRENT_SEASON,
    adp_season: int = SEASON,
    beat_ratio: float = 0.6,
    seed: int = 0,
    features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Refit on every label season, then score the upcoming draft.

    The output is a probability with a known and modest discrimination — read it
    next to the calibration table, not on its own.

    Returns: gsis_id, name, position, team, adp, adp_pos_rank, p_breakout,
    quartile.
    """
    train = training_frame(beat_ratio=beat_ratio, features=features)
    if not train.height:
        return pl.DataFrame()

    feats = features if features is not None else ft.build()
    board = adp_board(adp_season)
    if not board.height:
        return pl.DataFrame()

    current = board.join(
        feats.filter(pl.col("season") == feature_season)
        .rename({"player_id": "gsis_id"})
        .drop([c for c in ("position", "team", "player_name", "season") if c in feats.columns]),
        on="gsis_id",
        how="inner",
    )
    if not current.height:
        return pl.DataFrame()

    cols = model_features()
    x_tr, used = _design(train, cols)
    if not used:
        return pl.DataFrame()
    x_cu, _ = _design(current.select([c for c in used if c in current.columns]), used)
    if x_cu.shape[1] != len(used):
        return pl.DataFrame()

    y = train.get_column("beat_adp").to_numpy().astype(int)
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(scaler.transform(x_tr), y)
    p = model.predict_proba(scaler.transform(x_cu))[:, 1]

    return (
        current.select("gsis_id", "name", "position", "team", "adp", "adp_pos_rank")
        .with_columns(pl.Series("p_breakout", p).round(4))
        .with_columns(
            (pl.col("p_breakout").rank("ordinal") * 4 // (pl.len() + 1) + 1)
            .cast(pl.Int32)
            .alias("quartile")
        )
        .sort("p_breakout", descending=True)
    )


def coefficients(
    df: pl.DataFrame | None = None, seed: int = 0
) -> pl.DataFrame:
    """Standardized coefficients from a fit on all label seasons.

    Shown so the app can say what the model is actually keying on rather than
    implying that ten features are each doing work.

    Returns: feature, coef, odds_ratio.
    """
    train = df if df is not None else training_frame()
    if not train.height:
        return pl.DataFrame()

    x, used = _design(train, model_features())
    if not used:
        return pl.DataFrame()
    y = train.get_column("beat_adp").to_numpy().astype(int)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(scaler.transform(x), y)

    return pl.DataFrame(
        {
            "feature": used,
            "coef": [round(float(c), 4) for c in model.coef_[0]],
            "odds_ratio": [round(float(np.exp(c)), 4) for c in model.coef_[0]],
        }
    ).sort(pl.col("coef").abs(), descending=True)
