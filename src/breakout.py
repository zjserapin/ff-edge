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

**Stratified by position**, because positions are not interchangeable: rushing
share means something at quarterback and nothing at receiver, and a pooled fit
must either drop such columns or feed every position a mostly-irrelevant vector.
That choice is not free — it splits 629 labeled player-seasons into QB 77, RB
184, TE 67, WR 212 — so `sample_adequacy` reports events per variable per
position and should be read before any number below.

**What it found: no usable edge, but stratifying removed a real pathology.**

    configuration                          out-of-sample AUC
    pooled, 10 features, C=1.0                    0.401
    pooled, 10 features, C=0.25                   0.398
    pooled,  4 features, C=0.25                   0.447
    stratified, 4 per position                    0.514

    stratified vs ADP-only            +0.027   95% CI [-0.035, +0.094]
    base rate                          0.224   (629 labeled player-seasons)

The pooled model was *anti*-predictive — below 0.5, with inverted calibration
(lowest-probability quartile hit 28%, highest hit 14%). Stratifying moves it to
roughly chance. The attribution above isolates why: shrinkage alone changed
nothing (0.401 -> 0.398), cutting to four features recovered part of it
(0.447), and fitting separately per position recovered the rest (0.514).
Pooling was mixing four different relationships and learning their average,
which fit none of them.

It still does not beat price. The difference interval covers zero at every
position. But "no signal" is a different and more honest result than "reliably
wrong", and it is the one supported here.

Three checks before trusting any of that, since a sub-0.5 AUC usually means a
sign error: shuffled labels give 0.497 across twelve seeds, so the pipeline is
correct; in-sample AUC is 0.630, so the fit does find structure; and sweeping
regularization from C=1.0 to C=0.001 on the pooled model made out-of-sample
*worse*, which overfitting noise does not do.

Per position, with the caveat that only WR has even a marginal sample:

    WR   AUC 0.556   delta +0.059   5.5 events per variable   thin
    RB   AUC 0.536   delta +0.026   4.3                       very thin
    QB   AUC 0.504   delta +0.025   2.5                       very thin
    TE   AUC 0.465   delta -0.076   1.5                       insufficient

Nothing here inverts a model and calls it a signal, and nothing reports a
position's result without its denominator.
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


# Four features per position, fixed in advance. See `model_features`.
POSITION_FEATURES: dict[str, list[str]] = {
    # Rushing is what separates a fantasy quarterback from a good one.
    "QB": ["adp_pos_rank", "age", "rush_share", "exp_pts_share"],
    # Snap share is the job; target share is whether he catches passes, which is
    # the difference between a two-down back and a every-down one.
    "RB": ["adp_pos_rank", "age", "snap_pct", "target_share"],
    # WOPR blends target share and air-yards share, which keeps two correlated
    # columns from splitting into a large +/- pair on a small sample.
    "WR": ["adp_pos_rank", "age", "wopr", "snap_pct"],
    "TE": ["adp_pos_rank", "age", "target_share", "snap_pct"],
}

# The pooled set, retained so stratified and pooled fits can be compared.
POOLED_FEATURES: list[str] = [
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


def model_features(position: str | None = None) -> list[str]:
    """Features the backtest may see, price included. Position-specific by default.

    Four per position, not ten, and the arithmetic forces it. Stratifying splits
    629 labeled player-seasons into QB 77, RB 184, TE 67, WR 212 — and
    season-forward folds cut that again, so the first QB fold trains on 36 rows
    with 8 positives. The usual floor is ten events per variable; ten features
    there would be under one. Four is still thin (two events per variable) and
    is the most this sample can carry without the coefficients being noise.

    The columns are chosen in advance from what is known to separate players at
    each position, never by searching for what scores well — with samples this
    small, searching *will* find something.

    `seasons_exp` is excluded everywhere despite being available: it correlates
    0.964 with `age`, and with both in the fit the pooled model split them into
    a large positive and a large negative coefficient (+0.55 / -0.64) and keyed
    on the noisy difference between two measurements of the same thing.
    """
    if position is None:
        return list(POOLED_FEATURES)
    return list(POSITION_FEATURES.get(position, POOLED_FEATURES))


def sample_adequacy(
    df: pl.DataFrame, min_train_seasons: int = 2
) -> pl.DataFrame:
    """Whether each position has enough data to support a model at all.

    Events per variable — positives in the smallest training fold divided by
    feature count — is the standard diagnostic for a logistic fit, and the
    conventional floor is ten. Nothing here reaches it. Reporting it beside every
    result is the difference between a model that is honestly thin and one that
    looks authoritative because nobody printed the denominator.

    Returns: position, n_total, positives, n_train_min, positives_train_min,
    n_test_min, n_features, events_per_variable, verdict.
    """
    if not df.height:
        return pl.DataFrame()

    splits = season_forward_splits(df, min_train_seasons)
    rows = []
    for position in sorted(df.get_column("position").unique().to_list()):
        sub = df.filter(pl.col("position") == position)
        cols = model_features(position)

        train_sizes, train_pos, test_sizes = [], [], []
        for train_seasons, test_season in splits:
            tr = sub.filter(pl.col("label_season").is_in(train_seasons))
            te = sub.filter(pl.col("label_season") == test_season)
            train_sizes.append(tr.height)
            train_pos.append(int(tr.get_column("beat_adp").sum()))
            test_sizes.append(te.height)

        min_pos = min(train_pos) if train_pos else 0
        epv = round(min_pos / len(cols), 2) if cols else 0.0
        rows.append(
            {
                "position": position,
                "n_total": sub.height,
                "positives": int(sub.get_column("beat_adp").sum()),
                "n_train_min": min(train_sizes) if train_sizes else 0,
                "positives_train_min": min_pos,
                "n_test_min": min(test_sizes) if test_sizes else 0,
                "n_features": len(cols),
                "events_per_variable": epv,
                "verdict": (
                    "adequate" if epv >= 10
                    else "thin" if epv >= 5
                    else "very thin" if epv >= 2
                    else "insufficient"
                ),
            }
        )

    return pl.DataFrame(rows).sort("events_per_variable", descending=True)


def _fit_fold(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cols: list[str],
    seed: int,
    C: float,
) -> pl.DataFrame | None:
    """One train/test fold: the model, plus the price-only null fit the same way."""
    if not test.height or train.get_column("beat_adp").n_unique() < 2:
        return None

    x_tr, used = _design(train, cols)
    if not used:
        return None
    x_te, _ = _design(test.select(used), used)
    if x_te.shape[1] != len(used):
        return None

    y_tr = train.get_column("beat_adp").to_numpy().astype(int)
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=5000, C=C, random_state=seed)
    model.fit(scaler.transform(x_tr), y_tr)
    p_full = model.predict_proba(scaler.transform(x_te))[:, 1]

    # The null: price alone, same family, same fold, same regularization.
    j = used.index("adp_pos_rank") if "adp_pos_rank" in used else None
    if j is None:
        p_adp = np.full(test.height, float(y_tr.mean()))
    else:
        s2 = StandardScaler().fit(x_tr[:, [j]])
        m2 = LogisticRegression(max_iter=5000, C=C, random_state=seed).fit(
            s2.transform(x_tr[:, [j]]), y_tr
        )
        p_adp = m2.predict_proba(s2.transform(x_te[:, [j]]))[:, 1]

    return test.select(
        "label_season", "gsis_id", "name", "position", "adp", "adp_pos_rank", "beat_adp"
    ).with_columns(
        pl.Series("p_breakout", p_full).round(4),
        pl.Series("p_adp_only", p_adp).round(4),
        pl.lit(train.height, dtype=pl.Int32).alias("n_train"),
        pl.lit(len(used), dtype=pl.Int32).alias("n_features"),
    )


def fit_predict(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    seed: int = 0,
    min_train_seasons: int = 2,
    by_position: bool = True,
    C: float | None = None,
    min_train_rows: int = 25,
) -> pl.DataFrame:
    """Out-of-sample predictions from season-forward folds.

    Every row returned was predicted by a model that never saw its season.

    `by_position` fits a separate model per position, which is the default
    because positions are not interchangeable: rushing share means something at
    quarterback and nothing at receiver, and a pooled fit has to either drop such
    columns or feed every position a mostly-irrelevant vector. The cost is
    sample size — see `sample_adequacy`, which reports events per variable for
    each position and is worth reading before any of these numbers.

    `C` defaults to 0.25 when stratified and 1.0 when pooled. Stronger shrinkage
    on the smaller samples is not tuning: it is the standard response to fitting
    four parameters on thirty-odd rows, and it was set from the sample size
    rather than by checking which value scored best on the test seasons.

    Returns: label_season, gsis_id, name, position, adp, adp_pos_rank,
    p_breakout, p_adp_only, beat_adp, n_train, n_features, model.
    """
    if not df.height:
        return pl.DataFrame()

    if C is None:
        C = 0.25 if by_position else 1.0

    splits = season_forward_splits(df, min_train_seasons)
    out: list[pl.DataFrame] = []

    for train_seasons, test_season in splits:
        train_all = df.filter(pl.col("label_season").is_in(train_seasons))
        test_all = df.filter(pl.col("label_season") == test_season)

        groups = (
            sorted(df.get_column("position").unique().to_list())
            if by_position
            else [None]
        )
        for position in groups:
            if position is None:
                train, test, cols, tag = train_all, test_all, (feature_cols or model_features()), "pooled"
            else:
                train = train_all.filter(pl.col("position") == position)
                test = test_all.filter(pl.col("position") == position)
                cols = feature_cols or model_features(position)
                tag = position

            # Too few rows to fit anything meaningful. Skipped rather than
            # fitted-and-reported, so a position that cannot support a model
            # produces no predictions instead of confident nonsense.
            if train.height < min_train_rows:
                continue

            fold = _fit_fold(train, test, cols, seed, C)
            if fold is not None:
                out.append(fold.with_columns(pl.lit(tag).alias("model")))

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

    add("all", preds)
    for position in preds.get_column("position").unique().sort().to_list():
        add(position, preds.filter(pl.col("position") == position))
    for season in preds.get_column("label_season").unique().sort().to_list():
        add(f"test {season}", preds.filter(pl.col("label_season") == season))

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def calibration_by_position(preds: pl.DataFrame, bins: int = 2) -> pl.DataFrame:
    """Calibration within each position, at whatever bin count the sample allows.

    Two bins, not four, and even that is generous: a position contributes 20-60
    out-of-sample rows, so four bins would be cells of a dozen players where the
    interval spans most of the unit interval. The `n` column is the one to read
    first.

    Returns the same columns as `calibration`, plus `position`.
    """
    if not preds.height:
        return pl.DataFrame()

    parts = []
    for position in preds.get_column("position").unique().sort().to_list():
        sub = preds.filter(pl.col("position") == position)
        if sub.height < bins * 8:
            continue
        cal = calibration(sub, bins=bins)
        if cal.height:
            parts.append(cal.with_columns(pl.lit(position).alias("position")))

    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


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
    by_position: bool = True,
) -> pl.DataFrame:
    """Refit on every label season, then score the upcoming draft.

    Stratified by default, so a quarterback is scored by a quarterback model.
    That also means `p_breakout` is only comparable *within* a position — four
    separate models produce four separate probability scales, and `quartile` is
    therefore computed within position rather than across the board.

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

    parts: list[pl.DataFrame] = []
    groups = (
        sorted(current.get_column("position").unique().to_list()) if by_position else [None]
    )

    for position in groups:
        if position is None:
            tr, cu, cols = train, current, model_features()
        else:
            tr = train.filter(pl.col("position") == position)
            cu = current.filter(pl.col("position") == position)
            cols = model_features(position)
        if tr.height < 25 or not cu.height or tr.get_column("beat_adp").n_unique() < 2:
            continue

        x_tr, used = _design(tr, cols)
        if not used:
            continue
        x_cu, _ = _design(cu.select([c for c in used if c in cu.columns]), used)
        if x_cu.shape[1] != len(used):
            continue

        y = tr.get_column("beat_adp").to_numpy().astype(int)
        scaler = StandardScaler().fit(x_tr)
        model = LogisticRegression(
            max_iter=5000, C=0.25 if by_position else 1.0, random_state=seed
        ).fit(scaler.transform(x_tr), y)

        parts.append(
            cu.select("gsis_id", "name", "position", "team", "adp", "adp_pos_rank").with_columns(
                pl.Series("p_breakout", model.predict_proba(scaler.transform(x_cu))[:, 1]).round(4)
            )
        )

    if not parts:
        return pl.DataFrame()

    return (
        pl.concat(parts, how="diagonal_relaxed")
        # Ranked within position: a stratified model's probabilities are only
        # comparable inside the position they were fit on. A cross-position
        # ranking of them would compare four different models' outputs.
        .with_columns(
            (
                pl.col("p_breakout").rank("ordinal").over("position") * 4
                // (pl.len().over("position") + 1)
                + 1
            )
            .cast(pl.Int32)
            .alias("quartile")
        )
        .sort(["position", "p_breakout"], descending=[False, True])
    )


def coefficients(
    df: pl.DataFrame | None = None, seed: int = 0, by_position: bool = True
) -> pl.DataFrame:
    """Standardized coefficients from a fit on all label seasons.

    Shown so the app can say what each model is actually keying on rather than
    implying that every feature is doing work. On samples this small the
    coefficients are themselves unstable — read them as a description of this
    fit, not as estimates of an effect.

    Returns: model, feature, coef, odds_ratio, n_train.
    """
    train = df if df is not None else training_frame()
    if not train.height:
        return pl.DataFrame()

    groups = (
        sorted(train.get_column("position").unique().to_list()) if by_position else [None]
    )
    parts = []
    for position in groups:
        sub = train if position is None else train.filter(pl.col("position") == position)
        cols = model_features(position)
        if sub.height < 25 or sub.get_column("beat_adp").n_unique() < 2:
            continue

        x, used = _design(sub, cols)
        if not used:
            continue
        y = sub.get_column("beat_adp").to_numpy().astype(int)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            max_iter=5000, C=0.25 if by_position else 1.0, random_state=seed
        ).fit(scaler.transform(x), y)

        parts.append(
            pl.DataFrame(
                {
                    "model": [position or "pooled"] * len(used),
                    "feature": used,
                    "coef": [round(float(c), 4) for c in model.coef_[0]],
                    "odds_ratio": [round(float(np.exp(c)), 4) for c in model.coef_[0]],
                    "n_train": [sub.height] * len(used),
                }
            )
        )

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort(
        ["model", pl.col("coef").abs()], descending=[False, True]
    )
