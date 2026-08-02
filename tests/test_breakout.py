"""Backtest integrity.

The result of this module is a negative one — prior-season usage does not beat
ADP — and a negative result is only worth reporting if the machinery producing
it is known to be sound. So most of these tests are about the machinery: that
the labels are not degenerate, that no future information reaches the model, and
that a shuffled target collapses to chance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import breakout as bo
from src import features as ft
from src import uncertainty as unc


@pytest.fixture(scope="module")
def labeled() -> pl.DataFrame:
    df = bo.labels()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    df = bo.training_frame(features=ft.build())
    if not df.height:
        pytest.skip("cold cache")
    return df


@pytest.fixture(scope="module")
def preds(train: pl.DataFrame) -> pl.DataFrame:
    return bo.fit_predict(train)


# --- the label --------------------------------------------------------------


def test_ratio_label_is_flat_across_price_tiers(labeled: pl.DataFrame) -> None:
    """The label must measure outperformance, not how cheap a player was.

    This is why the interview's tier rule was replaced. Under the ratio rule the
    rate should be roughly constant across ADP tiers; a monotone climb means the
    label is a proxy for draft position and any model will just learn price.
    """
    by_tier = (
        labeled.filter(pl.col("adp_tier") <= 6)
        .group_by("adp_tier")
        .agg(pl.col("beat_adp").mean().alias("rate"), pl.len().alias("n"))
        .filter(pl.col("n") >= 30)
        .sort("adp_tier")
    )
    rates = by_tier.get_column("rate").to_list()
    assert len(rates) >= 4
    assert max(rates) - min(rates) < 0.20, f"label varies too much by tier: {rates}"


def test_tier_label_is_degenerate_at_the_top(labeled: pl.DataFrame) -> None:
    """Documents *why* the tier rule was rejected, so nobody restores it.

    A tier-1 player would have to finish above positional rank zero to beat his
    tier, so the rate there is exactly zero by construction — on a quarter of
    the sample.
    """
    tier1 = labeled.filter(pl.col("adp_tier") == 1)
    assert tier1.height > 100
    assert tier1.get_column("beat_adp_tier").sum() == 0
    # The ratio rule, on the same players, is not degenerate.
    assert tier1.get_column("beat_adp").mean() > 0.15


def test_base_rate_is_stable_across_seasons(labeled: pl.DataFrame) -> None:
    """A label whose rate swings by season would make folds incomparable."""
    rates = (
        labeled.group_by("season")
        .agg(pl.col("beat_adp").mean().alias("rate"))
        .get_column("rate")
        .to_list()
    )
    assert max(rates) - min(rates) < 0.10, f"base rate unstable: {rates}"


def test_missed_seasons_are_labeled_not_dropped(labeled: pl.DataFrame) -> None:
    """Drafted players who never posted a season must count as failures.

    Dropping them would score the model only on players who stayed healthy,
    which is survivorship bias of the most flattering kind.
    """
    missed = labeled.filter(pl.col("finish_pos_rank") == bo.MISSED_SEASON_RANK)
    assert missed.height > 0
    assert not missed.get_column("beat_adp").any()


def test_adp_join_rate_is_high_on_skill_positions(labeled: pl.DataFrame) -> None:
    """FFC ships no ids, so this is a name join and must be watched."""
    per_season = labeled.group_by("season").agg(pl.len().alias("n"))
    assert per_season.get_column("n").min() >= 100


# --- no leakage -------------------------------------------------------------


def test_features_come_from_the_prior_season(train: pl.DataFrame) -> None:
    """The whole design is a one-year offset. Verify it rather than trust it."""
    assert (train.get_column("label_season") - train.get_column("season") == 1).all()


def test_outcome_columns_are_not_model_features() -> None:
    """The label season's production must never appear among the inputs."""
    feats = set(bo.model_features())
    assert not (feats & {"ppg", "fantasy_points", "pos_rank", "finish_pos_rank", "beat_adp"})


def test_splits_are_forward_only(train: pl.DataFrame) -> None:
    """Every training season must precede its test season."""
    splits = bo.season_forward_splits(train)
    assert splits
    for train_seasons, test_season in splits:
        assert max(train_seasons) < test_season


def test_predictions_are_out_of_sample(train: pl.DataFrame, preds: pl.DataFrame) -> None:
    """Only seasons that served as a test fold may appear in the predictions."""
    tested = {s for _, s in bo.season_forward_splits(train)}
    assert set(preds.get_column("label_season").unique().to_list()) <= tested


def test_shuffled_labels_collapse_to_chance(train: pl.DataFrame) -> None:
    """The decisive check on a below-0.5 AUC: is the pipeline wired correctly?

    Shuffling the target within season destroys any real relationship while
    preserving class balance. If AUC still lands near 0.5 across seeds, the
    machinery is sound and the poor real-data AUC is a property of the data,
    not a sign error.
    """
    aucs = []
    for seed in range(6):
        shuffled = train.with_columns(
            pl.col("beat_adp").shuffle(seed=seed).over("label_season")
        )
        p = bo.fit_predict(shuffled)
        if p.height:
            aucs.append(
                unc.auc(
                    p.get_column("beat_adp").to_numpy().astype(int),
                    p.get_column("p_breakout").to_numpy(),
                )
            )
    assert aucs
    assert 0.40 < float(np.mean(aucs)) < 0.60, f"shuffled AUC {np.mean(aucs):.3f} is not chance"


# --- reporting --------------------------------------------------------------


def test_adp_baseline_is_always_reported(preds: pl.DataFrame) -> None:
    """An AUC without its null is uninterpretable, so the null is not optional."""
    assert "p_adp_only" in preds.columns
    disc = bo.discrimination(preds, n_boot=200)
    assert {"auc", "auc_adp_only", "delta_auc", "delta_lo", "delta_hi"} <= set(disc.columns)


def test_calibration_bins_are_large_enough(preds: pl.DataFrame) -> None:
    """Four bins, not ten. A decile here would be ~30 players and +/-7 points."""
    cal = bo.calibration(preds, bins=4)
    assert cal.height == 4
    assert cal.get_column("n").min() >= 50


def test_base_rates_carry_intervals(labeled: pl.DataFrame) -> None:
    br = bo.base_rates(labeled)
    assert {"rate", "ci_lo", "ci_hi", "n"} <= set(br.columns)
    overall = br.filter(pl.col("scope") == "overall")
    assert overall.height == 1
    lo, rate, hi = (
        overall.get_column("ci_lo")[0],
        overall.get_column("rate")[0],
        overall.get_column("ci_hi")[0],
    )
    assert lo < rate < hi


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = unc.wilson_interval(22, 100)
    assert lo < 0.22 < hi
    assert 0.0 <= lo and hi <= 1.0
    # Degenerate inputs must not raise.
    assert unc.wilson_interval(0, 0) == (0.0, 0.0)


def test_auc_matches_sklearn() -> None:
    """The hand-rolled rank-sum AUC must agree with the reference implementation."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y * 0.5
    assert abs(unc.auc(y, s) - roc_auc_score(y, s)) < 1e-9
    # Ties handled by mid-rank, and a single-class input returns nan.
    assert abs(unc.auc(y, np.ones(400)) - 0.5) < 1e-9
    assert np.isnan(unc.auc(np.ones(10), rng.normal(size=10)))


def test_cluster_bootstrap_is_wider_than_naive(preds: pl.DataFrame) -> None:
    """Resampling groups must report more uncertainty than resampling rows.

    This is the property the strategy simulation depends on. If the clustered
    interval were not wider, the correlated-observations problem would be
    invisible and the sim would report intervals an order of magnitude too tight.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 250)
    # Strong between-group variation, little within-group.
    values = np.repeat(rng.normal(0, 1.0, 4), 250) + rng.normal(0, 0.05, 1000)

    n_lo, n_hi = unc.bootstrap_ci(values, np.mean, n_boot=500)
    c_lo, c_hi = unc.cluster_bootstrap_ci(values, groups, np.mean, n_boot=500)
    assert (c_hi - c_lo) > (n_hi - n_lo) * 3


# --- stratification ---------------------------------------------------------


def test_each_position_gets_its_own_model(train: pl.DataFrame) -> None:
    """Stratified fitting must produce a model tagged per position."""
    preds = bo.fit_predict(train, by_position=True)
    assert preds.height > 200
    models = set(preds.get_column("model").unique().to_list())
    assert models == {"QB", "RB", "WR", "TE"}
    # A player is only ever scored by his own position's model.
    assert (preds.get_column("model") == preds.get_column("position")).all()


def test_position_feature_sets_are_position_appropriate() -> None:
    """The features must differ by position, or stratifying buys nothing."""
    qb = set(bo.model_features("QB"))
    wr = set(bo.model_features("WR"))
    assert qb != wr
    # Rushing share is a quarterback feature; air-yards volume is a receiver one.
    assert "rush_share" in qb and "rush_share" not in wr
    assert "wopr" in wr and "wopr" not in qb
    # Price and age anchor every model.
    for position in ("QB", "RB", "WR", "TE"):
        cols = bo.model_features(position)
        assert "adp_pos_rank" in cols and "age" in cols
        assert len(cols) == 4, f"{position} should stay compact on this sample"


def test_stratifying_beats_pooling_on_this_data(train: pl.DataFrame) -> None:
    """The reason the default changed, pinned so a regression is visible.

    On the original 2020-2025 window the pooled model was *anti*-predictive
    (AUC ~0.40) and stratifying moved it to roughly chance. Widening to 2018
    softened that — pooled now lands near 0.47 rather than 0.40, which is what
    more data does to a fit that was mostly variance — so the assertion here is
    the durable half of the finding: stratifying is better than pooling, and
    pooling is not above chance.
    """
    strat = bo.fit_predict(train, by_position=True)
    pooled = bo.fit_predict(train, by_position=False)

    def auc_of(p: pl.DataFrame) -> float:
        return unc.auc(
            p.get_column("beat_adp").to_numpy().astype(int),
            p.get_column("p_breakout").to_numpy(),
        )

    assert auc_of(pooled) < 0.50, "pooled model is no longer below chance"
    assert auc_of(strat) > auc_of(pooled) + 0.02


def test_sample_adequacy_reports_the_denominator(train: pl.DataFrame) -> None:
    """Every position's result must ship with the sample that produced it.

    Nothing here clears the conventional ten-events-per-variable floor, and the
    table has to say so rather than letting four positions look equally solid.
    """
    adequacy = bo.sample_adequacy(train)
    assert set(adequacy.get_column("position").to_list()) == {"QB", "RB", "WR", "TE"}
    assert (adequacy.get_column("events_per_variable") < 10).all()

    # Ordering must reflect reality: WR has the most data, TE the least.
    ranked = adequacy.sort("events_per_variable", descending=True)
    assert ranked.get_column("position")[0] == "WR"
    assert ranked.get_column("position")[-1] == "TE"
    # Widening the window to 2018 moved TE from "insufficient" to "very thin".
    # The assertion is that it is still nowhere near adequate, not that it sits
    # in one particular bucket — pinning the bucket would fail on good news.
    assert ranked.get_column("verdict")[-1] in ("insufficient", "very thin")
    assert ranked.get_column("events_per_variable")[-1] < 5


def test_current_scores_are_ranked_within_position() -> None:
    """Four models mean four probability scales, so quartiles must be per position."""
    scored = bo.score_current(features=ft.build())
    if not scored.height:
        pytest.skip("cold cache")
    for position in scored.get_column("position").unique().to_list():
        sub = scored.filter(pl.col("position") == position)
        if sub.height < 8:
            continue
        assert set(sub.get_column("quartile").unique().to_list()) <= {1, 2, 3, 4}
        assert sub.get_column("quartile").n_unique() >= 2


def test_coefficients_are_reported_per_model(train: pl.DataFrame) -> None:
    coefs = bo.coefficients(train, by_position=True)
    assert set(coefs.get_column("model").unique().to_list()) == {"QB", "RB", "WR", "TE"}
    # Each model reports exactly its own feature set, with its own sample size.
    for position in ("QB", "RB", "WR", "TE"):
        sub = coefs.filter(pl.col("model") == position)
        assert set(sub.get_column("feature").to_list()) == set(bo.model_features(position))
        assert sub.get_column("n_train").n_unique() == 1
