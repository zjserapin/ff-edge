"""Continuous-target backtest: the honest version of "does this beat the market"."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import breakout as bk
from src import projection as pj


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    df = bk.training_frame()
    if not df.height:
        pytest.skip("cold cache")
    return df


@pytest.fixture(scope="module")
def preds(train: pl.DataFrame) -> pl.DataFrame:
    p = pj.fit_predict(train)
    if not p.height:
        pytest.skip("nothing fit")
    return p


def test_target_is_a_percentile_and_bigger_is_better(train: pl.DataFrame) -> None:
    """`finish_pct` must point the same way as every other score in the project."""
    scored = pj.target(train)
    values = scored.get_column("finish_pct")
    assert values.min() >= 0.0
    assert values.max() <= 1.0

    # The best positional finish in a season must carry the highest percentile.
    one = scored.filter(pl.col("label_season") == scored.get_column("label_season")[0])
    for position in one.get_column("position").unique().to_list():
        sub = one.filter(pl.col("position") == position)
        if sub.height < 5:
            continue
        best = sub.sort("finish_pos_rank").head(1)
        worst = sub.sort("finish_pos_rank", descending=True).head(1)
        assert best.get_column("finish_pct")[0] > worst.get_column("finish_pct")[0], (
            f"{position}: finishing first scored below finishing last"
        )


def test_missed_seasons_rank_last(train: pl.DataFrame) -> None:
    """Missing the year is the outcome, not missing data.

    Dropping those rows would score the model only on players who stayed healthy,
    which is the single easiest way to make a draft model look good.
    """
    scored = pj.target(train)
    if not scored.filter(pl.col("finish_pos_rank") == bk.MISSED_SEASON_RANK).height:
        pytest.skip("no missed seasons in this window")

    # Within season and position, because that is the group the percentile is
    # taken over. Pooling them compares a percentile against a different scale.
    checked = 0
    for (season, position), grp in scored.group_by(["label_season", "position"]):
        missed = grp.filter(pl.col("finish_pos_rank") == bk.MISSED_SEASON_RANK)
        played = grp.filter(pl.col("finish_pos_rank") < bk.MISSED_SEASON_RANK)
        if not missed.height or not played.height:
            continue
        checked += 1
        assert missed.get_column("finish_pct").max() <= played.get_column(
            "finish_pct"
        ).min() + 1e-9, f"{season} {position}: a missed season outranked a played one"
    assert checked > 0


def test_every_prediction_is_out_of_sample(preds: pl.DataFrame, train: pl.DataFrame) -> None:
    """Season-forward folds only: the first two label seasons are never tested on."""
    seasons = sorted(train.get_column("label_season").unique().to_list())
    tested = set(preds.get_column("label_season").unique().to_list())
    assert tested == set(seasons[2:]), "a fold tested on a season it could have trained on"


def test_skill_is_measured_against_price_not_against_nothing(preds: pl.DataFrame) -> None:
    """Predicting the finish is easy; beating ADP is the question.

    Usage persists, so any sane model ranks next season at roughly 0.5. Reporting
    that number alone would be a claim of skill the data does not support, which
    is why `skill()` always carries the price-only column beside it.
    """
    table = pj.skill(preds, n_boot=200)
    row = table.filter(pl.col("scope") == "all")
    assert row.height == 1
    assert row.get_column("spearman")[0] > 0.3, "the model cannot even rank the season"
    assert row.get_column("spearman_adp_only")[0] > 0.3, "the price baseline broke"
    assert "delta" in table.columns


def test_the_delta_interval_brackets_the_point_estimate(preds: pl.DataFrame) -> None:
    """A bootstrap interval that does not contain its own estimate is a bug."""
    table = pj.skill(preds, n_boot=400)
    for row in table.filter(~pl.col("scope").str.starts_with("test")).iter_rows(named=True):
        if np.isnan(row["delta_lo"]):
            continue
        assert row["delta_lo"] <= row["delta"] <= row["delta_hi"], (
            f"{row['scope']}: delta {row['delta']} outside [{row['delta_lo']}, {row['delta_hi']}]"
        )


def test_shuffled_target_scores_like_a_coin_flip(train: pl.DataFrame) -> None:
    """Permute the outcome and the model must lose all rank correlation.

    A sub-chance result usually means a sign error somewhere. This is the check
    that separates "no signal" from "signal, inverted".
    """
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(3):
        # Permute the *source* column. `fit_predict` derives `finish_pct` from
        # `finish_pos_rank` itself, so shuffling the derived column is silently
        # undone and the test measures the real model.
        shuffled = train.with_columns(
            pl.Series(
                "finish_pos_rank",
                rng.permutation(train.get_column("finish_pos_rank").to_numpy()),
            )
        )
        p = pj.fit_predict(shuffled)
        if not p.height:
            continue
        scores.append(
            abs(pj._spearman(p.get_column("pred").to_numpy(), p.get_column("finish_pct").to_numpy()))
        )
    if not scores:
        pytest.skip("nothing fit")
    assert max(scores) < 0.2, f"permuted target still ranks at {max(scores):.3f}"


def test_coefficients_carry_their_sample(train: pl.DataFrame) -> None:
    """In-sample and labelled as such — the out-of-sample number is `skill`."""
    coefs = pj.coefficients(train)
    if not coefs.height:
        pytest.skip("nothing fit")
    assert set(coefs.columns) >= {"position", "feature", "coefficient", "n"}
    # Price must enter negatively everywhere: a bigger positional ADP rank is a
    # cheaper player, and cheaper players finish worse on average.
    price = coefs.filter(pl.col("feature") == "adp_pos_rank")
    assert (price.get_column("coefficient") < 0).all(), (
        "ADP rank stopped predicting finish downward — check the target's direction"
    )
