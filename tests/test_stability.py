"""Year-over-year persistence — the gate every feature has to clear."""

from __future__ import annotations

import polars as pl
import pytest

from src import archetypes as ar
from src import features as ft
from src import stability as st


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    df = st.year_over_year()
    if not df.height:
        pytest.skip("cold cache")
    return df


def test_correlations_are_correlations(table: pl.DataFrame) -> None:
    values = table.get_column("r_yoy")
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert (table.get_column("n_pairs") >= st.MIN_PAIRS).all()


def test_opportunity_persists_better_than_quality(table: pl.DataFrame) -> None:
    """The premise the whole project rests on, asserted rather than assumed.

    Volume and role carry forward; per-touch efficiency largely does not. If this
    ever flips, the three-axis split in `features.py` is wrong and the valuation
    board is ranking noise.
    """
    summary = st.axis_summary(table)
    for position in ("QB", "RB", "WR", "TE"):
        rows = summary.filter(pl.col("position") == position)
        opp = rows.filter(pl.col("axis") == "opportunity")
        qual = rows.filter(pl.col("axis") == "quality")
        if not opp.height or not qual.height:
            continue
        assert opp.get_column("mean_r")[0] > qual.get_column("mean_r")[0], (
            f"{position}: quality now persists better than opportunity"
        )


def test_shipped_features_clear_the_noise_floor(table: pl.DataFrame) -> None:
    """Nothing in a live feature set may be below the floor that removed the rest.

    Six columns were dropped for failing this — `contested_catch_rate` at 0.061,
    `drop_rate` at 0.096, `catch_rate` at running back. The test is what stops
    one drifting back in.
    """
    noisy = {
        (row["position"], row["metric"])
        for row in st.noisy_features(table).iter_rows(named=True)
    }
    for position in ("QB", "RB", "WR", "TE"):
        for col in ft.quality_features(position) + ft.opportunity_features(position):
            assert (position, col) not in noisy, (
                f"{col} is below the noise floor at {position} and is still shipping"
            )


def test_the_dropped_features_would_still_fail() -> None:
    """The casualties, pinned. If one becomes usable, this fails and says so."""
    dropped = {
        "WR": ["drop_rate", "contested_catch_rate"],
        "TE": ["drop_rate", "catch_rate", "contested_catch_rate"],
        "RB": ["catch_rate", "ypt"],
    }
    table = st.year_over_year(columns=dropped, positions=tuple(dropped))
    if not table.height:
        pytest.skip("cold cache")
    assert (table.get_column("r_yoy") < st.NOISE_FLOOR).all(), (
        "a dropped feature now clears the floor — worth reinstating"
    )


def test_weights_are_non_negative_and_cover_the_feature_sets() -> None:
    """`_weighted` divides by the weight sum, so a negative weight would invert it."""
    weights = ar.stability_weights()
    if not weights:
        pytest.skip("cold cache")
    assert all(w >= 0 for w in weights.values())
    for position in ("WR", "RB", "TE"):
        known = [c for c in ft.quality_features(position) if (position, c) in weights]
        assert len(known) >= 4, f"{position} has almost no weighted quality features"


def test_weighting_beats_the_flat_mean_on_next_season() -> None:
    """The reason the score changed shape, measured rather than asserted.

    Weighting each standardized feature by how well it repeats moved the quality
    score's rank correlation with *next* season's points from 0.464 to 0.502 at
    receiver. Uses no outcome data to build the weights, so this is a check on
    the construction, not a fitted result.
    """
    import numpy as np

    feats = ft.build()
    if not feats.height:
        pytest.skip("cold cache")
    weights = ar.stability_weights(feats)
    if not weights:
        pytest.skip("no weights")

    pool = feats.filter((pl.col("position") == "WR") & (pl.col("games") >= 8))
    parts = []
    for season in sorted(pool.get_column("season").unique().to_list()):
        grp = pool.filter(pl.col("season") == season)
        if grp.height < 20:
            continue
        x, used = ar._matrix(grp, ft.quality_features("WR"))
        if not used:
            continue
        parts.append(
            grp.select("season", "player_id", "ppg").with_columns(
                pl.Series("flat", x.mean(axis=1)),
                pl.Series("weighted", ar._weighted(x, used, "WR", weights)),
            )
        )
    if not parts:
        pytest.skip("no seasons")

    scored = pl.concat(parts)
    nxt = scored.select(
        "player_id", (pl.col("season") - 1).alias("season"), pl.col("ppg").alias("next_ppg")
    )
    paired = scored.join(nxt, on=["season", "player_id"], how="inner")
    if paired.height < 100:
        pytest.skip("too few pairs")

    def rank_corr(col: str) -> float:
        return float(
            paired.select(pl.corr(pl.col(col).rank(), pl.col("next_ppg").rank())).item()
        )

    assert rank_corr("weighted") > rank_corr("flat"), (
        "stability weighting no longer beats the flat mean — revisit `_weighted`"
    )
    assert not np.isnan(rank_corr("weighted"))


# --- sticky feature selection -----------------------------------------------


def _stability_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "position": ["WR"] * 5 + ["RB"],
            "axis": ["outcome", "opportunity", "quality", "opportunity", "quality", "quality"],
            "metric": ["ppg", "target_share", "adot", "route_share", "yprr", "ypc"],
            "r_yoy": [0.90, 0.66, 0.64, 0.30, 0.50, 0.55],
            "n_pairs": [500] * 6,
            "verdict": ["sticky", "sticky", "sticky", "usable", "sticky", "sticky"],
        }
    )


def test_sticky_features_excludes_outcome_columns() -> None:
    """`ppg` against draft price is close to price against itself.

    Draft price is built almost entirely on last season's production, so an
    outcome column on the y-axis draws a tight diagonal that looks like a finding
    and is a tautology. It has the highest persistence in the table, so it would
    lead the panel if it were not filtered.
    """
    got = st.sticky_features(_stability_table(), "WR")
    assert "ppg" not in got.get_column("metric").to_list()
    assert got.get_column("axis").unique().to_list() != ["outcome"]

    kept = st.sticky_features(_stability_table(), "WR", include_outcomes=True)
    assert "ppg" in kept.get_column("metric").to_list()


def test_sticky_features_drops_anything_below_the_sticky_verdict() -> None:
    got = st.sticky_features(_stability_table(), "WR")
    assert "route_share" not in got.get_column("metric").to_list()


def test_sticky_features_orders_by_persistence_and_caps() -> None:
    got = st.sticky_features(_stability_table(), "WR", top_n=2)
    assert got.get_column("metric").to_list() == ["target_share", "adot"]
    assert got.get_column("r_yoy").to_list() == sorted(
        got.get_column("r_yoy").to_list(), reverse=True
    )


def test_sticky_features_is_per_position() -> None:
    got = st.sticky_features(_stability_table(), "RB")
    assert got.get_column("metric").to_list() == ["ypc"]
