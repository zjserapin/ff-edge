"""The quality space: how distance is weighted, and what a block restricts it to.

`quality_score` has always been a stability-weighted mean of the standardized
features — a metric that barely repeats contributes almost nothing. Every
*distance* built from the identical matrix was a plain Euclidean norm until
2026-08-14, giving each column an equal vote. So the score and the comparables
disagreed about what mattered while claiming to describe the same space.

That gap is not academic in this data. Running back quality features span `tprr`
at 0.402 down to `ryoe_per_att` at 0.202, a two-fold spread, so two backs could
read as similar largely on a fluky efficiency season neither would repeat.

The tests below pin three things: the weighting reduces exactly to the old
behaviour when weights are flat (so it is a reweighting, not a change of units),
a difference in a noisy metric costs less than the same difference in a sticky
one, and `restrict_to` really does confine the comparison to a block.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import archetypes as ar


# --- the weighting ----------------------------------------------------------


def test_flat_weights_reproduce_the_unweighted_distance_exactly() -> None:
    """The property that makes this a reweighting rather than new units.

    Distances are normalised to mean weight 1, so equal weights must return the
    plain Euclidean norm the old panel showed — not a rescaled version of it.
    """
    x = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    used = ["a", "b"]
    flat = {("RB", "a"): 0.5, ("RB", "b"): 0.5}

    got = ar._distance(x, used, "RB", flat, 0)

    assert np.allclose(got, np.linalg.norm(x - x[0], axis=1))
    assert got[1] == pytest.approx(5.0)


def test_a_difference_in_a_noisy_metric_costs_less() -> None:
    """The whole point. Same raw gap, different metric, different distance.

    One candidate differs from the anchor only on a metric that repeats at 0.8;
    the other only on one that repeats at 0.2. They are equally far apart in raw
    standardized units, and they must not be equally far apart here.
    """
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    used = ["sticky", "noisy"]
    weights = {("RB", "sticky"): 0.8, ("RB", "noisy"): 0.2}

    d = ar._distance(x, used, "RB", weights, 0)

    assert d[1] > d[2], "the noisy metric was not discounted"
    # mean weight 0.5, so sticky scales by 1.6 and noisy by 0.4.
    assert d[1] == pytest.approx(np.sqrt(1.6))
    assert d[2] == pytest.approx(np.sqrt(0.4))


def test_missing_weights_fall_back_to_unweighted() -> None:
    """Matches `_weighted`. An unknown position must not zero every distance.

    Without the guard `w.sum()` is 0, the division is 0/0, and every player comes
    back nan-or-zero distance from everyone — a comparables table that looks
    computed and is empty of information.
    """
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    used = ["a", "b"]

    for weights in (None, {}, {("WR", "a"): 0.5}):
        got = ar._distance(x, used, "RB", weights, 0)
        assert np.isfinite(got).all()
        assert got[1] == pytest.approx(5.0)


def test_a_zero_weight_metric_is_ignored_entirely() -> None:
    x = np.array([[0.0, 0.0], [0.0, 9.0]])
    used = ["counts", "ignored"]
    weights = {("RB", "counts"): 0.4, ("RB", "ignored"): 0.0}

    d = ar._distance(x, used, "RB", weights, 0)

    assert d[1] == pytest.approx(0.0)


# --- restricting to a block -------------------------------------------------


@pytest.fixture(scope="module")
def live():
    """Real scores and features, or a skip. These need a hydrated cache."""
    from src import features as ft

    feats = ft.build()
    if not feats.height:
        pytest.skip("no features built")
    season = int(feats.get_column("season").max())
    scored = ar.scores(season, min_games=8, df=feats)
    if scored.height < 12:
        pytest.skip("not enough scored players")
    return feats, scored, season


def test_restrict_to_confines_the_comparison_to_the_block(live) -> None:
    """The board's blocks are the point of the parameter.

    "Who does this back resemble among sixty backs" and "which of these nine is
    he actually like" are different questions, and only the second is asked at a
    pick.
    """
    import polars as pl

    feats, scored, season = live
    pool = scored.filter(pl.col("position") == "RB").head(6)
    ids_in = pool.get_column("player_id").to_list()
    anchor = ids_in[0]

    got = ar.neighbors(
        anchor, scored, feats, n=20, season=season, restrict_to=ids_in
    )

    assert got.height <= len(ids_in) - 1
    assert set(got.get_column("player_id")).issubset(set(ids_in) - {anchor})


def test_a_block_of_one_returns_nothing_rather_than_raising(live) -> None:
    """A block with a single player is common and is not an error."""
    import polars as pl

    feats, scored, season = live
    anchor = scored.filter(pl.col("position") == "RB").get_column("player_id")[0]

    got = ar.neighbors(
        anchor, scored, feats, n=8, season=season, restrict_to=[anchor]
    )

    assert got.height == 0


def test_the_anchor_never_appears_among_its_own_comparables(live) -> None:
    import polars as pl

    feats, scored, season = live
    pool = scored.filter(pl.col("position") == "WR").head(8)
    ids_in = pool.get_column("player_id").to_list()
    anchor = ids_in[0]

    got = ar.neighbors(
        anchor, scored, feats, n=20, season=season, restrict_to=ids_in
    )

    assert anchor not in got.get_column("player_id").to_list()
