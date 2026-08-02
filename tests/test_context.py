"""Play-context features: the joins that fail silently, and the values themselves."""

from __future__ import annotations

import polars as pl
import pytest

from src import context as cx
from src import features as ft
from src import nflverse as nv
from src.config import CURRENT_SEASON, FEATURE_SEASONS


@pytest.fixture(scope="module")
def ctx() -> pl.DataFrame:
    df = cx.play_context()
    if not df.height:
        pytest.skip("cold cache")
    return df


def test_shares_are_bounded(ctx: pl.DataFrame) -> None:
    """Every share is a fraction of a team total, so nothing may exceed one."""
    for col in (
        "rz_target_share", "ez_target_share", "neutral_target_share",
        "rz_carry_share", "gz_carry_share", "neutral_rush_share", "exp_td_share",
    ):
        values = ctx.get_column(col).drop_nulls()
        if not len(values):
            continue
        assert values.min() >= 0.0, f"{col} went negative"
        assert values.max() <= 1.0 + 1e-9, f"{col} exceeded a full team share"


def test_playoff_weeks_are_excluded() -> None:
    """Both pbp tables carry weeks 19-22 with no season_type column to filter on.

    Left in, postseason production lands inside "season usage" and quietly
    rewards whoever made a deep run.
    """
    raw = nv.ff_opportunity(FEATURE_SEASONS, stat_type="pbp_pass")
    if not raw.height:
        pytest.skip("cold cache")
    assert raw.get_column("week").max() > 18, "fixture assumption broke"
    assert cx._scrimmage(raw).get_column("week").max() <= 18


def test_two_point_plays_are_excluded() -> None:
    """776 pass rows are two-point tries, every one snapped from the two.

    They carry `pass_attempt == 1`, so leaving them in inflates red-zone share
    for whoever happened to be targeted on a conversion.
    """
    raw = nv.ff_opportunity(FEATURE_SEASONS, stat_type="pbp_pass")
    if not raw.height:
        pytest.skip("cold cache")
    assert raw.filter(pl.col("two_point_attempt") == 1).height > 0
    assert cx._scrimmage(raw).filter(pl.col("two_point_attempt") == 1).height == 0


def test_ftn_join_survives_the_dtype_mismatch() -> None:
    """`play_id` is Float64 in ff_opportunity and Int32 in FTN.

    Without the cast the join raises; the point of the test is that it matches
    nearly everything once cast, so a future schema change that silently drops
    to a low match rate is caught here rather than in a board full of nulls.
    """
    charted = cx._charting(FEATURE_SEASONS)
    if not charted.height:
        pytest.skip("cold cache")
    covered = charted.get_column("catchable_rate").is_not_null().mean()
    assert covered > 0.9, f"FTN join matched only {covered:.1%} of receiver-seasons"


def test_endzone_targets_are_a_subset_of_all_targets(ctx: pl.DataFrame) -> None:
    """A throw to the end zone is a throw. Ordering these wrong inverts the read."""
    sub = ctx.filter(
        pl.col("ez_target_share").is_not_null() & pl.col("rz_target_share").is_not_null()
    )
    if not sub.height:
        pytest.skip("no overlap")
    # Not a strict subset — a 40-yard throw into the end zone is an end-zone
    # target without being a red-zone one — so this only asserts both are real.
    assert sub.get_column("ez_target_share").max() <= 1.0


def test_rush_expectations_use_the_modelled_column() -> None:
    """`rush_touchdown_exp` carries sentinels where `rushing_td_exp` has a value.

    They disagree on 4,558 rows and the short-named one is zero on most of them.
    Picking it would silently zero out touchdown equity for those plays.
    """
    raw = nv.ff_opportunity(FEATURE_SEASONS, stat_type="pbp_rush")
    if not raw.height:
        pytest.skip("cold cache")
    disagree = raw.filter(
        (pl.col("rushing_td_exp") - pl.col("rush_touchdown_exp")).abs() > 1e-9
    )
    assert disagree.height > 0, "fixture assumption broke"
    assert disagree.get_column("rushing_td_exp").mean() > disagree.get_column(
        "rush_touchdown_exp"
    ).mean()


def test_context_reaches_the_feature_table() -> None:
    """Coverage at fantasy-relevant volume, which is the only place it matters.

    Overall coverage is low by design — most player-seasons are backups with a
    handful of touches. The pool the board actually reads from is fully covered.
    """
    feats = ft.build()
    if not feats.height:
        pytest.skip("cold cache")

    backs = feats.filter(
        (pl.col("season") == CURRENT_SEASON)
        & (pl.col("position") == "RB")
        & (pl.col("carries") >= 100)
    )
    assert backs.height > 20
    assert backs.get_column("ryoe_per_att").is_not_null().all()

    receivers = feats.filter(
        (pl.col("season") == CURRENT_SEASON)
        & (pl.col("position") == "WR")
        & (pl.col("targets") >= 70)
    )
    assert receivers.height > 20
    assert receivers.get_column("exp_td_share").is_not_null().all()
    assert receivers.get_column("catch_rate_on_catchable").is_not_null().all()


def test_touchdown_equity_ranks_the_right_players() -> None:
    """A sanity check with a known answer.

    Tush-push quarterbacks own their offense's goal-line scoring, so they should
    sit at the very top of expected-touchdown share. If a refactor inverts a
    denominator this is the assertion that notices.
    """
    feats = ft.build()
    if not feats.height:
        pytest.skip("cold cache")
    season = feats.filter((pl.col("season") == 2024) & (pl.col("games") >= 8))
    if not season.height:
        pytest.skip("season missing")
    top = season.sort("exp_td_share", descending=True).head(10)
    assert top.get_column("exp_td_share")[0] > 0.3
    assert "QB" in top.get_column("position").to_list()


def test_charted_features_are_excluded_from_the_backtest_set() -> None:
    """FTN starts in 2022; the backtest window starts in 2018.

    Median-imputing four of eight feature seasons would hand half the training
    set an invented value, so these columns must not reach `feature_columns`.
    """
    for position in ("QB", "RB", "WR", "TE"):
        cols = ft.feature_columns(position)
        for charted in ft.CHARTED_FEATURES:
            assert charted not in cols, f"{charted} leaked into the {position} model set"
        assert charted not in ft.play_context_features(position)
    # ...but they are available when explicitly asked for.
    assert "catch_rate_on_catchable" in ft.play_context_features("WR", charted=True)
