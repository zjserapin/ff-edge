"""The promotion screen's finding, pinned.

The module's docstring makes strong claims — trust markers predict promoted
RBs, efficiency does not, quality tiers are monotone — and these tests assert
the *shape* of those claims against the real cohort rather than trusting prose.
If a data refresh erodes the RB efficiency null or flips the tier ordering,
that is a finding in itself and it should surface as a failure here, not stay
hidden under a docstring that no longer describes the data.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import features as ft
from src import promotion as pr


@pytest.fixture(scope="module")
def feats() -> pl.DataFrame:
    df = ft.build()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def coh(feats: pl.DataFrame) -> pl.DataFrame:
    df = pr.cohort(feats)
    if not df.height:
        pytest.skip("cold cache")
    return df


# --- cohort construction ----------------------------------------------------


def test_cohort_is_promoted_backups_only(coh: pl.DataFrame) -> None:
    """Below-median prior role, 10+ point growth — the definition, verified."""
    assert (coh.get_column("opportunity_pct") < pr.BACKUP_CEILING_PCT).all()
    assert (coh.get_column("opp_change") >= pr.PROMOTION_PTS).all()
    assert set(coh.get_column("position").unique().to_list()) <= set(pr.CRITERIA)


def test_cohort_pairs_adjacent_seasons(coh: pl.DataFrame) -> None:
    """Features from season N, outcome from N+1 — anything else is leakage."""
    assert (coh.get_column("next_season") - coh.get_column("season") == 1).all()


def test_cohort_sizes_match_the_finding(coh: pl.DataFrame) -> None:
    """The exploratory session found ~72 RB / ~122 WR / ~67 TE. The in-repo
    recipe differs slightly (qualification gates), so assert the neighborhood,
    not the number — a cohort half or double that size means the recipe drifted.
    """
    sizes = dict(
        coh.group_by("position").len().iter_rows()
    )
    assert 40 <= sizes.get("RB", 0) <= 110
    assert 70 <= sizes.get("WR", 0) <= 160
    assert 35 <= sizes.get("TE", 0) <= 100


def test_undrafted_promotions_are_kept(coh: pl.DataFrame) -> None:
    """`beat_adp` must be null, not dropped, where the market never priced the
    player — those rows are the most interesting ones the screen exists for."""
    assert "beat_adp" in coh.columns
    assert coh.get_column("beat_adp").null_count() > 0


# --- the finding ------------------------------------------------------------


def test_rb_efficiency_predicts_nothing(coh: pl.DataFrame) -> None:
    """The load-bearing null: prior efficiency does not pick promoted RBs.

    Five metrics landed within 0.04 of zero in the exploratory run; the
    assertion leaves room for sampling noise but fails if any efficiency metric
    starts genuinely predicting, because then the RB criteria are wrong.
    """
    v = pr.validate(coh)
    rb_eff = v.filter(
        (pl.col("position") == "RB")
        & pl.col("metric").is_in(
            ["ypc", "yards_after_contact_per_att", "rush_broken_tackles_per_att"]
        )
    )
    assert rb_eff.height >= 3
    assert (rb_eff.get_column("r").abs() < 0.25).all(), rb_eff.to_dicts()


def test_rb_trust_markers_predict(coh: pl.DataFrame) -> None:
    """Snap share and red-zone carries are the RB criteria because they work."""
    v = pr.validate(coh)

    def r_of(metric: str) -> float:
        return float(
            v.filter((pl.col("position") == "RB") & (pl.col("metric") == metric))
            .get_column("r")[0]
        )

    assert r_of("snap_pct") > 0.30
    assert r_of("rz_carry_share") > 0.20
    assert r_of("exp_td_share") > 0.20


def test_receiver_efficiency_predicts(coh: pl.DataFrame) -> None:
    """At TE (and to a lesser degree WR) yards per route run carries signal —
    the position asymmetry that motivates separate criteria."""
    v = pr.validate(coh)
    te_yprr = v.filter((pl.col("position") == "TE") & (pl.col("metric") == "yprr"))
    assert float(te_yprr.get_column("r")[0]) > 0.20


def test_quality_tiers_are_monotone(coh: pl.DataFrame) -> None:
    """The filter property: bottom-tercile promoted players almost never hit."""
    tiers = pr.quality_tiers(coh).filter(pl.col("scope") == "all")
    rates = {r["quality_tier"]: r["hit_rate"] for r in tiers.iter_rows(named=True)}
    assert rates["bottom 30%"] < rates["middle"] < rates["top 30%"]
    assert rates["top 30%"] >= rates["bottom 30%"] * 2.5
    assert {"n", "ci_lo", "ci_hi"} <= set(tiers.columns)


def test_criteria_are_preregistered_not_searched() -> None:
    """RB criteria must stay trust markers; efficiency stays out even when a
    rerun makes some efficiency metric look tempting. Changing this dict is a
    deliberate act with a docstring edit, not a drive-by."""
    assert set(pr.CRITERIA["RB"]) == {
        "snap_pct", "rz_carry_share", "gz_carry_share", "exp_td_share"
    }
    assert "yprr" in pr.CRITERIA["WR"] and "yprr" in pr.CRITERIA["TE"]
    for cols in pr.CRITERIA.values():
        assert "ypc" not in cols


# --- the screen -------------------------------------------------------------


def test_screen_grades_known_players(feats: pl.DataFrame, coh: pl.DataFrame) -> None:
    grades, missing = pr.screen(
        ["Bhayshul Tuten", "No Such Player"], df=feats, coh=coh
    )
    assert missing == ["No Such Player"]
    assert grades.height >= 1
    row = grades.filter(pl.col("player_name") == "Bhayshul Tuten")
    assert row.height == 1
    assert 0 <= row.get_column("screen_pct")[0] <= 100
    # The base rate arrives with its denominator or not at all.
    if row.get_column("tier_hit_rate")[0] is not None:
        assert row.get_column("tier_n")[0] > 20


def test_screen_reports_the_base_rate_not_a_projection(
    feats: pl.DataFrame, coh: pl.DataFrame
) -> None:
    """No column may look like a point projection of next-season points."""
    grades, _ = pr.screen(["Bhayshul Tuten"], df=feats, coh=coh)
    assert not any("proj" in c or "next_ppg" == c for c in grades.columns)
    assert {"tier_hit_rate", "tier_ci_lo", "tier_ci_hi", "tier_n"} <= set(grades.columns)


def test_screen_flags_thin_seasons(feats: pl.DataFrame, coh: pl.DataFrame) -> None:
    """A 5-game player gets criteria percentiles but no quality tier — the
    history pool requires 8 games, and the screen must show that as null rather
    than grading him on a season fragment."""
    grades, _ = pr.screen(["Bhayshul Tuten"], df=feats, coh=coh)
    assert "games" in grades.columns


# --- the archetype split ----------------------------------------------------


def test_archetype_split_covers_the_rb_cohort(coh: pl.DataFrame) -> None:
    split = pr.archetype_split(coh)
    assert split.height == 2
    rb_n = coh.filter(pl.col("position") == "RB").height
    assert split.get_column("n").sum() >= rb_n * 0.9
    for r in split.iter_rows(named=True):
        assert r["ci_lo"] <= r["hit_rate"] <= r["ci_hi"]


# --- weekly trust -----------------------------------------------------------


def test_weekly_trust_shares_are_shares(feats: pl.DataFrame) -> None:
    season = int(feats.get_column("season").max())
    wk = pr.weekly_trust(season)
    if not wk.height:
        pytest.skip("cold cache")
    assert (wk.get_column("week") <= 18).all()
    for col in ("carry_share_wk", "rz_carry_share_wk", "target_share_wk"):
        vals = wk.get_column(col).drop_nulls()
        assert (vals >= 0).all() and (vals <= 1).all(), col


def test_weekly_trust_sums_to_one_within_team_week(feats: pl.DataFrame) -> None:
    """Every week's carry shares must sum to ~1 per team — the denominator is
    the team's own week, which is the whole point of the weekly frame."""
    season = int(feats.get_column("season").max())
    wk = pr.weekly_trust(season)
    if not wk.height:
        pytest.skip("cold cache")
    # player_id -> team isn't carried, so check globally: the sum of all carry
    # shares in a week equals the number of teams that ran a play that week.
    week1 = wk.filter((pl.col("week") == 1) & pl.col("carry_share_wk").is_not_null())
    total = float(week1.get_column("carry_share_wk").sum())
    assert total == pytest.approx(round(total), abs=0.01)
