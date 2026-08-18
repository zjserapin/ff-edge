"""The four screens, and the market they are read in.

`peek` and `adp` were finished, tested, and imported by nothing — `app.py` pulled
in sixteen modules and neither was among them. Surfacing them is display work,
so most of what can go wrong here is not in the rendering:

**The season was a literal.** All four `peek` entry points defaulted to
`season=2025` while `config.SEASON` said 2026. Correct today, silently wrong the
first August nobody re-reads them, and invisible either way because a screen
against the wrong year returns a full, plausible table.

**The market is not the default.** `adp.movement` defaults to
`config.ADP_SCORING`/`ADP_TEAMS` — ppr/12 — which prices the 828 league, not the
Shiva Bowl's 2qb/10. Reading movement in the wrong market renders a well-formed
table of somebody else's drift. This is the `profiles.py` lesson: a roster format
and the market that prices it travel together or they disagree in silence.
"""

from __future__ import annotations

import inspect

import pytest

from src import adp as adp_mod
from src import config
from src import peek
from src import profiles as pf


# --- the season is derived, not typed ---------------------------------------


def test_peek_reads_the_season_from_config() -> None:
    """`LAST_PLAYED` trails `config.SEASON`, so rolling the year moves it."""
    assert peek.LAST_PLAYED == config.SEASON - 1


@pytest.mark.parametrize(
    "fn", [peek.regression_candidates, peek.snap_trend, peek.usage_leaders]
)
def test_no_screen_defaults_to_a_hardcoded_season(fn) -> None:
    """The defect this file was written for, pinned per function.

    Asserted against `LAST_PLAYED` rather than a number, so the test cannot rot
    into the same literal it exists to forbid.
    """
    default = inspect.signature(fn).parameters["season"].default
    assert default == peek.LAST_PLAYED
    assert default == config.SEASON - 1


# --- the market -------------------------------------------------------------


def test_the_profiles_market_is_not_the_module_default() -> None:
    """If these ever coincide, the test below stops proving anything.

    Not a preference — it is the precondition that makes passing the market
    explicitly a load-bearing act rather than a no-op.
    """
    profile = pf.resolve("shiva_bowl")
    assert (profile.adp_scoring, profile.adp_teams) != (
        config.ADP_SCORING,
        config.ADP_TEAMS,
    ), "shiva_bowl must differ from the module default for this to be a real test"


def test_movement_reads_a_different_market_when_given_one() -> None:
    """Behavioural rather than mocked: the two markets disagree on real data.

    Asserting on the frames instead of on a patched path means this still fails
    if `movement` ever ignores its arguments *and* keeps the same file-naming
    scheme — which a mock of the path would not catch.
    """
    profile = pf.resolve("shiva_bowl")
    league = adp_mod.movement(scoring=profile.adp_scoring, teams=profile.adp_teams)
    default = adp_mod.movement()

    if not league.height or not default.height:
        pytest.skip("need two snapshots in both markets")

    assert not league.equals(default), (
        "the profile's market and the module default returned identical frames; "
        "movement is ignoring the market it was handed"
    )


def test_movement_is_empty_rather_than_raising_without_history() -> None:
    """One snapshot is the normal state early, and it must not look like a crash."""
    out = adp_mod.movement(scoring="nonexistent", teams=99)
    assert out.is_empty()


# --- the screens return something shaped like a screen ----------------------


@pytest.mark.parametrize(
    "name, columns",
    [
        ("regression_candidates", {"pts_over_exp", "exp_pts", "act_pts"}),
        ("market_disagreement", {"ecr", "sd"}),
    ],
)
def test_screen_columns_are_stable(name, columns) -> None:
    """The app renders these by name, so a rename upstream is a blank table."""
    out = getattr(peek, name)()
    if not out.height:
        pytest.skip(f"{name}: nothing cached")
    assert columns <= set(out.columns)


def test_negative_points_over_expected_is_the_buy_low_end() -> None:
    """Direction check. The screen is worthless if the sign is read backwards.

    `pts_over_exp` is actual minus expected, so the buy-low candidates sort to
    the *front* ascending. The app's caption depends on that being true.
    """
    out = peek.regression_candidates()
    if out.height < 2:
        pytest.skip("nothing cached")
    ordered = out.sort("pts_over_exp", nulls_last=True).get_column("pts_over_exp")
    assert ordered[0] <= ordered[-1]
    assert ordered.min() < 0, "a screen with no negative side has nothing to buy"


def test_movement_direction_matches_its_sign() -> None:
    """Negative `adp_change` means he is going EARLIER.

    The column is signed the opposite way to how it reads in English, which is
    exactly the kind of thing the UI gets backwards — so the label that ships
    beside it is pinned here rather than trusted.
    """
    profile = pf.resolve("shiva_bowl")
    mv = adp_mod.movement(scoring=profile.adp_scoring, teams=profile.adp_teams)
    if not mv.height or "direction" not in mv.columns:
        pytest.skip("no movement history yet")

    risers = mv.filter(adp_mod.pl.col("direction") == "rising")
    if risers.height:
        assert risers.get_column("adp_change").max() <= 0, (
            "a 'rising' player whose ADP number went up means the label and the "
            "sign disagree"
        )


# --- driving the tab --------------------------------------------------------
#
# Ported from `test_draft_day.py` rather than re-derived, for the reason that
# file documents: rendering once passed and rendering nine times segfaulted, and
# the same fifteen steps in reverse never failed. So these walk forward, the
# whole range. **A failure here may not look like one** — pytest exits 139 with
# no summary, and a vanished run is a red result.


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=600).run()
    if at.exception:
        pytest.fail(f"app raised on first render: {at.exception[0].value}")
    return at


@pytest.fixture(scope="module")
def screens():
    """The screens expander, or a skip when there is nothing cached to screen."""
    at = _app()
    if not [s for s in at.selectbox if s.key == "screen_snap_pos"]:
        pytest.skip("screens did not render — cold cache")
    return at


def test_every_screen_heading_rendered(screens) -> None:
    """All four are present, so a silently-empty section cannot pass as working."""
    headings = {m.value for m in screens.markdown}
    for expected in (
        "**Points over expected — the buy-low screen**",
        "**ADP movement — camp news made numeric**",
        "**Where the experts disagree**",
        "**Snap trend — role change before the box score**",
    ):
        assert expected in headings, f"missing screen: {expected}"


def test_the_movement_window_can_be_driven_the_whole_way_up(screens) -> None:
    """Repetition, forward, on the control that re-reads a parquet each step."""
    at = screens
    slider = at.slider(key="screen_move_days")
    values = list(range(int(slider.min), int(slider.max) + 1, int(slider.step)))
    assert len(values) > 1, "a single-step slider cannot exercise repetition"

    for value in values:
        at = at.slider(key="screen_move_days").set_value(value).run()
        assert not at.exception, f"window {value} raised: {at.exception[0].value}"


def test_every_snap_position_renders(screens) -> None:
    """Each option re-runs a different nflverse slice; one empty must not raise."""
    at = screens
    for position in ("RB", "WR", "TE", "QB"):
        at = at.selectbox(key="screen_snap_pos").set_value(position).run()
        assert not at.exception, f"{position} raised: {at.exception[0].value}"


def test_filtering_the_buy_low_screen_to_each_position_renders(screens) -> None:
    at = screens
    options = at.multiselect(key="screen_reg_pos").options
    if not options:
        pytest.skip("no positions to filter")
    for position in options:
        at = at.multiselect(key="screen_reg_pos").set_value([position]).run()
        assert not at.exception, f"{position} raised: {at.exception[0].value}"


# --- the Footballers disagreement panel -------------------------------------


def test_the_disagreement_panel_and_usage_toggle_render(screens) -> None:
    """Both of the last two cheap items reached the page.

    Cheap to assert and worth asserting: a section that silently fails to render
    looks identical to one that was never added.
    """
    at = screens
    assert [c for c in at.checkbox if c.key == "big_board_usage"], "usage toggle missing"
    assert [s for s in at.slider if s.key == "ffb_cmp_top"], "disagreement panel missing"


def test_the_usage_toggle_can_be_driven_both_ways(screens) -> None:
    """On, off, on — the column set changes each time, which is where it breaks."""
    at = screens
    for value in (True, False, True):
        at = at.checkbox(key="big_board_usage").set_value(value).run()
        assert not at.exception, f"usage={value} raised: {at.exception[0].value}"


def test_the_disagreement_cut_can_be_driven_the_whole_way_up(screens) -> None:
    at = screens
    slider = at.slider(key="ffb_cmp_top")
    values = list(range(int(slider.min), int(slider.max) + 1, int(slider.step)))
    assert len(values) > 1, "a single-step slider cannot exercise repetition"

    for value in values:
        at = at.slider(key="ffb_cmp_top").set_value(value).run()
        assert not at.exception, f"top {value} raised: {at.exception[0].value}"
