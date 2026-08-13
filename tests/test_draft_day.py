"""The draft-day dry run: drive the tab, do not merely render it.

`HANDOFF.md` asked for this as the second priority before 2026-08-22, and
described exactly why: the Draft Day tab had "never been driven under time
pressure, only rendered", and *"that is a different test from renders without
exceptions, and it is the only one that matters on the 22nd."*

That turned out to be right in a way nobody expected. Rendering the app once
passed. Rendering it nine times, changing the pick each time the way a real
draft does, killed the interpreter — a pyarrow 25.0.0 segfault on the
pandas/Arrow round trip behind `st.dataframe`, with no traceback and no error
page. 270 unit tests passed against that build. See
`test_display.test_pyarrow_is_not_the_release_that_segfaults`.

So this file exists to exercise the one thing unit tests structurally cannot:
**repetition**. The crash needed an allocation sequence, not an input, and the
only way to produce an allocation sequence is to actually drive the widget.

**A failure here may not look like a test failure.** If the crash returns,
pytest dies with SIGSEGV and reports nothing — exit code 139, no summary. Treat
a vanished test run as a red result, not a flake.

Needs a league. Skips cleanly without one, like `test_board.py`.
"""

from __future__ import annotations

import pytest

from src.config import SLEEPER_USERNAME


def _app():
    """A run app, or a skip. Import is inside the helper — `AppTest` is slow."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=300).run()
    if at.exception:
        pytest.fail(f"app raised on first render: {at.exception[0].value}")
    return at


@pytest.fixture(scope="module")
def picker():
    """The Draft Day pick selector, or a skip if there is no league.

    The selector only exists when `board.picks` resolves an owner, which needs
    both `FF_EDGE_LEAGUE_ID` and `FF_EDGE_SLEEPER_USER`. Without the handle the
    tab falls back to a free-text pick number — a legitimate state for a cold
    clone, and not the thing under test.
    """
    if SLEEPER_USERNAME == "CHANGE_ME":
        pytest.skip("no handle configured (FF_EDGE_SLEEPER_USER unset)")
    at = _app()
    if not [s for s in at.selectbox if s.key == "draft_pick_no"]:
        pytest.skip("no pick list — needs FF_EDGE_LEAGUE_ID and a draft order")
    return at


def test_every_pick_you_own_can_be_selected_in_order(picker) -> None:
    """Walk the whole pick list forward, which is how a draft is actually read.

    Forward specifically. The crash this test was written for fired on the
    ninth *forward* selection and never once in reverse, because the heap
    arrives in a different state. A test that walked the list in any convenient
    order would have been green against a build that could not survive a draft.
    """
    at = picker
    options = at.selectbox(key="draft_pick_no").options
    assert len(options) > 1, "a pick list of one cannot exercise repetition"

    for pick in options:
        at = at.selectbox(key="draft_pick_no").select(pick).run()
        assert not at.exception, f"pick {pick} raised: {at.exception[0].value}"


def test_the_board_answers_fast_enough_to_use_on_the_clock(picker) -> None:
    """A pick change has to resolve inside the time a real pick allows.

    The budget is deliberately loose. A Sleeper clock is 90 seconds and the
    handoff's standard was "the ten seconds a real pick allows", so 10s is the
    number that matters; anything near it means a cache boundary moved and the
    tab started recomputing the board on every keystroke. Warm reruns measure
    around 0.3s, so this fails long before Zach would notice a lag.
    """
    import time

    at = picker
    options = at.selectbox(key="draft_pick_no").options
    slowest, worst = 0.0, None
    for pick in options:
        start = time.perf_counter()
        at = at.selectbox(key="draft_pick_no").select(pick).run()
        elapsed = time.perf_counter() - start
        if elapsed > slowest:
            slowest, worst = elapsed, pick

    assert slowest < 10.0, f"pick {worst} took {slowest:.1f}s to resolve"
