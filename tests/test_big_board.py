"""The big board: the vegas join, the signal label, and the tab being driven.

Two different classes of failure live here and neither of them raises.

**The join fans out.** `attach_vegas` keys on a normalized name, and
`ids.normalize` strips generational suffixes — so Michael Pittman Jr. and Sr.
collapse to one key. Without the dedupe, a left join multiplies rows instead of
erroring; the props join went 145 -> 151 exactly this way. So the tests here
assert the row count rather than the values, because the row count is the thing
that silently changes.

**The label overstates coverage.** `vegas_gap` is null for roughly two thirds of
the board by construction, and a label that folded those into "quiet" would
report a board far better checked than it is. `signal` must stay null wherever
either input is missing.

The driven-tab test at the bottom is a port of `test_draft_day.py`, and it is
here for the reason that file documents: rendering once passed and rendering
nine times segfaulted. **A failure may not look like a failure** — pytest exits
139 with no summary. A vanished run is a red result.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import board as bd
from src.config import SLEEPER_USERNAME


def _players(rows: list[tuple[str, str, float | None]]) -> pl.DataFrame:
    """(name, position, value_gap) -> a board-shaped frame."""
    return pl.DataFrame(
        [{"name": n, "position": p, "value_gap": v} for n, p, v in rows],
        schema={"name": pl.Utf8, "position": pl.Utf8, "value_gap": pl.Float64},
    )


def _priced(rows: list[tuple[str, float | None]]) -> pl.DataFrame:
    """(name, vegas_gap) -> a frame shaped like valuation + against_price."""
    return pl.DataFrame(
        [
            {"name": n, "market": "rec_yds", "line": 900.0, "line_pct": 80.0,
             "vegas_gap": v}
            for n, v in rows
        ],
        schema={
            "name": pl.Utf8, "market": pl.Utf8, "line": pl.Float64,
            "line_pct": pl.Float64, "vegas_gap": pl.Float64,
        },
    )


# --- the join ---------------------------------------------------------------


def test_duplicate_names_do_not_fan_the_board_out() -> None:
    """The whole reason `attach_vegas` dedupes before joining.

    Two rows that normalize to the same key is not a hypothetical: suffix
    stripping puts father and son on one key, and defenders share names with
    skill players. A left join against an undeduped right side multiplies the
    left row silently.
    """
    players = _players([("Michael Pittman Jr.", "WR", 10.0)])
    priced = _priced([("Michael Pittman Jr.", 12.0), ("Michael Pittman Sr.", -40.0)])

    out = bd.attach_vegas(players, priced)

    assert out.height == 1, "the board grew — the dedupe before the join is gone"


def test_a_player_with_no_line_survives_the_join() -> None:
    """A null is "no line posted". It must never remove him from the board."""
    players = _players([("Priced Player", "WR", 5.0), ("Unpriced Player", "TE", 5.0)])
    priced = _priced([("Priced Player", 20.0)])

    out = bd.attach_vegas(players, priced)

    assert out.height == 2
    unpriced = out.filter(pl.col("name") == "Unpriced Player")
    assert unpriced.get_column("vegas_gap").item() is None


def test_an_unreachable_book_still_returns_a_board() -> None:
    """FanDuel being down costs the tab a column, not the page.

    The columns still have to exist and be typed, or every consumer downstream
    has to branch on their presence.
    """
    players = _players([("Someone", "WR", 5.0)])

    out = bd.attach_vegas(players, pl.DataFrame())

    assert out.height == 1
    for column in ("market", "line", "line_pct", "vegas_gap"):
        assert column in out.columns
    assert out.get_column("vegas_gap").dtype == pl.Float64


def test_the_vegas_gap_is_carried_not_recomputed() -> None:
    """Percentiles only subtract meaningfully over one population.

    `against_price` takes both sides over valuation's ~143 players. Re-deriving
    against this board's ~159 would shift the denominator under one side only
    and quietly change every number, so the value must arrive unchanged.
    """
    players = _players([("Someone", "WR", 5.0)])
    priced = _priced([("Someone", 17.5)])

    out = bd.attach_vegas(players, priced)

    assert out.get_column("vegas_gap").item() == 17.5


# --- the label --------------------------------------------------------------


def test_signal_stays_null_when_either_read_is_missing() -> None:
    """"No line posted" and "a line that agrees with ADP" are different facts.

    Folding the first into `quiet` would report a board far better checked than
    it is — with two thirds of it unpriced, that is not a rounding error.
    """
    players = _players([("No line", "WR", 40.0), ("No quality", "RB", None)])
    priced = _priced([("No line", None), ("No quality", 30.0)])

    out = bd.signal(bd.attach_vegas(players, priced))

    assert out.get_column("signal").null_count() == 2


def test_signal_names_agreement_and_opposition() -> None:
    """The four levels, one row each, at values either side of the thresholds."""
    players = _players(
        [
            ("Both up", "WR", 20.0), ("Both down", "WR", -20.0),
            ("Split", "WR", 20.0), ("Quiet", "WR", 2.0),
        ]
    )
    priced = _priced(
        [("Both up", 15.0), ("Both down", -15.0), ("Split", -15.0), ("Quiet", 1.0)]
    )

    out = bd.signal(bd.attach_vegas(players, priced))
    got = dict(zip(out.get_column("name"), out.get_column("signal")))

    assert got == {
        "Both up": "both up", "Both down": "both down",
        "Split": "split", "Quiet": "quiet",
    }


def test_opposition_is_split_in_both_directions() -> None:
    """A split is a split whichever read is the optimistic one.

    Worth its own test because the two branches are written separately in
    `signal`, and a copy-paste that repeated one condition would still produce a
    plausible-looking board.
    """
    players = _players([("Quality high", "WR", 30.0), ("Book high", "WR", -30.0)])
    priced = _priced([("Quality high", -20.0), ("Book high", 20.0)])

    out = bd.signal(bd.attach_vegas(players, priced))

    assert out.get_column("signal").to_list() == ["split", "split"]


def test_signal_survives_a_board_with_no_vegas_column_at_all() -> None:
    players = _players([("Someone", "WR", 5.0)])

    out = bd.signal(players)

    assert out.get_column("signal").item() is None


# --- the drop column, which is the shape next to PAR's level ----------------


def _par_board(rows: list[tuple[str, str, float | None]]) -> pl.DataFrame:
    """(name, position, par) -> a board-shaped frame."""
    return pl.DataFrame(
        [{"name": n, "position": p, "par": v} for n, p, v in rows],
        schema={"name": pl.Utf8, "position": pl.Utf8, "par": pl.Float64},
    )


def test_drop_measures_the_fall_to_the_nth_next_at_the_position() -> None:
    board = _par_board([(f"WR{i}", "WR", float(100 - 10 * i)) for i in range(8)])

    out = bd.positional_drop(board, spots=3)
    got = dict(zip(out.get_column("name"), out.get_column("drop")))

    assert got["WR0"] == 30.0  # 100 -> 70
    assert got["WR4"] == 30.0  # 60  -> 30


def test_a_flat_top_reports_a_drop_of_zero() -> None:
    """The 2026 running backs, and the whole reason this column exists.

    Six identical PARs is not a bug in the curve — `expected.tiers` pools ranks
    the data cannot order rather than forcing one. A drop of zero is the board
    saying "waiting costs you nothing here", which is the single most useful
    thing it can say about the top of that position.
    """
    board = _par_board(
        [(f"RB{i}", "RB", 72.6) for i in range(6)]
        + [("RB6", "RB", 70.5), ("RB7", "RB", 67.4)]
    )

    out = bd.positional_drop(board, spots=5)
    got = dict(zip(out.get_column("name"), out.get_column("drop")))

    assert got["RB0"] == 0.0


def test_drop_never_crosses_positions() -> None:
    """A receiver's drop must not be measured against a running back.

    The failure mode is a frame that happens to be sorted by PAR globally: the
    shift would walk straight out of the position and return a number that looks
    entirely plausible.
    """
    board = _par_board(
        [("RB1", "RB", 90.0), ("WR1", "WR", 80.0), ("RB2", "RB", 70.0),
         ("WR2", "WR", 60.0), ("RB3", "RB", 50.0), ("WR3", "WR", 40.0)]
    )

    out = bd.positional_drop(board, spots=2)
    got = dict(zip(out.get_column("name"), out.get_column("drop")))

    assert got["RB1"] == 40.0  # 90 -> RB3's 50, not WR2's 60
    assert got["WR1"] == 40.0  # 80 -> WR3's 40, not RB3's 50


def test_drop_is_null_when_nothing_is_left_to_fall_to() -> None:
    """Null is "the position runs out here", which is not a drop of zero."""
    board = _par_board([("A", "TE", 30.0), ("B", "TE", 20.0), ("C", "TE", 10.0)])

    out = bd.positional_drop(board, spots=5)

    assert out.get_column("drop").null_count() == 3


def test_drop_does_not_reorder_the_board() -> None:
    """The shift needs a PAR-sorted frame; the caller's order is not ours to keep.

    `build` hands back a board the app renders in the order it was given, so a
    silent re-sort here would change what the first screen shows without
    changing a single number.
    """
    board = _par_board([("C", "WR", 10.0), ("A", "WR", 30.0), ("B", "WR", 20.0)])

    out = bd.positional_drop(board, spots=1)

    assert out.get_column("name").to_list() == ["C", "A", "B"]


def test_drop_puts_unscored_players_last_within_a_position() -> None:
    """`sort(descending=True)` defaults to nulls FIRST — the CLAUDE.md trap.

    Unguarded, a player with no PAR heads his position group and every drop below
    him is measured from nothing.
    """
    board = _par_board([("Scored", "WR", 50.0), ("Unscored", "WR", None),
                        ("Lower", "WR", 20.0)])

    out = bd.positional_drop(board, spots=1)
    got = dict(zip(out.get_column("name"), out.get_column("drop")))

    assert got["Scored"] == 30.0, "the null PAR player was sorted to the top"


def test_drop_rejects_a_meaningless_horizon() -> None:
    with pytest.raises(ValueError):
        bd.positional_drop(_par_board([("A", "WR", 10.0)]), spots=0)


# --- the trap that ships a board of unscored players ------------------------


def test_sorting_the_board_puts_unranked_players_last() -> None:
    """Polars defaults `nulls_last=False`, ascending included.

    A player the ADP curve could not match has a null `board_rank`. Sorted with
    the default, those open the board — so the top of the big board would be the
    players it knows least about, looking exactly like the players it rates most.
    """
    board = pl.DataFrame(
        {"name": ["Ranked", "Unranked", "Also ranked"], "board_rank": [2, None, 1]},
        schema={"name": pl.Utf8, "board_rank": pl.Int32},
    )

    ordered = board.sort("board_rank", nulls_last=True).get_column("name").to_list()

    assert ordered[0] == "Also ranked"
    assert ordered[-1] == "Unranked", "a null rank opened the board"


# --- driving the tab --------------------------------------------------------


def _app():
    """A run app, or a failure. Import inside — `AppTest` is slow to import."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=300).run()
    if at.exception:
        pytest.fail(f"app raised on first render: {at.exception[0].value}")
    return at


@pytest.fixture(scope="module")
def board_tab():
    """The big board's controls, or a skip when there is no league to price."""
    if SLEEPER_USERNAME == "CHANGE_ME":
        pytest.skip("no handle configured (FF_EDGE_SLEEPER_USER unset)")
    at = _app()
    if not [s for s in at.multiselect if s.key == "big_board_positions"]:
        pytest.skip("no big board — needs FF_EDGE_LEAGUE_ID and an ADP board")
    return at


def test_the_row_slider_can_be_driven_the_whole_way_up(board_tab) -> None:
    """Repetition, forward, on the tab's primary control.

    This is the `test_draft_day.py` pattern ported rather than re-derived. The
    segfault it was written for needed an allocation sequence and not an input:
    one render passed, nine killed the interpreter, and the same fifteen steps in
    reverse never failed once. So this walks up, deliberately.
    """
    at = board_tab
    slider = at.slider(key="big_board_rows")
    values = list(range(int(slider.min), int(slider.max) + 1, int(slider.step)))
    assert len(values) > 1, "a single-step slider cannot exercise repetition"

    for value in values:
        at = at.slider(key="big_board_rows").set_value(value).run()
        assert not at.exception, f"{value} rows raised: {at.exception[0].value}"


def test_the_waiting_horizon_can_be_driven_the_whole_way_up(board_tab) -> None:
    """The second control on the tab, and it recomputes on every step.

    `cost_of_waiting` is real work — a survival model walked over every player at
    every pick — so unlike the row slider this one actually rebuilds a frame each
    time. That makes it the more likely of the two to find an allocation-sequence
    bug, and the reason it is driven forward rather than sampled.
    """
    at = board_tab
    if not [s for s in at.slider if s.key == "big_board_horizon"]:
        pytest.skip("no pick list — cost-of-waiting panel not rendered")

    slider = at.slider(key="big_board_horizon")
    values = list(range(int(slider.min), int(slider.max) + 1, int(slider.step)))
    if len(values) < 2:
        pytest.skip("too few picks to exercise repetition")

    for value in values:
        at = at.slider(key="big_board_horizon").set_value(value).run()
        assert not at.exception, f"horizon {value} raised: {at.exception[0].value}"


def test_every_signal_filter_renders(board_tab) -> None:
    """Each filter in turn, including the ones that select nothing.

    An empty filtered board is a legitimate state — there may be no splits on a
    given day — and it has to render a message rather than an empty table or a
    traceback.
    """
    at = board_tab
    for level in ("both up", "split", "both down", "quiet"):
        at = at.multiselect(key="big_board_signals").set_value([level]).run()
        assert not at.exception, f"{level!r} raised: {at.exception[0].value}"
