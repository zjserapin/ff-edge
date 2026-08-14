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
