"""Column headers and chart footnotes.

The display layer is where a correct number becomes a wrong one — a column
called `par_mean_starter` is not readable, and a header that says "Ppg" or
"Player display name" reads as unfinished. These are cheap to test and cheap to
regress.
"""

from __future__ import annotations

import app
from src import glossary


def test_glossary_labels_win_over_the_fallback() -> None:
    assert app._header("par_mean_starter") == "Average starter PAR"
    assert app._header("yprr") == "Yards per route run"
    assert app._header("value_gap") == "Value gap"


def test_acronyms_are_not_mangled() -> None:
    """capitalize() turns 'ppg' into 'Ppg' and 'adp' into 'Adp'."""
    assert app._header("ppg") == "Points per game"     # glossary
    assert app._header("adp") == "ADP"                 # glossary
    assert app._header("ci_lo") == "CI lo"             # fallback keeps the acronym
    assert app._header("auc_delta") == "AUC delta"


def test_identity_columns_read_as_english() -> None:
    for raw in ("player_display_name", "player_name", "name"):
        assert app._header(raw) == "Player"
    assert app._header("season") == "Season"
    assert app._header("position") == "Position"


def test_unknown_columns_degrade_readably() -> None:
    """A new column must never render as a raw snake_case name."""
    assert app._header("some_new_metric") == "Some new metric"
    assert "_" not in app._header("another_unmapped_thing")


def test_every_header_is_unique_within_a_frame() -> None:
    """Two raw columns sharing a label would collapse into one pandas column."""
    frames = {
        "valuation": [
            "name", "position", "team", "adp", "quality_pct", "opportunity_pct",
            "market_pct", "value_gap", "path_score", "yprr", "teammate_top_share",
            "verdict",
        ],
        "scarcity": [
            "season", "position", "pos_rank", "player_display_name", "games",
            "fantasy_points", "ppg", "par_ppg",
        ],
        "par": [
            "season", "position", "demand", "replacement_rank", "replacement_ppg",
            "par_mean_starter", "par_total",
        ],
    }
    for name, cols in frames.items():
        headers = [app._header(c) for c in cols]
        assert len(headers) == len(set(headers)), f"{name} has duplicate headers: {headers}"


def test_charted_metrics_all_have_definitions() -> None:
    """Every column a chart footnote names must resolve, or the footer is blank."""
    charted = [
        "par_mean_starter", "demand", "replacement_rank", "replacement_ppg",
        "pos_rank", "ppg", "fantasy_points", "games",
        "overall_par_rank", "par_ppg", "share", "pool_size",
    ]
    for column in charted:
        assert glossary.lookup(column) is not None, f"{column} has no glossary entry"
        assert glossary.describe(column), f"{column} has an empty description"


# --- the crash that had no traceback ---------------------------------------


def test_pyarrow_is_not_the_release_that_segfaults() -> None:
    """pyarrow 25.0.0 kills the process while rendering a table.

    Every frame in `app.py` is polars, and `st.dataframe` converts each one
    through pandas and back into Arrow bytes for the browser. On pyarrow 25.0.0
    that round trip corrupts memory inside `pandas_compat` and takes the
    interpreter with it — SIGSEGV, no traceback, no Streamlit error page, the
    server simply gone.

    It was found by walking the Draft Day pick selector through the fifteen
    picks Zach owns. It died on the ninth render, every time, in the *picks*
    table — whose contents are identical on every rerun. Walking the same
    fifteen picks in the opposite order never died. That is heap-layout
    dependence, not a bad input, which is why the whole test suite passed
    against a build that could not survive a draft.

    Pinned here rather than trusted to `pyproject.toml` alone because the lock
    is what actually gets installed, and a lock can carry a version the
    constraint would now reject.
    """
    import pyarrow

    assert pyarrow.__version__ != "25.0.0", (
        "pyarrow 25.0.0 segfaults on the pandas/Arrow round trip that every "
        "st.dataframe call makes — run `uv sync` to pick up the exclusion in "
        "pyproject.toml"
    )
