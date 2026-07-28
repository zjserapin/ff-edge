"""The glossary must actually cover what the app renders.

A definition file that drifts out of sync with the columns is worse than none —
it looks authoritative and silently stops explaining the thing in front of you.
These tests tie it to the real column names.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import breakout as bo
from src import features as ft
from src import glossary


def test_every_term_is_complete() -> None:
    for key, term in glossary.TERMS.items():
        assert term.label, f"{key} has no label"
        assert term.short, f"{key} has no short definition"
        assert term.group, f"{key} has no group"
        # The tooltip has to fit in a tooltip.
        assert len(term.short) < 200, f"{key} short definition is too long"


def test_feature_columns_are_all_documented() -> None:
    """Every modelling feature needs a definition — those are the opaque ones."""
    documented = set(glossary.TERMS)
    for position in (None, "QB", "RB", "WR", "TE"):
        for col in ft.feature_columns(position) + ft.cluster_feature_columns(position):
            assert col in documented, f"feature {col!r} has no glossary entry"
        for col in bo.model_features(position):
            assert col in documented, f"model feature {col!r} has no glossary entry"


def test_numbered_columns_resolve() -> None:
    """`p_available_at_20` must find the `p_available` definition."""
    assert glossary.describe("p_available_at_20")
    assert glossary.describe("p_available_at_147")
    assert glossary.lookup("p_available_at_3") is glossary.TERMS["p_available"]


def test_unknown_columns_degrade_quietly() -> None:
    """An undocumented column must not raise — it just gets no tooltip."""
    assert glossary.describe("some_new_column") == ""
    assert glossary.lookup("some_new_column") is None
    assert glossary.column_help(["totally_made_up", "some_new_column"]) == {}


def test_column_help_maps_only_known_columns() -> None:
    helped = glossary.column_help(["adp", "stdev", "totally_made_up", "market_var"])
    assert set(helped) == {"adp", "stdev", "market_var"}
    assert all(v for v in helped.values())


def test_groups_partition_the_terms() -> None:
    grouped = glossary.groups()
    total = sum(len(v) for v in grouped.values())
    assert total == len(glossary.TERMS)
    assert len(grouped) >= 6


def test_misleading_metrics_carry_their_caveat() -> None:
    """The definitions that exist mainly to prevent a wrong reading must say so.

    These are the columns a reader is most likely to misinterpret, and the long
    definition is the only place the caveat lives.
    """
    assert "not a projection" in glossary.TERMS["market_ppg"].long.lower()
    assert "negative" in glossary.TERMS["air_yards_share"].long.lower()
    assert "not comparable across positions" in glossary.TERMS["p_breakout"].short.lower()
    assert "full ppr" in glossary.TERMS["exp_ppg"].long.lower()
