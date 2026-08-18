"""Readable headers and cell text — app.py's `_header`/`table` for HTML.

Every table on the site goes through here. A column named `yprr` or
`par_mean_starter` means nothing on its own, and a glossary nobody opens is a
glossary nobody reads — so the header is rewritten to the glossary label and
the definition rides on the header's tooltip.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from src import glossary

# Acronyms a naive capitalize() would mangle into "Ppg" or "Adp".
_ACRONYMS = {"ppg", "adp", "par", "auc", "yprr", "tprr", "ypt", "ypc", "ypa", "hhi", "ci"}


def header(column: str) -> str:
    """Readable header for a column: its glossary label, else a tidied name."""
    term = glossary.lookup(column)
    if term:
        return term.label
    words = [
        w.upper() if w.lower() in _ACRONYMS else w
        for w in column.replace("_", " ").split()
    ]
    if not words:
        return column
    head, *rest = words
    return " ".join([head if head.isupper() else head.capitalize(), *rest])


def fmt(value: Any) -> str:
    """One cell's text. None is an em-dash — *not measured*, never zero."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else ""
    if isinstance(value, float):
        if value != value:  # NaN travels in from joins the same way None does
            return "—"
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) < 2:
            return f"{value:.2f}"
        return f"{value:.1f}"
    return str(value)


def table_ctx(df: pl.DataFrame, pretty: bool = True) -> dict[str, Any]:
    """A frame as template-ready headers and formatted rows.

    `pretty=False` keeps raw column names, for frames meant to be copied out.
    """
    helps = glossary.column_help(list(df.columns))
    numeric = {c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()}
    return {
        "headers": [
            {
                "raw": c,
                "label": header(c) if pretty else c,
                "help": helps.get(c, ""),
                "num": c in numeric,
            }
            for c in df.columns
        ],
        "rows": [[fmt(v) for v in row] for row in df.iter_rows()],
        "n": df.height,
    }
