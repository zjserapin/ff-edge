"""The join layer. Every source names players differently.

nflverse uses gsis_id, Sleeper uses its own integer-as-string IDs, FFC's ADP
feed has no IDs at all — just names. A join that silently drops 40% of its rows
returns a perfectly plausible-looking DataFrame, which is the single most common
way a project like this goes quietly wrong. So all of it lives in one file, and
`match_report()` exists so the match rate is something you *see* rather than
something you assume.
"""

from __future__ import annotations

import polars as pl

from src.cache import frame
from src.nflverse import ff_playerids

# Generational suffixes carried inconsistently across sources: nflverse says
# "Marvin Harrison", FantasyPros says "Marvin Harrison Jr.".
_SUFFIXES = r"\s+(jr|sr|ii|iii|iv|v)$"


def normalize(col: str) -> pl.Expr:
    """Name -> join key: lowercase, punctuation-free, suffix-free, single-spaced.

    Handles the three shapes that break naive joins:
      D.J. Moore            -> dj moore
      Marvin Harrison Jr.   -> marvin harrison
      Amon-Ra St. Brown     -> amon ra st brown
    """
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"[.'`]", "")  # D.J. -> DJ, O'Neal -> ONeal
        .str.replace_all(r"-", " ")  # Amon-Ra -> Amon Ra
        .str.strip_chars()
        .str.replace_all(_SUFFIXES, "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("nkey")
    )


_KEEP = [
    "gsis_id",
    "sleeper_id",
    "pfr_id",
    "espn_id",
    "yahoo_id",
    "fantasypros_id",
    "mfl_id",
    "name",
    "merge_name",
    "position",
    "team",
    "birthdate",
    "draft_year",
]


def crosswalk(force: bool = False) -> pl.DataFrame:
    """The ID bridge, with every *_id column cast to Utf8.

    That cast is the whole point. ff_playerids types sleeper_id, espn_id, and
    friends as Int64, while the Sleeper API returns those same IDs as strings.
    Joining Int64 to Utf8 doesn't error in polars — it either raises late or
    matches nothing depending on how you got there. Cast once, here, and no
    downstream join has to think about it.
    """

    def _load() -> pl.DataFrame:
        raw = ff_playerids(force=force)
        cols = [c for c in _KEEP if c in raw.columns]
        df = raw.select(cols).with_columns(
            [pl.col(c).cast(pl.Utf8, strict=False) for c in cols if c.endswith("_id")]
        )
        return df.with_columns(normalize("name")).unique(
            subset=["nkey", "position"], keep="first"
        )

    return frame("crosswalk", "static", _load, force)


def attach_gsis(df: pl.DataFrame, sleeper_col: str = "player_id") -> pl.DataFrame:
    """Sleeper frame -> gsis_id / pfr_id / canonical name."""
    xw = crosswalk().select(["sleeper_id", "gsis_id", "pfr_id", "name", "position"])
    return df.with_columns(pl.col(sleeper_col).cast(pl.Utf8, strict=False)).join(
        xw, left_on=sleeper_col, right_on="sleeper_id", how="left"
    )


def attach_sleeper(df: pl.DataFrame, gsis_col: str = "player_id") -> pl.DataFrame:
    """nflverse frame -> sleeper_id, so you can join to your league's rosters."""
    xw = crosswalk().select(["gsis_id", "sleeper_id", "name", "position"])
    return df.with_columns(pl.col(gsis_col).cast(pl.Utf8, strict=False)).join(
        xw, left_on=gsis_col, right_on="gsis_id", how="left"
    )


def match_by_name(
    df: pl.DataFrame, name_col: str = "name", pos_col: str | None = "position"
) -> pl.DataFrame:
    """Last-resort join for name-only sources (FFC ADP).

    Uses position as a tiebreaker when available — there are genuinely two
    Michael Carters and two Josh Allens, and position separates them.
    """
    xw = crosswalk()
    left = df.with_columns(normalize(name_col))

    if pos_col and pos_col in df.columns:
        left = left.with_columns(pl.col(pos_col).cast(pl.Utf8).str.to_uppercase().alias("_pos"))
        right = xw.with_columns(pl.col("position").cast(pl.Utf8).str.to_uppercase().alias("_pos"))
        return left.join(
            right.drop("position", "name"), on=["nkey", "_pos"], how="left"
        ).drop("_pos")

    return left.join(xw.drop("name", "position"), on="nkey", how="left")


def unmatched(df: pl.DataFrame, id_col: str = "gsis_id") -> pl.DataFrame:
    """Rows that failed to match. Always look at this before trusting a join."""
    return df.filter(pl.col(id_col).is_null())


def match_report(df: pl.DataFrame, id_col: str = "gsis_id") -> dict:
    """{rows, matched, rate} — the number you should print after every join."""
    rows = df.height
    matched = df.filter(pl.col(id_col).is_not_null()).height
    return {
        "rows": rows,
        "matched": matched,
        "rate": round(matched / rows, 4) if rows else 0.0,
    }
