"""Fantasy Football Calculator ADP — free REST, no key, no scraping.

ADP is the market price. On its own that's mildly useful; what makes this file
worth having is the two things around it:

  1. `stdev` — FFC ships the dispersion of each player's draft position, not
     just the mean. `survival()` turns that into P(still there at pick N), which
     is the actual question you have on the clock.
  2. `snapshot()` — ADP is a moving target in August and the API only serves
     *today*. Nobody sells the history, so you have to accumulate it yourself,
     one day at a time. Start early; it cannot be backfilled.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import requests

from src.cache import frame
from src.config import ADP_SCORING, ADP_TEAMS, ADP_YEAR, DATA_DIR

BASE = "https://fantasyfootballcalculator.com/api/v1/adp"

# Minimum standard deviation, in picks, for a player's draft slot. See slot_scale.
_MIN_SLOT_SD = 0.5

_session = requests.Session()
_session.headers.update({"User-Agent": "ff-edge/0.1 (personal research)"})


def fetch(
    scoring: str = ADP_SCORING,
    teams: int = ADP_TEAMS,
    year: int = ADP_YEAR,
    position: str = "all",
    force: bool = False,
) -> pl.DataFrame:
    """Current ADP for one format. Rows tagged with format + pull date."""

    def _load() -> pl.DataFrame:
        params = {"teams": teams, "year": year, "position": position}
        resp = _session.get(f"{BASE}/{scoring}", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        df = pl.DataFrame(
            payload.get("players", []), infer_schema_length=None, strict=False
        )
        if df.height == 0:
            return df
        return df.with_columns(
            pl.lit(scoring).alias("scoring"),
            pl.lit(teams).alias("teams"),
            pl.lit(payload.get("meta", {}).get("total_drafts")).alias("total_drafts"),
            pl.lit(date.today().isoformat()).alias("pulled_on"),
        )

    return frame(f"adp_{scoring}_{teams}_{year}_{position}", "live", _load, force)


def multi_format(
    scorings: tuple[str, ...] = ("ppr", "half-ppr", "standard"),
    teams: int = ADP_TEAMS,
    year: int = ADP_YEAR,
    force: bool = False,
) -> pl.DataFrame:
    """Stack several scoring formats.

    The standard-vs-PPR rank gap is a clean, free read on how reception-dependent
    a player's value is — a back who moves 20 spots between formats is being
    priced almost entirely on receiving work, which is the volume most exposed
    to a coaching change.
    """
    parts = [fetch(s, teams, year, force=force) for s in scorings]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def snapshot(
    scoring: str = ADP_SCORING,
    teams: int = ADP_TEAMS,
    year: int = ADP_YEAR,
    force: bool = False,
) -> pl.DataFrame:
    """Append today's ADP to the rolling history file. Idempotent per day.

    force defaults to False because `fetch` already has a 1h TTL: a daily cron
    is always past it and gets a live pull anyway, while re-running bootstrap
    twice in a row stays completely offline.
    """
    today = fetch(scoring, teams, year, force=force)
    if today.height == 0:
        return today

    path = DATA_DIR / f"adp_history_{scoring}_{teams}_{year}.parquet"
    stamp = date.today().isoformat()

    if path.exists():
        prior = pl.read_parquet(path).filter(pl.col("pulled_on") != stamp)
        history = pl.concat([prior, today], how="diagonal_relaxed")
    else:
        history = today

    history.write_parquet(path)
    return history


def movement(
    days: int = 7,
    scoring: str = ADP_SCORING,
    teams: int = ADP_TEAMS,
    year: int = ADP_YEAR,
) -> pl.DataFrame:
    """Biggest risers and fallers between the oldest and newest snapshot in window.

    Negative `adp_change` means the ADP number went down, i.e. the player is
    going earlier — rising. Returns empty until you have two days of snapshots.
    """
    path = DATA_DIR / f"adp_history_{scoring}_{teams}_{year}.parquet"
    if not path.exists():
        return pl.DataFrame()

    history = pl.read_parquet(path)
    dates = sorted(history.get_column("pulled_on").unique().to_list())
    if len(dates) < 2:
        return pl.DataFrame()

    newest = dates[-1]
    oldest = dates[max(0, len(dates) - 1 - days)]

    cols = ["name", "position", "team", "adp"]
    old = history.filter(pl.col("pulled_on") == oldest).select(cols)
    new = history.filter(pl.col("pulled_on") == newest).select(cols)

    return (
        new.join(old, on=["name", "position", "team"], how="inner", suffix="_prior")
        .with_columns((pl.col("adp") - pl.col("adp_prior")).alias("adp_change"))
        .with_columns(
            pl.when(pl.col("adp_change") < 0)
            .then(pl.lit("rising"))
            .otherwise(pl.lit("falling"))
            .alias("direction")
        )
        .sort(pl.col("adp_change").abs(), descending=True)
    )


def slot_scale(stdev: Any) -> np.ndarray:
    """FFC's draft-slot dispersion, floored — the one place that floor is defined.

    FFC reports stdev 0 for players with almost no draft sample. Taken literally
    that makes a player's draft slot deterministic, which turns `survival()` into
    a step function and makes the simulator's draft board identical in every run.
    0.5 picks is small enough to be honest about a genuinely tightly-drafted
    player and large enough to keep the math continuous.

    Shared deliberately: `survival()` evaluates the normal CDF at one pick, and
    `simulate.draft_orders()` samples from the same normal. They must agree about
    the distribution or the board you plan against isn't the board you tested.
    """
    sd = np.asarray(stdev, dtype=float)
    return np.maximum(np.nan_to_num(sd, nan=0.0), _MIN_SLOT_SD)


def survival(adp_df: pl.DataFrame, pick_no: int) -> pl.DataFrame:
    """P(player is still on the board at `pick_no`).

    Treats a player's draft slot as normal around their ADP with the observed
    stdev, so survival is 1 - Phi((pick_no - adp) / stdev).

    Why this reframes the draft: two players both at ADP 40, one with stdev 3 and
    one with stdev 14, have roughly a 0.4% and a 28% chance of lasting to pick 48.
    Identical price, completely different decision. "Who's the best player
    available" is the wrong question; "who do I lose by waiting" is the right one,
    and only the second one uses the dispersion you already have for free.

    stdev is floored by `slot_scale`, which the draft simulator samples from too
    so both agree about the distribution.
    """
    if adp_df.height == 0:
        return adp_df

    col = f"p_available_at_{pick_no}"

    def _p(adp: float | None, sd: float | None) -> float | None:
        if adp is None:
            return None
        sd = float(slot_scale(sd or 0.0))
        z = (pick_no - float(adp)) / sd
        return round(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 4)

    return adp_df.with_columns(
        pl.struct(["adp", "stdev"])
        .map_elements(lambda r: _p(r["adp"], r["stdev"]), return_dtype=pl.Float64)
        .alias(col)
    )
