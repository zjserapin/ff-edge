"""League profiles: one object that carries a format and its matching market.

Everything in this project that values a player depends on two settings that
have to agree with each other:

  1. **The roster format**, which sets how many starters the league demands and
     therefore where replacement level falls.
  2. **The ADP market**, which sets the price the board is read against and the
     historical curve expected points are estimated from.

The 2026 superflex bug is what happens when those two drift apart. The league
added a SUPER_FLEX, so demand for quarterbacks doubled — but the board was still
priced from a 1QB market, which says quarterbacks are cheap. Read together, that
combination reports that QBs are nearly free *and* start twice. Neither setting
was wrong on its own. The pairing was.

So a profile is not a convenience wrapper. It exists so a format cannot be
selected without also selecting the market that prices it, and so switching
leagues is one named choice rather than four independent ones that have to be
kept in sync by hand.

Pick one with the environment, the same way the league id is passed:

    FF_EDGE_PROFILE=standard_12 uv run python -c "..."

`shiva_bowl` is the default and the only Sleeper-backed profile; the others are
synthetic formats for reusing the analysis against a league this project does
not read live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    LEAGUE_ADP_SCORING,
    LEAGUE_ADP_TEAMS,
    SUPERFLEX_ADP_SCORING,
)


@dataclass(frozen=True)
class LeagueProfile:
    """A format and the market that prices it, as one indivisible choice."""

    name: str
    label: str
    teams: int
    roster_positions: list[str]
    scoring: dict[str, float]

    # The FFC format the draft board and the historical curve are both read
    # from. `adp_teams` is carried for the cache key and for honesty about what
    # was requested — see the note on FFC pooling below.
    adp_scoring: str
    adp_teams: int

    # Whether managers keep players. False makes the whole keeper layer a
    # documented no-op rather than a branch: draft demand equals league demand,
    # and keeper-adjusted ADP equals ADP.
    keepers: bool

    # Whether to read live settings, keepers, and picks from Sleeper. Synthetic
    # profiles describe a format nobody has configured an account for.
    sleeper_backed: bool

    notes: str = ""
    assumptions: list[str] = field(default_factory=list)


# Standard scoring is half-PPR with the reception point removed, and nothing
# else. Spelling it as a delta rather than a second literal dict keeps the two
# from drifting when the league changes a rule that is not about receptions.
_STANDARD_SCORING = dict(DEFAULT_SCORING) | {"rec": 0.0}
_PPR_SCORING = dict(DEFAULT_SCORING) | {"rec": 1.0}

# A dynasty startup roster. Bench and taxi depth is cosmetic for valuation —
# `config.NON_STARTING_SLOTS` means neither contributes demand — so the only
# load-bearing line here is the starters.
_DYNASTY_ROSTER = [
    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF",
] + ["BN"] * 10 + ["TAXI"] * 4


PROFILES: dict[str, LeagueProfile] = {
    "shiva_bowl": LeagueProfile(
        name="shiva_bowl",
        label="Shiva Bowl — 10-team superflex keeper (live)",
        teams=10,
        roster_positions=list(DEFAULT_ROSTER_POSITIONS),
        scoring=dict(DEFAULT_SCORING),
        adp_scoring=SUPERFLEX_ADP_SCORING,
        adp_teams=LEAGUE_ADP_TEAMS,
        keepers=True,
        sleeper_backed=True,
        notes=(
            "The real league. Roster, scoring, keepers and picks all come from "
            "Sleeper at runtime; the values here are the offline fallback."
        ),
    ),
    "standard_12": LeagueProfile(
        name="standard_12",
        label="12-team standard redraft",
        teams=12,
        roster_positions=[
            "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
        ] + ["BN"] * 6,
        scoring=_STANDARD_SCORING,
        adp_scoring="standard",
        adp_teams=12,
        keepers=False,
        sleeper_backed=False,
        notes=(
            "Fully supported. FFC publishes standard ADP for every season in "
            "the label window (2018-2026), so both the draft board and the "
            "historical expected-points curve come from the matching market."
        ),
        assumptions=[
            "One FLEX, not two. Change `roster_positions` if yours differs — "
            "it moves replacement level at RB/WR/TE directly.",
            "Standard is half-PPR with `rec` set to 0.0 and every other scoring "
            "rule inherited from the Shiva Bowl.",
        ],
    ),
    "dynasty_10": LeagueProfile(
        name="dynasty_10",
        label="10-team dynasty startup",
        teams=10,
        roster_positions=_DYNASTY_ROSTER,
        scoring=_PPR_SCORING,
        adp_scoring="dynasty",
        adp_teams=10,
        keepers=False,
        sleeper_backed=False,
        notes=(
            "PARTIAL — see `market_gap`. FFC has collected only 55 dynasty "
            "drafts for 2026 and publishes no player rows at any team count, "
            "so there is no 2026 dynasty board to price against. History is "
            "there (2018-2025), so the curve can be estimated; the current "
            "board cannot. The profile refuses to substitute a redraft board "
            "rather than reporting a confident wrong number."
        ),
        assumptions=[
            "PPR, one QB, two FLEX. Superflex dynasty is at least as common — "
            "swap `roster_positions` and set `adp_scoring='2qb'` together.",
            "A startup is valued here as a one-year redraft. Dynasty value is "
            "multi-year and age-dependent, and this project has no age curve, "
            "so the ranking answers 'who helps in 2026', not 'who to build on'.",
        ],
    ),
}

DEFAULT_PROFILE = "shiva_bowl"


def resolve(name: str | None = None) -> LeagueProfile:
    """The profile to use, from the argument, the environment, or the default.

    Reads `FF_EDGE_PROFILE` so a profile can be chosen per-command the same way
    the league id is, and raises on an unknown name rather than falling back —
    a typo that silently returned the Shiva Bowl would price a 12-team standard
    league as a superflex keeper league and look entirely plausible doing it.
    """
    name = name or os.environ.get("FF_EDGE_PROFILE") or DEFAULT_PROFILE
    if name not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; known profiles: {known}")
    return PROFILES[name]


def customize(base: str | LeagueProfile, **overrides) -> LeagueProfile:
    """A one-off variant of a built-in profile.

    For the common case of "my league is standard_12 but with two FLEX". Returns
    a new frozen profile; the built-ins are never mutated.
    """
    profile = base if isinstance(base, LeagueProfile) else resolve(base)
    return replace(profile, **overrides)


def as_settings(profile: LeagueProfile) -> dict:
    """The profile in the shape `scoring.league_settings` returns.

    Lets a synthetic profile flow through every function that already takes a
    settings dict, so nothing downstream needs to know whether the format came
    from Sleeper or from this file.
    """
    return {
        "season": None,
        "teams": profile.teams,
        "scoring": dict(profile.scoring),
        "roster_positions": list(profile.roster_positions),
        "playoff_week_start": 15,
        "playoff_teams": 6,
        "source": f"profile: {profile.name}",
    }


def market_gap(profile: LeagueProfile, season: int) -> str | None:
    """Why this profile cannot be priced this season, or None if it can.

    Checked before a board is built so the failure is a sentence rather than an
    empty frame. Only the known-empty case is hardcoded; anything else is left
    to the fetch, which reports its own emptiness.
    """
    if profile.adp_scoring == "dynasty" and season >= 2026:
        return (
            f"FFC publishes no dynasty ADP for {season} (55 drafts collected, "
            "zero player rows at every team count). Historical dynasty ADP "
            "exists through 2025, so the expected-points curve is estimable, "
            "but there is no current board to draft off."
        )
    return None
