"""Read-only Sleeper client. No auth, no writes, no way to touch a league.

Sleeper's v1 API is fully public — no key, no OAuth, generous rate limits. That
makes it the one place you can get *your specific leaguemates'* behavior for
free, which is worth more for draft prep than any generic projection set.

Two functions here matter more than the rest:

  - `draft_history` walks the `previous_league_id` chain backwards, so a single
    current league ID unspools four years of the same twelve people drafting.
  - `all_transactions` gives FAAB bids, hold times, and positional bias.

Also note that `league()` returns `scoring_settings` and `roster_positions`.
Replacement level should be derived from those — a league with a superflex slot
and 6-point passing TDs has a different RB2 than the generic PPR baseline every
public ranking assumes.
"""

from __future__ import annotations

import time
from typing import Any

import polars as pl
import requests

from cache import blob, frame
from config import LEAGUE_IDS, SEASON, SLEEPER_USERNAME

BASE = "https://api.sleeper.app/v1"

_session = requests.Session()
_session.headers.update({"User-Agent": "ff-edge/0.1 (personal research)"})


def _get(path: str, retries: int = 4) -> Any:
    """GET a Sleeper endpoint. None on 404, exponential backoff on 429."""
    url = f"{BASE}/{path.lstrip('/')}"
    for attempt in range(retries):
        resp = _session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


def _frame(records: Any) -> pl.DataFrame:
    """Build a frame from Sleeper's ragged JSON.

    infer_schema_length=None scans every row (Sleeper adds keys partway through
    a list — settings blobs differ by league age), and strict=False keeps a
    single odd value from failing the whole build.
    """
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records, infer_schema_length=None, strict=False)


# --- Primitives -------------------------------------------------------------


def nfl_state(force: bool = False) -> dict:
    """Current season, week, and season type. Drives 'what week is it' logic."""
    return blob("sleeper_state", "live", lambda: _get("state/nfl"), force)


def user(username: str | None = None, force: bool = False) -> dict | None:
    """Resolve a display name to a user object (contains the user_id)."""
    username = username or SLEEPER_USERNAME
    return blob(f"sleeper_user_{username}", "static", lambda: _get(f"user/{username}"), force)


def my_leagues(season: int = SEASON, force: bool = False) -> pl.DataFrame:
    """Leagues for the configured user, or just LEAGUE_IDS if that's set."""
    if LEAGUE_IDS:
        return _frame([league(lid, force=force) for lid in LEAGUE_IDS])

    me = user(force=force)
    if not me:
        return pl.DataFrame()
    uid = me["user_id"]
    return frame(
        f"sleeper_leagues_{uid}_{season}",
        "season",
        lambda: _frame(_get(f"user/{uid}/leagues/nfl/{season}")),
        force,
    )


def league(league_id: str, force: bool = False) -> dict | None:
    """League settings, scoring, and roster_positions."""
    return blob(f"sleeper_league_{league_id}", "season", lambda: _get(f"league/{league_id}"), force)


def rosters(league_id: str, force: bool = False) -> pl.DataFrame:
    """Current rosters keyed by roster_id, with owner_id and player_id lists."""
    return frame(
        f"sleeper_rosters_{league_id}",
        "live",
        lambda: _frame(_get(f"league/{league_id}/rosters")),
        force,
    )


def league_users(league_id: str, force: bool = False) -> pl.DataFrame:
    """Managers in the league — maps user_id to a human display_name."""
    return frame(
        f"sleeper_users_{league_id}",
        "season",
        lambda: _frame(_get(f"league/{league_id}/users")),
        force,
    )


def matchups(league_id: str, week: int, force: bool = False) -> pl.DataFrame:
    """Head-to-head matchups and starters for a week."""
    return frame(
        f"sleeper_matchups_{league_id}_w{week}",
        "live",
        lambda: _frame(_get(f"league/{league_id}/matchups/{week}")),
        force,
    )


def transactions(league_id: str, week: int, force: bool = False) -> pl.DataFrame:
    """Adds, drops, trades, and waiver bids for a single week."""
    return frame(
        f"sleeper_txn_{league_id}_w{week}",
        "live",
        lambda: _frame(_get(f"league/{league_id}/transactions/{week}")),
        force,
    )


def drafts(league_id: str, force: bool = False) -> pl.DataFrame:
    """Draft objects attached to a league (usually one)."""
    return frame(
        f"sleeper_drafts_{league_id}",
        "season",
        lambda: _frame(_get(f"league/{league_id}/drafts")),
        force,
    )


def draft_picks(draft_id: str, force: bool = False) -> pl.DataFrame:
    """Every pick in a draft, in order, with the drafting roster."""
    return frame(
        f"sleeper_picks_{draft_id}",
        "static",
        lambda: _frame(_get(f"draft/{draft_id}/picks")),
        force,
    )


def user_drafts(season: int = SEASON, force: bool = False) -> pl.DataFrame:
    """All drafts the configured user participated in, across leagues."""
    me = user(force=force)
    if not me:
        return pl.DataFrame()
    uid = me["user_id"]
    return frame(
        f"sleeper_userdrafts_{uid}_{season}",
        "season",
        lambda: _frame(_get(f"user/{uid}/drafts/nfl/{season}")),
        force,
    )


def players_nfl(force: bool = False) -> pl.DataFrame:
    """Sleeper's full player dictionary (~5MB).

    Sleeper explicitly asks that this be pulled at most once a day, hence the
    hardcoded 24h TTL rather than a config key.
    """

    def _load() -> pl.DataFrame:
        data = _get("players/nfl") or {}
        return _frame([{"player_id": pid, **rec} for pid, rec in data.items()])

    return frame("sleeper_players", 24, _load, force)


# --- Composites -------------------------------------------------------------


def league_chain(
    league_id: str, seasons_back: int = 4, force: bool = False
) -> list[tuple[str, str]]:
    """Walk `previous_league_id` backwards, returning [(league_id, season), ...].

    Sleeper models a recurring league as a linked list, one node per season. This
    is the traversal that makes multi-year history possible at all, so it lives
    in one place rather than being re-implemented by every caller that wants it.

    Newest season first. Stops early if the chain ends (the league's first year).
    """
    chain: list[tuple[str, str]] = []
    current: str | None = league_id

    for _ in range(seasons_back):
        if not current:
            break
        meta = league(current, force=force)
        if not meta:
            break
        chain.append((current, str(meta.get("season"))))
        current = meta.get("previous_league_id")

    return chain


def all_transactions(league_id: str, weeks: int = 18, force: bool = False) -> pl.DataFrame:
    """Every transaction in a season, week-tagged.

    This is the input to modeling manager behavior: who overbids FAAB early,
    who streams defenses, who holds injured players too long, who never trades.
    Those tendencies are stable year to year and they're exploitable in a draft.

    Concatenated `diagonal_relaxed` because the shape of a transaction record
    varies (waivers carry `settings.waiver_bid`, trades carry `draft_picks`).
    """
    parts = []
    for week in range(1, weeks + 1):
        df = transactions(league_id, week, force=force)
        if df.height:
            parts.append(df.with_columns(pl.lit(week).alias("week")))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def draft_history(league_id: str, seasons_back: int = 4, force: bool = False) -> pl.DataFrame:
    """Walk `previous_league_id` backwards and pull every prior draft.

    The single highest-value function in this file. Sleeper links each league
    season to its predecessor, so one current league ID yields years of *your*
    leaguemates' actual draft behavior — reach patterns, positional runs, who
    always takes a QB early. No paid service sells that, because it's specific
    to your twelve people.

    Rows are tagged with `season` and `league_id` so you can compare across years.
    """
    parts = []

    for current, season in league_chain(league_id, seasons_back, force=force):
        season_drafts = drafts(current, force=force)
        draft_ids = (
            season_drafts.get_column("draft_id").to_list()
            if "draft_id" in season_drafts.columns
            else []
        )
        for draft_id in draft_ids:
            picks = draft_picks(str(draft_id), force=force)
            if picks.height:
                parts.append(
                    picks.with_columns(
                        pl.lit(season).alias("season"),
                        pl.lit(current).alias("league_id"),
                    )
                )

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def transaction_history(
    league_id: str, seasons_back: int = 4, weeks: int = 18, force: bool = False
) -> pl.DataFrame:
    """Every transaction across prior seasons of the same league.

    The seasonal counterpart to `draft_history`, and the reason it exists: during
    the offseason the *current* league has zero transactions, so asking only
    about it tells you nothing. The behavior worth modeling — FAAB aggression,
    how long a manager holds an injured player, who actually trades — is all in
    the seasons that already finished.

    Rows are tagged with `season` and `league_id`.
    """
    parts = []

    for current, season in league_chain(league_id, seasons_back, force=force):
        df = all_transactions(current, weeks=weeks, force=force)
        if df.height:
            parts.append(
                df.with_columns(
                    pl.lit(season).alias("season"),
                    pl.lit(current).alias("league_id"),
                )
            )

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def league_bundle(league_id: str, force: bool = False) -> dict:
    """Everything about one league in a single call — the usual starting point."""
    meta = league(league_id, force=force) or {}
    return {
        "settings": meta.get("settings", {}),
        "scoring": meta.get("scoring_settings", {}),
        "roster_positions": meta.get("roster_positions", []),
        "users": league_users(league_id, force=force),
        "rosters": rosters(league_id, force=force),
        "draft_history": draft_history(league_id, force=force),
        "transactions": transaction_history(league_id, force=force),
    }
