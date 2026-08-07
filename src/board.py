"""The draft board: expected points, priced against who is actually draftable.

Every other valuation in this project answers "what is this player worth in a
league like yours". This module answers the narrower and more useful question
"what is he worth in *your* draft, on the day, given who is already gone".

The distinction is not cosmetic in a keeper league. Replacement level is
conventionally the last startable player at a position, derived from roster
slots times teams. That is the right baseline for an empty league and the wrong
one for a keeper draft, because a slot filled by a keeper is not a slot anyone
will draft for. The correct baseline is:

    draft demand = league starter demand - starters already kept

and replacement is the first player past *that* line among the players still
available. In the 2026 Shiva Bowl the difference is severe: the superflex puts
league-wide quarterback demand at 20, but 13 quarterbacks are kept, so only 7
QB-capable slots will be drafted for. Pricing quarterbacks against QB21 — the
right answer for a fresh league — would badly misprice a draft where the 8th
best available quarterback is the true replacement.

Running that both ways is the point of `compare_baselines`, which exists so the
size of the correction is visible rather than asserted.

**What this module does not claim.** Expected points come from the ADP curve in
`expected.py`, which is a prior derived from price: every player at a given
positional rank gets the same number. So the board reorders players *across*
positions honestly, and within a position it inherits the market's ordering
rather than improving on it. That is a deliberate limit — the project measured
that it cannot beat ADP within a position — and it is exactly why the tiers
matter and why the promotion screen and claims ledger exist to break ties
inside a tier.
"""

from __future__ import annotations

import polars as pl

from src import adp as adp_mod
from src import expected as ex
from src import ids
from src import scoring as sc
from src import sleeper
from src.config import (
    FANTASY_POSITIONS,
    LEAGUE_ADP_TEAMS,
    SEASON,
    SLEEPER_USERNAME,
    SUPERFLEX_ADP_SCORING,
)


def kept_players(league_id: str | None = None, force: bool = False) -> pl.DataFrame:
    """Every player declared as a keeper, league-wide, from Sleeper.

    Reads the `keepers` field on each roster, which is populated once managers
    declare. A team that has not declared yet simply contributes no rows — and
    that absence is load-bearing information, so `keeper_summary` reports which
    teams are still outstanding rather than treating undeclared as none.

    Returns: owner, player_id, player_name, position, team.
    """
    league_id = league_id or sc.resolve_league_id(None, force=force)
    if not league_id:
        return pl.DataFrame()

    rosters = sleeper._get(f"league/{league_id}/rosters") or []
    users = {
        u["user_id"]: u.get("display_name")
        for u in (sleeper._get(f"league/{league_id}/users") or [])
    }
    players = sleeper.players_nfl(force=force)
    if not players.height:
        return pl.DataFrame()

    cols = [
        c for c in ("player_id", "full_name", "position", "team")
        if c in players.columns
    ]
    lookup = {r["player_id"]: r for r in players.select(cols).iter_rows(named=True)}

    rows = []
    for roster in rosters:
        owner = users.get(roster.get("owner_id"), "?")
        for pid in (roster.get("keepers") or []):
            p = lookup.get(str(pid), {})
            rows.append(
                {
                    "owner": owner,
                    "player_id": str(pid),
                    "player_name": p.get("full_name"),
                    "position": p.get("position"),
                    "team": p.get("team"),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def keeper_summary(
    kept: pl.DataFrame | None = None,
    league_id: str | None = None,
    roster_positions: list[str] | None = None,
    teams: int | None = None,
) -> pl.DataFrame:
    """Per-position: how many are kept, and how many slots the draft must fill.

    `draft_demand` is what the board prices against. `undeclared_teams` is
    carried on every row because an undeclared team is unfilled demand hiding
    in the numbers — with two keepers outstanding the true draft demand is
    somewhere between `draft_demand` and `draft_demand` minus two, and a board
    that hid that would be overconfident about scarcity.

    Returns: position, league_demand, kept, draft_demand, undeclared_teams.
    """
    kept = kept if kept is not None else kept_players(league_id)
    settings = sc.league_settings(league_id)
    roster_positions = roster_positions or settings["roster_positions"]
    teams = teams or settings["teams"]

    demand = sc.starter_demand(
        roster_positions,
        teams=teams,
        season_points=sc.score_season([ex.SEASON - 1], settings["scoring"]),
    )

    counts = {}
    if kept.height:
        counts = dict(
            kept.group_by("position").len().iter_rows()
        )

    undeclared = 0
    if kept.height:
        league_id = league_id or sc.resolve_league_id(None)
        rosters = sleeper._get(f"league/{league_id}/rosters") or []
        undeclared = sum(1 for r in rosters if not (r.get("keepers") or []))

    rows = []
    for position in FANTASY_POSITIONS:
        league = float(demand.get(position, 0.0))
        n_kept = float(counts.get(position, 0))
        rows.append(
            {
                "position": position,
                "league_demand": round(league, 1),
                "kept": int(n_kept),
                "draft_demand": round(max(league - n_kept, 0.0), 1),
                "undeclared_teams": undeclared,
            }
        )
    return pl.DataFrame(rows)


def draftable(
    kept: pl.DataFrame | None = None,
    season: int = SEASON,
    scoring: str = SUPERFLEX_ADP_SCORING,
    teams: int = LEAGUE_ADP_TEAMS,
    curve: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """This season's ADP board with keepers removed and expectations attached.

    Defaults to the **2QB** ADP board, not the league's half-PPR/10 board. In a
    superflex league the 1QB market misprices quarterbacks by a round or more,
    and since this is the board you draft off, it has to be the market that
    matches the format.

    Name-matched against Sleeper, since the two sources share no ids. Matching
    is on `ids.normalize`, and `keeper_match_report` says how many keepers
    failed to match so a silent miss cannot quietly leave a kept player on the
    board.

    Returns the `expected.expected_points` columns plus `kept` and `kept_by`.
    """
    curve = curve if curve is not None else ex.adp_curve()
    board = adp_mod.fetch(scoring, teams, season)
    if not board.height or not curve.height:
        return pl.DataFrame()

    board = board.filter(pl.col("position").is_in(list(FANTASY_POSITIONS)))
    board = (
        board.with_columns(
            pl.col("adp").rank("ordinal").over("position").cast(pl.Int32)
            .alias("adp_pos_rank")
        )
        .join(curve, on=["position", "adp_pos_rank"], how="left")
        .with_columns(ids.normalize("name").alias("_norm"))
    )

    if kept is not None and kept.height:
        keeper_keys = kept.select(
            ids.normalize("player_name").alias("_norm"), pl.col("owner").alias("kept_by")
        )
        board = board.join(keeper_keys, on="_norm", how="left").with_columns(
            pl.col("kept_by").is_not_null().alias("kept")
        )
    else:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("kept_by"),
            pl.lit(False).alias("kept"),
        )

    return board.drop("_norm").sort("adp")


def keeper_match_report(
    kept: pl.DataFrame, board: pl.DataFrame
) -> pl.DataFrame:
    """Keepers that did not match a row on the ADP board.

    A miss means a kept player is still sitting on the draft board, which is
    the one error in this module that would actively mislead on draft day. Most
    misses are benign — a keeper too deep to be in the top ~200 of ADP never
    appears on the board at all — so the report distinguishes nothing.

    Returns: owner, player_name, position (the unmatched keepers).
    """
    if not kept.height or not board.height:
        return pl.DataFrame()
    matched = board.filter(pl.col("kept")).select(
        ids.normalize("name").alias("_norm")
    )
    return (
        kept.with_columns(ids.normalize("player_name").alias("_norm"))
        .join(matched, on="_norm", how="anti")
        .drop("_norm")
        .sort(["position", "player_name"])
    )


def replacement(
    board: pl.DataFrame,
    summary: pl.DataFrame,
    value_col: str = "exp_points",
    use_draft_demand: bool = True,
) -> pl.DataFrame:
    """Replacement value per position, among players who can actually be drafted.

    With `use_draft_demand` (the default) the baseline is the first available
    player past `draft_demand` — league demand less what is already kept. Set
    it False to price against `league_demand` instead, which is the
    conventional baseline and the one appropriate to a league with no keepers.

    Returns: position, demand_used, replacement_rank, replacement_points.
    """
    if not board.height or not summary.height:
        return pl.DataFrame()

    col = "draft_demand" if use_draft_demand else "league_demand"
    rows = []
    for r in summary.iter_rows(named=True):
        position = r["position"]
        pool = (
            board.filter(
                (pl.col("position") == position)
                & ~pl.col("kept")
                & pl.col(value_col).is_not_null()
            )
            .sort(value_col, descending=True)
        )
        if not pool.height:
            continue
        idx = min(int(round(r[col])), pool.height - 1)
        rows.append(
            {
                "position": position,
                "demand_used": r[col],
                "replacement_rank": idx + 1,
                "replacement_points": float(pool.get_column(value_col)[idx]),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build(
    league_id: str | None = None,
    season: int = SEASON,
    use_draft_demand: bool = True,
    gap: float = ex.TIER_GAP_POINTS,
) -> dict[str, pl.DataFrame]:
    """The whole board, end to end.

    Returns a dict of frames rather than one wide table because the pieces are
    read at different moments — `summary` and `unmatched` are checked once
    before the draft, `players` is what sits open on the screen during it.

    Keys: kept, summary, unmatched, replacement, players.
    """
    kept = kept_players(league_id)
    summary = keeper_summary(kept, league_id)
    pool = draftable(kept, season=season)
    if not pool.height:
        return {
            "kept": kept, "summary": summary, "unmatched": pl.DataFrame(),
            "replacement": pl.DataFrame(), "players": pl.DataFrame(),
        }

    repl = replacement(pool, summary, use_draft_demand=use_draft_demand)
    players = (
        pool.filter(~pl.col("kept"))
        .join(repl.select("position", "replacement_points"), on="position", how="left")
        .with_columns(
            (pl.col("exp_points") - pl.col("replacement_points")).round(1).alias("par")
        )
    )
    # Tiers are cut on PAR, not raw points: the board's job is cross-position
    # ordering, and a tier should mean "these are interchangeable *picks*".
    players = ex.tiers(players, value_col="par", gap=gap)
    players = players.with_columns(
        pl.col("par").rank("ordinal", descending=True).cast(pl.Int32).alias("board_rank")
    ).sort("par", descending=True)

    return {
        "kept": kept,
        "summary": summary,
        "unmatched": keeper_match_report(kept, pool),
        "replacement": repl,
        "players": players,
    }


def picks(
    league_id: str | None = None,
    user_id: str | None = None,
    rounds: int = 15,
) -> pl.DataFrame:
    """Which picks you actually own, after trades and keeper placements.

    Three things have to compose correctly here, and getting any of them wrong
    silently produces a plausible but wrong pick list:

    1. **Snake order.** Odd rounds run 1..N, even rounds N..1, so a slot's pick
       number alternates rather than stepping by a constant.
    2. **Traded picks.** `roster_id` on a traded pick is whose pick it
       *originally* was, not who holds it — so a pick you acquired sits at the
       original owner's slot, not yours. Reading it the other way puts your
       picks in the wrong rounds entirely.
    3. **Keeper placements.** Once declared, the commissioner slots each keeper
       onto a specific pick, which consumes it. A keeper can land on a pick you
       acquired rather than your own.

    Returns: round, pick_no, from_owner, keeper, usable — one row per pick you
    hold, in draft order.
    """
    league_id = league_id or sc.resolve_league_id(None)
    if not league_id:
        return pl.DataFrame()

    draft = (sleeper._get(f"league/{league_id}/drafts") or [{}])[0]
    draft_id = draft.get("draft_id")
    order = draft.get("draft_order") or {}
    if not draft_id or not order:
        return pl.DataFrame()

    rosters = sleeper._get(f"league/{league_id}/rosters") or []
    users = {
        u["user_id"]: u.get("display_name")
        for u in (sleeper._get(f"league/{league_id}/users") or [])
    }
    rid_by_uid = {r.get("owner_id"): r.get("roster_id") for r in rosters}
    slot_by_rid = {
        rid_by_uid[uid]: slot for uid, slot in order.items() if uid in rid_by_uid
    }
    name_by_rid = {
        rid_by_uid[uid]: users.get(uid) for uid in order if uid in rid_by_uid
    }

    # Resolve by display name from the environment when no id is passed, which
    # keeps a personal handle out of the repo the same way the league id is.
    if user_id is None:
        user_id = next(
            (uid for uid in order if users.get(uid) == SLEEPER_USERNAME), None
        )
    me = rid_by_uid.get(user_id)
    if me is None:
        return pl.DataFrame()

    teams = len(slot_by_rid) or 10

    def pick_no(rnd: int, slot: int) -> int:
        return (rnd - 1) * teams + (slot if rnd % 2 == 1 else teams - slot + 1)

    owner = {(r, rid): rid for r in range(1, rounds + 1) for rid in slot_by_rid}
    for traded in (sleeper._get(f"draft/{draft_id}/traded_picks") or []):
        key = (traded.get("round"), traded.get("roster_id"))
        if key in owner:
            owner[key] = traded.get("owner_id")

    keeper_at: dict[int, tuple[int, str]] = {}
    for p in (sleeper._get(f"draft/{draft_id}/picks") or []):
        md = p.get("metadata") or {}
        keeper_at[p.get("pick_no")] = (
            p.get("roster_id"),
            f"{md.get('first_name', '')} {md.get('last_name', '')}".strip(),
        )

    rows = []
    for (rnd, original), holder in owner.items():
        if holder != me:
            continue
        pk = pick_no(rnd, slot_by_rid[original])
        kept = keeper_at.get(pk)
        rows.append(
            {
                "round": rnd,
                "pick_no": pk,
                "from_owner": name_by_rid.get(original),
                "keeper": kept[1] if kept else None,
                "usable": kept is None,
            }
        )
    return pl.DataFrame(rows).sort("pick_no") if rows else pl.DataFrame()


def targets(
    players: pl.DataFrame,
    pick_no: int,
    min_available: float = 0.35,
    top: int = 5,
) -> pl.DataFrame:
    """Best players by PAR who plausibly last until `pick_no`.

    Uses FFC's draft-slot dispersion through `adp.survival`, so this is "who is
    worth taking *and* likely to still be there" rather than a wish list. The
    probability is the honest half: two players at the same ADP with different
    dispersion are very different planning problems, and the one with a tight
    distribution is the one you cannot wait on.

    Needs the `stdev` column, which rides along from the FFC board through
    `draftable`. Returns empty rather than guessing if it is missing, since a
    survival curve invented from a default dispersion would look exactly as
    confident as a real one.
    """
    if not players.height or "stdev" not in players.columns:
        return pl.DataFrame()
    col = f"p_available_at_{pick_no}"
    return (
        adp_mod.survival(players, pick_no)
        .filter(pl.col(col) >= min_available)
        .sort("par", descending=True)
        .head(top)
        .select("name", "position", "adp", "par", "tier", col)
    )


def compare_baselines(league_id: str | None = None, season: int = SEASON) -> pl.DataFrame:
    """How much the keeper adjustment actually moves each position.

    The honesty check on this module's central claim. If pricing against the
    draft pool rather than the whole league barely changes the baseline, the
    complexity here is not earning its keep and should be deleted.

    Returns: position, league_replacement, draft_replacement, shift.
    """
    kept = kept_players(league_id)
    summary = keeper_summary(kept, league_id)
    pool = draftable(kept, season=season)
    if not pool.height:
        return pl.DataFrame()

    full = replacement(pool, summary, use_draft_demand=False).select(
        "position", pl.col("replacement_points").alias("league_replacement")
    )
    draft = replacement(pool, summary, use_draft_demand=True).select(
        "position", pl.col("replacement_points").alias("draft_replacement")
    )
    return (
        full.join(draft, on="position", how="inner")
        .with_columns(
            (pl.col("draft_replacement") - pl.col("league_replacement"))
            .round(1).alias("shift")
        )
        .sort("shift")
    )
