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

import numpy as np
import polars as pl

from src import adp as adp_mod
from src import expected as ex
from src import ids
from src import profiles as pf
from src import scoring as sc
from src import sleeper
from src.config import (
    FANTASY_POSITIONS,
    LEAGUE_ADP_TEAMS,
    SEASON,
    SLEEPER_USERNAME,
)
from src.profiles import LeagueProfile


def kept_players(
    league_id: str | None = None,
    force: bool = False,
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """Every player declared as a keeper, league-wide, from Sleeper.

    Reads the `keepers` field on each roster, which is populated once managers
    declare. A team that has not declared yet simply contributes no rows — and
    that absence is load-bearing information, so `keeper_summary` reports which
    teams are still outstanding rather than treating undeclared as none.

    A profile without keepers returns empty here, which is what makes the whole
    keeper layer downstream a no-op instead of a branch at every call site.

    Returns: owner, player_id, player_name, position, team.
    """
    profile = profile or pf.resolve()
    if not profile.keepers or not profile.sleeper_backed:
        return pl.DataFrame()

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
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """Per-position: how many are kept, and how many slots the draft must fill.

    `draft_demand` is what the board prices against. `undeclared_teams` is
    carried on every row because an undeclared team is unfilled demand hiding
    in the numbers — with two keepers outstanding the true draft demand is
    somewhere between `draft_demand` and `draft_demand` minus two, and a board
    that hid that would be overconfident about scarcity.

    Returns: position, league_demand, kept, draft_demand, undeclared_teams.
    """
    profile = profile or pf.resolve()
    kept = kept if kept is not None else kept_players(league_id, profile=profile)
    # A synthetic profile describes a format nobody has a Sleeper league for, so
    # asking Sleeper what the settings are would return the wrong league's.
    settings = (
        sc.league_settings(league_id)
        if profile.sleeper_backed
        else pf.as_settings(profile)
    )
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
    scoring: str | None = None,
    teams: int | None = None,
    curve: pl.DataFrame | None = None,
    profile: LeagueProfile | None = None,
    drafted: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """This season's ADP board with keepers removed and expectations attached.

    The market comes from the profile, and for the Shiva Bowl that is the **2QB**
    board rather than the league's half-PPR/10 one. In a superflex league the
    1QB market misprices quarterbacks by a round or more, and since this is the
    board you draft off, it has to be the market that matches the format. That
    pairing is exactly what `profiles.LeagueProfile` exists to keep together.

    Name-matched against Sleeper, since the two sources share no ids. Matching
    is on `ids.normalize`, and `keeper_match_report` says how many keepers
    failed to match so a silent miss cannot quietly leave a kept player on the
    board.

    Returns the `expected.expected_points` columns plus `kept` and `kept_by`.
    """
    profile = profile or pf.resolve()
    scoring = scoring or profile.adp_scoring
    teams = teams or profile.adp_teams
    curve = curve if curve is not None else ex.adp_curve(profile=profile)
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

    # Live picks, kept separate from keepers rather than folded into `kept`.
    # They mean different things: a keeper reduces the demand the draft has to
    # fill, while a player taken on the clock consumes demand that was always
    # going to be filled. Merging them would quietly move replacement level
    # every time somebody picked.
    if drafted is not None and drafted.height:
        gone = (
            drafted.select(ids.normalize("player_name").alias("_norm"), "pick_no")
            .unique(subset=["_norm"], keep="first")
            .rename({"pick_no": "drafted_at"})
        )
        board = board.join(gone, on="_norm", how="left").with_columns(
            pl.col("drafted_at").is_not_null().alias("drafted")
        )
    else:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("drafted_at"),
            pl.lit(False).alias("drafted"),
        )

    return board.drop("_norm").sort("adp")


def drafted_players(league_id: str | None = None) -> pl.DataFrame:
    """Everyone already selected, read live while the draft runs.

    **The board is otherwise frozen at its pre-draft state.** `kept_players`
    reads the roster `keepers` field, which is a static declaration made weeks
    earlier and never changes once the draft starts. Nothing else removed a
    player from the pool, so by the third round the board would still be
    offering players taken in the first — useful for one pick and misleading
    for the rest of the night.

    Sleeper publishes every pick as it happens on the same endpoint the keeper
    placements come from, distinguished by `is_keeper`. Keepers are excluded
    here because `kept_players` already accounts for them and counting a player
    twice would double-subtract him from positional demand.

    Names come from the pick metadata rather than the player file, matching how
    `picks` reads the same rows, because the metadata is present on every pick
    and needs no second request on the clock.

    Returns: pick_no, round, player_name, picked_by — empty before the draft
    opens, which makes every caller's pre-draft behaviour unchanged.
    """
    league_id = league_id or sc.resolve_league_id(None)
    if not league_id:
        return pl.DataFrame()

    draft = (sleeper._get(f"league/{league_id}/drafts") or [{}])[0]
    draft_id = draft.get("draft_id")
    if not draft_id:
        return pl.DataFrame()

    rows = []
    for p in (sleeper._get(f"draft/{draft_id}/picks") or []):
        if p.get("is_keeper"):
            continue
        md = p.get("metadata") or {}
        name = f"{md.get('first_name', '')} {md.get('last_name', '')}".strip()
        if not name:
            continue
        rows.append(
            {
                "pick_no": p.get("pick_no"),
                "round": p.get("round"),
                "player_name": name,
                "picked_by": p.get("picked_by"),
            }
        )
    return pl.DataFrame(rows).sort("pick_no") if rows else pl.DataFrame()


def keeper_slots(league_id: str | None = None, rounds: int = 15) -> list[int]:
    """Absolute pick numbers consumed by a keeper, league-wide.

    A declared keeper is slotted onto a specific pick by the commissioner, and
    that pick is then spent — nobody selects with it. So the draft has fewer
    real selections than it has picks, and the two numbering systems come apart.
    `keeper_adjusted_adp` needs this to translate between them.

    Returns an empty list when there is no league or nothing is declared, which
    makes every caller degrade to plain ADP rather than branch.
    """
    league_id = league_id or sc.resolve_league_id(None)
    if not league_id:
        return []
    draft = (sleeper._get(f"league/{league_id}/drafts") or [{}])[0]
    draft_id = draft.get("draft_id")
    if not draft_id:
        return []
    return sorted(
        p["pick_no"]
        for p in (sleeper._get(f"draft/{draft_id}/picks") or [])
        if p.get("pick_no")
    )


def keeper_adjusted_adp(
    board: pl.DataFrame,
    slots: list[int] | None = None,
    teams: int = LEAGUE_ADP_TEAMS,
    rounds: int = 15,
) -> pl.DataFrame:
    """Where players actually go once the keepers are off the board.

    Public ADP is priced in redraft leagues where nobody is kept. Every keeper
    in this league has an ADP inside the top 150, so each one is a player the
    market expects to be drafted who will not be — and everyone behind him moves
    up. **Any mock draft that does not model keepers is showing you players
    roughly fifteen picks later than they will actually go**, which is the
    difference between planning to take someone at 44 and watching him leave at
    30.

    Two columns, because keepers do two different things and conflating them is
    the easy mistake:

    `adj_adp` is his **selection index** — the count of players drafted before
    him once keepers are removed from the pool. This is the pure removal effect
    and it is what makes Barkley the 13th player taken rather than the 26th.

    `exp_pick` is the **absolute pick number** that selection index lands on.
    These differ because a keeper does not merely vanish from the pool, he also
    consumes a pick: the draft is 150 picks but only 132 selections. Comparing
    `adj_adp` against a pick number from `picks()` would therefore count the
    keeper adjustment twice, in the wrong direction. Compare `exp_pick` against
    your picks; read `adj_adp` when you want to know how the pool reordered.

    With no keepers — every synthetic profile, and this league before anyone
    declares — both columns equal `adp` and the function is a documented no-op.

    Returns the board with `adj_adp`, `exp_pick`, and `adp_shift` added.
    """
    if not board.height or "adp" not in board.columns:
        return board

    kept_adp = (
        board.filter(pl.col("kept")).get_column("adp").drop_nulls().sort().to_list()
        if "kept" in board.columns
        else []
    )

    if not kept_adp:
        return board.with_columns(
            pl.col("adp").alias("adj_adp"),
            pl.col("adp").alias("exp_pick"),
            pl.lit(0.0).alias("adp_shift"),
        )

    # Selection index: subtract the keepers priced ahead of him. searchsorted on
    # a sorted list is the same count as a self-join and does not blow up on a
    # board this size.
    adp = board.get_column("adp").to_numpy()
    ahead = np.searchsorted(np.asarray(kept_adp), adp, side="left")
    adj = np.maximum(adp - ahead, 1.0)

    # The pick numbers still available to select with, in order. The s-th
    # selection happens at the s-th of these, interpolated because ADP is
    # fractional and rounding it here would quietly discard the sub-pick
    # precision that makes two players at the same rank distinguishable.
    total = teams * rounds
    consumed = set(slots or [])
    open_picks = np.array(
        [n for n in range(1, total + 1) if n not in consumed], dtype=float
    )
    if open_picks.size:
        idx = np.clip(adj - 1.0, 0.0, float(open_picks.size - 1))
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, open_picks.size - 1)
        exp_pick = open_picks[lo] + (idx - lo) * (open_picks[hi] - open_picks[lo])
    else:
        exp_pick = adj

    return board.with_columns(
        pl.Series("adj_adp", np.round(adj, 1)),
        pl.Series("exp_pick", np.round(exp_pick, 1)),
        pl.Series("adp_shift", np.round(adj - adp, 1)),
    )


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
    profile: LeagueProfile | None = None,
) -> dict[str, pl.DataFrame]:
    """The whole board, end to end, for one league profile.

    Returns a dict of frames rather than one wide table because the pieces are
    read at different moments — `summary` and `unmatched` are checked once
    before the draft, `players` is what sits open on the screen during it.

    `warnings` carries the reason a frame came back empty. A board that cannot
    be priced should say so in a sentence; an empty DataFrame looks identical to
    a network blip and invites the caller to retry forever.

    Keys: profile, warnings, kept, summary, unmatched, replacement, players.
    """
    profile = profile or pf.resolve()
    warnings: list[str] = []

    kept = kept_players(league_id, profile=profile)
    summary = keeper_summary(kept, league_id, profile=profile)
    gone = drafted_players(league_id) if profile.sleeper_backed else pl.DataFrame()
    pool = draftable(kept, season=season, profile=profile, drafted=gone)
    if not pool.height:
        warnings.append(
            f"No {profile.adp_scoring} ADP board for {season} at "
            f"{profile.adp_teams} teams, or no expected-points curve to attach "
            "to it. FFC suppresses a format until it has collected enough "
            "drafts, so a thin market reads as an empty one."
        )
        return {
            "profile": profile, "warnings": warnings, "drafted": gone,
            "kept": kept, "summary": summary, "unmatched": pl.DataFrame(),
            "replacement": pl.DataFrame(), "players": pl.DataFrame(),
        }

    pool = keeper_adjusted_adp(
        pool, slots=keeper_slots(league_id) if profile.keepers else None,
        teams=profile.teams,
    )
    # Replacement is computed *before* live picks are removed, deliberately.
    # PAR is a valuation, and a valuation should not move because a leaguemate
    # picked: excluding drafted players from the baseline pool while leaving
    # demand fixed would point the baseline steadily deeper into what remains
    # and inflate everyone's PAR as the night went on. Drafted players leave the
    # board below; they stay in the pool that sets the baseline.
    repl = replacement(pool, summary, use_draft_demand=use_draft_demand)
    players = (
        pool.filter(~pl.col("kept") & ~pl.col("drafted"))
        .join(repl.select("position", "replacement_points"), on="position", how="left")
        .with_columns(
            (pl.col("exp_points") - pl.col("replacement_points")).round(1).alias("par")
        )
    )
    # Tiers are cut on PAR, not raw points: the board's job is cross-position
    # ordering, and a tier should mean "these are interchangeable *picks*".
    players = ex.tiers(players, value_col="par", gap=gap)
    # Tiers cut on a fixed 7-point gap; this cuts on the curve's own uncertainty.
    # They are different questions — "is this the same asset" against "can we
    # even tell these two apart" — and the board needs both, because a tier break
    # that falls inside the standard error is a break the data did not earn.
    players = indistinguishable(players, value_col="par")
    players = players.with_columns(
        pl.col("par").rank("ordinal", descending=True).cast(pl.Int32).alias("board_rank")
    ).sort("par", descending=True)

    return {
        "profile": profile,
        "warnings": warnings,
        "drafted": gone,
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

    **Survival is measured against `exp_pick`, not raw ADP, whenever the board
    carries it.** `pick_no` is a real pick in a draft where keepers have already
    consumed slots and left the pool; public ADP is priced in redraft leagues
    where neither is true. Comparing the two directly reports players as
    available who will be gone — the error runs to 45x on the players it matters
    for. Falls back to `adp` for a board with no keeper adjustment, which is the
    correct baseline rather than a degraded one.
    """
    if not players.height or "stdev" not in players.columns:
        return pl.DataFrame()
    col = f"p_available_at_{pick_no}"
    adp_col = "exp_pick" if "exp_pick" in players.columns else "adp"
    shown = ["name", "position", "adp"]
    if adp_col == "exp_pick":
        shown.append("exp_pick")
    shown += ["par", "tier", col]

    return (
        adp_mod.survival(players, pick_no, adp_col=adp_col)
        .filter(pl.col(col) >= min_available)
        .sort("par", descending=True)
        .head(top)
        .select(shown)
    )


def cost_of_waiting(
    players: pl.DataFrame,
    picks: list[int],
    positions: tuple[str, ...] = FANTASY_POSITIONS,
) -> pl.DataFrame:
    """What it costs to wait a round, per position, at your own picks.

    The board says what a player is worth. It cannot say what waiting costs,
    and that is the question actually being asked on the clock — take the
    quarterback now or take him next round. Answering it needs three things
    nothing else here combines: the *shape* of the positional curve rather than
    its level, the draft-slot dispersion FFC ships beside ADP, and your real
    pick list.

    For each pick, the expected PAR of the **best player of that position still
    on the board**. Walking a position from the top, a player supplies the
    answer when he is still there and everyone above him is not:

        E[best] = sum_i  par_i * P(i available) * prod_{j<i} P(j gone)

    which is exact under the independence the survival model already assumes,
    rather than a simulation of it.

    **Availability is measured on `exp_pick` where the board carries it**, for
    the same reason `targets` is: a raw ADP against a real pick number overstates
    who is left in a keeper league.

    Note what the output is *not*. A large drop is not an instruction to draft
    that position — a position can be expensive to wait on and still be worth
    less than another. It is the opportunity cost of the wait, to be read next
    to PAR, not instead of it.

    Returns: position, pick_no, best_par, cost_of_waiting — where the cost on
    each row is what falls away between that pick and the next one you own, and
    is null on the last pick because there is nothing after it to wait for.
    """
    if not players.height or not picks or "stdev" not in players.columns:
        return pl.DataFrame()

    adp_col = "exp_pick" if "exp_pick" in players.columns else "adp"
    ordered = sorted(picks)
    rows: list[dict[str, object]] = []

    for position in positions:
        sub = (
            players.filter(
                (pl.col("position") == position)
                & pl.col("par").is_not_null()
                & pl.col(adp_col).is_not_null()
            )
            .sort("par", descending=True)
        )
        if not sub.height:
            continue

        for pick_no in ordered:
            surv = adp_mod.survival(sub, pick_no, adp_col=adp_col)
            col = f"p_available_at_{pick_no}"
            still_gone = 1.0
            expected = 0.0
            for par, p_here in zip(
                surv.get_column("par"), surv.get_column(col)
            ):
                p = float(p_here or 0.0)
                expected += float(par) * p * still_gone
                still_gone *= 1.0 - p
            rows.append(
                {
                    "position": position,
                    "pick_no": pick_no,
                    "best_par": round(expected, 1),
                }
            )

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .sort(["position", "pick_no"])
        .with_columns(
            (
                pl.col("best_par") - pl.col("best_par").shift(-1).over("position")
            ).round(1).alias("cost_of_waiting")
        )
    )


# Measured over 128 team-seasons (2022-2025): team skill-position fantasy
# points regressed on mean implied team total. Not a fitted feature — a slope
# between two things already measured, reported so the size of a context
# argument can be checked rather than asserted.
POINTS_PER_IMPLIED = 45.3
LEAGUE_MEAN_IMPLIED = 22.0
MEAN_TEAM_FF = 947.0


def attach_environment(
    players: pl.DataFrame, season: int = SEASON
) -> pl.DataFrame:
    """Add each player's team scoring environment, and what it is worth.

    `env_swing` estimates the season-long points a player gains or loses purely
    from the offence he plays in, relative to a league-average one: the
    measured slope above, times how far his team's implied total sits from the
    league mean, times his share of a typical offence.

    **This is not double counting, and the reason is worth stating.** The
    expected-points curve maps *positional ADP rank* to points, so it is blind
    to team — it assigns this year's TE1 whatever the historical average TE1
    scored, whether he plays for the best offence in football or the worst. The
    market's view of the team shows up in the ADP *level* (a good player on a
    bad offence is drafted later) but not in his positional rank. So the
    adjustment adds information the curve genuinely does not have.

    It is still an upper bound. Some of the discount is already in the price,
    and `env_swing` does not know how much, so read it as "the size of the
    argument" rather than a correction to subtract. A player whose PAR edge is
    small next to his `env_swing` gap is one where context should decide.

    Team codes are normalized on both sides — FFC says LAR where nflverse says
    LA, and an unnormalized join silently drops those players to null rather
    than failing.
    """
    if not players.height:
        return players

    env = ex.preseason_environment(season)
    if not env.height:
        return players.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("team_implied"),
            pl.lit(None, dtype=pl.Float64).alias("env_z"),
            pl.lit(None, dtype=pl.Float64).alias("env_swing"),
        )

    env = env.with_columns(ids.normalize_team("team")).select(
        "team", pl.col("early_implied").alias("team_implied"), "env_z"
    )
    return (
        players.with_columns(ids.normalize_team("team"))
        .join(env, on="team", how="left")
        .with_columns(
            (
                POINTS_PER_IMPLIED
                * (pl.col("team_implied") - LEAGUE_MEAN_IMPLIED)
                * (pl.col("exp_points") / MEAN_TEAM_FF)
            ).round(1).alias("env_swing")
        )
    )


def attach_quality(
    players: pl.DataFrame,
    valued: pl.DataFrame | None = None,
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """Add a second opinion that is not derived from ADP.

    **Why the board needs one.** `par` is honest but it is a function of price:
    `exp_points` maps a player's *positional ADP rank* to what players at that
    rank have historically scored, so within a position the board reproduces the
    market's ordering exactly. Two receivers eleven picks apart have different
    PAR for one reason — eleven picks. That is the correct answer to "what is
    this draft slot worth" and no answer at all to "how good is this player".

    `quality_pct` is the independent read: a percentile, within position, of a
    stability-weighted blend of per-opportunity metrics — weighted by how much
    each one actually repeats year over year, so a metric that is mostly noise
    contributes almost nothing. `value_gap` is that percentile minus the
    percentile of his draft price, which makes it a **disagreement score**: it
    says where this project's read differs from the market's, which is a reason
    to look closer rather than a reason to be right.

    Two limits that must stay visible, because both produce nulls rather than
    errors:

    - **Quarterbacks are scored, but more thinly than everyone else.** They were
      excluded until 2026-08-10; see `valuation.VALUED_POSITIONS` for the
      measurement that switched them on and the two caveats that travel with any
      QB number — three quality features rather than eight or ten, weighted into
      what is largely a rushing read, and a recent window that does not clear
      zero on its own. `path_score` is null at QB by design: its terms are about
      earning targets, which is not a quarterback's route to volume.
    - **Thin seasons are dropped.** Under 8 games, or under the position's volume
      floor (100 routes; 150 pass attempts at QB), a per-opportunity metric is a
      rumour rather than a measurement. A null here means "not measured", never
      "bad".

    Returns `players` with `quality_pct`, `market_pct`, `value_gap` and
    `path_score` added, left-joined so unscored players survive as nulls.
    """
    if not players.height:
        return players

    profile = profile or pf.resolve()
    if valued is None:
        from src import valuation as val  # local: valuation imports features

        valued = val.board(profile=profile)

    cols = ["quality_pct", "market_pct", "value_gap", "path_score"]
    if not valued.height:
        return players.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in cols]
        )

    keys = valued.select(
        ids.normalize("name").alias("_norm"),
        *[pl.col(c) for c in cols if c in valued.columns],
    ).unique(subset=["_norm"], keep="first")

    return (
        players.with_columns(ids.normalize("name").alias("_norm"))
        .join(keys, on="_norm", how="left")
        .drop("_norm")
    )


def indistinguishable(players: pl.DataFrame, value_col: str = "par") -> pl.DataFrame:
    """Group players the expected-points curve cannot actually tell apart.

    **The problem this exists for.** `par` is printed to a tenth of a point and
    the curve it comes from carries a standard error of 6 to 13. At the top of a
    position those two facts collide: Puka Nacua and Jaxon Smith-Njigba differ by
    10.8 points of PAR against standard errors of 10.8 and 9.0, so the board
    displays an ordering it has no evidence for. Smith-Njigba was the WR1 in 2025
    and outscored Nacua by 30 points; he sits lower here purely because he is
    drafted 2.4 picks later.

    Two players are called indistinguishable when the gap between them is inside
    the pooled standard error of the pair, `sqrt(se_a**2 + se_b**2)`.

    **Compared against the group leader, not the previous player** — the same
    choice `expected.tiers` makes and for the same reason. Single-linkage down a
    shallow slope chains every player at the position into one group: each is
    within a standard error of the man above him, and forty of them are not
    thereby interchangeable. Comparing against the leader asks "is this the same
    asset as the best one in the group", which is the question a drafter has.

    Adds `indist_group` (1-based within position) and `indist_n` (how many
    players share the group). A player alone in his group has `indist_n == 1`,
    which is the board saying it can genuinely separate him from his neighbours.
    """
    if not players.height or value_col not in players.columns:
        return players
    if "se" not in players.columns:
        return players.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("indist_group"),
            pl.lit(None, dtype=pl.UInt32).alias("indist_n"),
        )

    parts: list[pl.DataFrame] = []
    for position in players.get_column("position").unique().sort().to_list():
        sub = players.filter(pl.col("position") == position).sort(
            value_col, descending=True, nulls_last=True
        )
        values = sub.get_column(value_col).to_list()
        errors = sub.get_column("se").to_list()

        groups: list[int | None] = []
        group = 1
        lead_value: float | None = None
        lead_error: float = 0.0
        for value, error in zip(values, errors):
            if value is None:
                # Unscored players form no group — a null PAR is not evidence of
                # similarity to anything.
                groups.append(None)
                continue
            error = float(error) if error is not None else 0.0
            if lead_value is None:
                lead_value, lead_error = float(value), error
                groups.append(group)
                continue
            pooled = float(np.hypot(lead_error, error))
            if lead_value - float(value) > pooled:
                group += 1
                lead_value, lead_error = float(value), error
            groups.append(group)

        parts.append(sub.with_columns(pl.Series("indist_group", groups, dtype=pl.Int32)))

    if not parts:
        return players
    out = pl.concat(parts, how="diagonal_relaxed")
    return out.with_columns(
        pl.len().over(["position", "indist_group"]).cast(pl.UInt32).alias("indist_n")
    )


def tier_map(players: pl.DataFrame, limit: int = 6) -> pl.DataFrame:
    """How many of each asset are left, by position and tier.

    The board sorted by PAR answers "who is next". On the clock the more useful
    question is "how many more of this are there" — reaching for the last player
    in a tier is worth something, reaching for the fourth of nine is not. This is
    the same tier assignment already on the board, counted rather than listed.

    Counts only players still available: `build` removes kept and drafted players
    before this runs, so the numbers fall as the draft goes and a tier that empties
    disappears rather than lingering at zero.

    Returns: position, tier, n_left, par_top, par_bottom, best_available — sorted
    by position then tier, capped at `limit` tiers per position because tier 9 at
    running back is not a draft-day input.
    """
    if not players.height or "tier" not in players.columns:
        return pl.DataFrame()

    ranked = (
        players.filter(pl.col("par").is_not_null())
        .sort("par", descending=True, nulls_last=True)
        .group_by(["position", "tier"])
        .agg(
            pl.len().alias("n_left"),
            pl.col("par").max().alias("par_top"),
            pl.col("par").min().alias("par_bottom"),
            pl.col("name").first().alias("best_available"),
        )
        .sort(["position", "tier"])
    )
    if not ranked.height:
        return pl.DataFrame()
    return (
        ranked.with_columns(
            pl.col("tier").rank("ordinal").over("position").alias("_ord")
        )
        .filter(pl.col("_ord") <= limit)
        .drop("_ord")
    )


def context_flags(players: pl.DataFrame, margin: float = 1.0) -> pl.DataFrame:
    """Players whose team environment outweighs their edge on the board.

    The pairs worth a second thought: someone ranked above a positional rival
    by less than the environment gap between them. Those are the picks where
    the board is not really the deciding input, and pretending otherwise is how
    a defensible ranking produces an indefensible pick.

    Returns: position, better_player, worse_player, par_edge, env_edge, verdict.
    """
    if not players.height or "env_swing" not in players.columns:
        return pl.DataFrame()

    rows = []
    for position in players.get_column("position").unique().to_list():
        sub = (
            players.filter(
                (pl.col("position") == position)
                & pl.col("env_swing").is_not_null()
                & pl.col("par").is_not_null()
            )
            .sort("par", descending=True)
            .head(6)
        )
        seen = sub.iter_rows(named=True)
        ranked = list(seen)
        for i, better in enumerate(ranked):
            for worse in ranked[i + 1:]:
                par_edge = better["par"] - worse["par"]
                env_edge = worse["env_swing"] - better["env_swing"]
                if env_edge > par_edge + margin:
                    rows.append(
                        {
                            "position": position,
                            "better_player": better["name"],
                            "worse_player": worse["name"],
                            "par_edge": round(par_edge, 1),
                            "env_edge": round(env_edge, 1),
                            "verdict": "context outweighs the board edge",
                        }
                    )
    return (
        pl.DataFrame(rows).sort("env_edge", descending=True)
        if rows
        else pl.DataFrame()
    )


def compare_baselines(
    league_id: str | None = None,
    season: int = SEASON,
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """How much the keeper adjustment actually moves each position.

    The honesty check on this module's central claim. If pricing against the
    draft pool rather than the whole league barely changes the baseline, the
    complexity here is not earning its keep and should be deleted.

    Returns: position, league_replacement, draft_replacement, shift.
    """
    profile = profile or pf.resolve()
    kept = kept_players(league_id, profile=profile)
    summary = keeper_summary(kept, league_id, profile=profile)
    pool = draftable(kept, season=season, profile=profile)
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
