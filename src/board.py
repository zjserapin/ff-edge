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
from src import footballers as ffb
from src import ids
from src import profiles as pf
from src import scoring as sc
from src import sleeper
from src.config import (
    ECR_PAGE_STANDARD,
    ECR_PAGE_SUPERFLEX,
    ECR_WEIGHT,
    ENV_WEIGHT,
    FANTASY_POSITIONS,
    FLEX_SLOTS,
    FOOTBALLERS_MIN_ANALYSTS,
    FOOTBALLERS_WEIGHT,
    LEAGUE_ADP_TEAMS,
    NON_STARTING_SLOTS,
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


def attach_footballers(
    pool: pl.DataFrame,
    summary: pl.DataFrame,
    min_analysts: int = FOOTBALLERS_MIN_ANALYSTS,
    use_draft_demand: bool = True,
    scoring: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Blend the Fantasy Footballers' projections into the board, on PAR.

    **Why this blends PAR and not rank.** Averaging two rank orderings throws
    away how *far apart* the players are, which is the only thing the board is
    actually for — a tier is a statement about distance, not order. Both sources
    here produce a season point total, so they can be blended as numbers instead.

    **Why each side gets its own replacement level, and why that is the load-
    bearing decision in this function.** `exp_points` comes from the ADP curve,
    which is fitted on *realized* seasons and therefore has injuries, benchings
    and missed games already baked into it. A projection does not: analysts
    project the season a player has if things go normally, so their totals run
    systematically higher than the curve's. Subtract a common baseline and that
    level difference lands entirely on the blend, quietly tilting the board
    toward whichever source is more optimistic.

    Subtracting *each system's own* replacement level removes it. `ffb_par` is
    "points above the replacement player as the Footballers see him"; `par` is
    the same sentence about the ADP curve. A uniform +12% on every projection
    lifts their replacement by 12% too and cancels. What survives is the only
    thing worth blending: disagreement about the *shape* of the position.

    The Footballers' baseline is computed over the same draftable pool and the
    same `draft_demand` as the board's, so the two are answering the identical
    question about the identical league.

    Adds `ffb_par`, `blend_par`, and everything `footballers.attach` brings.
    Players the panel has not published survive as nulls and fall back to their
    own `par` in `blend_par` — never dropped, and never silently blended against
    a zero, which would rank an unpublished player as though three analysts had
    projected him at replacement level.
    """
    pool = ffb.attach(pool, scoring=scoring)

    if "ffb_points" not in pool.columns:
        return pool.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ffb_par"),
            pl.lit(None, dtype=pl.Float64).alias("blend_par"),
        )

    # A thin panel is not a consensus. Blanked here rather than filtered so the
    # player keeps his row and his own par.
    pool = pool.with_columns(
        pl.when(pl.col("n_analysts") >= min_analysts)
        .then(pl.col("ffb_points"))
        .otherwise(None)
        .alias("ffb_points")
    )

    repl_ffb = replacement(
        pool, summary, value_col="ffb_points", use_draft_demand=use_draft_demand
    )
    if not repl_ffb.height:
        return pool.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ffb_par"),
            pl.lit(None, dtype=pl.Float64).alias("blend_par"),
        )

    return pool.join(
        repl_ffb.select(
            "position", pl.col("replacement_points").alias("_ffb_repl")
        ),
        on="position",
        how="left",
    ).with_columns(
        (pl.col("ffb_points") - pl.col("_ffb_repl")).round(1).alias("ffb_par")
    ).drop("_ffb_repl")


# Below this many blendable rows there is no scale to standardize on and the
# blend degrades to `par`. Per position the same floor guards the *center*: a
# median over three players is noise, and a wrong per-position center is worse
# than the shared one it would replace.
_BLEND_MIN_ROWS = 4
_BLEND_MIN_POSITION_ROWS = 4


def blend_par(
    players: pl.DataFrame, weight: float = FOOTBALLERS_WEIGHT
) -> pl.DataFrame:
    """Blend `par` and `ffb_par` on a common scale, falling back to `par` alone.

    Kept separate from `attach_footballers` because `ffb_par` needs the pool
    *before* keepers and drafted players are removed (that is where replacement
    is set) while the blend needs `par`, which only exists after. Splitting them
    is what lets both be computed against the right frame.

    **Standardized first, and skipping that step is a silent bug.** Both columns
    are in league points and both are above their own replacement level, which
    makes them look directly averageable. They are not: measured on the 2026
    board, `ffb_par` carries **1.76x the dispersion** of `par` (sd 60.6 against
    34.4), because the ADP curve maps a whole positional rank to one number and
    flattens hard at the top — the top four backs all price at exactly 72.6 —
    while three analysts projecting touches spread them from 84 to 157.

    Averaged raw, a nominal weight of 0.5 hands the Footballers about 64% of the
    variance in the result. Worse, the ratio is not constant across positions:
    2.2x at QB, 1.7x at RB, 1.55x at WR. So the raw blend tilts hardest at
    quarterback, which in a superflex league is the position the board is most
    sensitive about and the last one you would choose to distort by accident.

    So each side is standardized over the players where **both** exist — a
    shared row set, or the two standardizations describe different populations —
    and the blend is mapped back onto the `par` scale so the output stays
    readable as points above replacement. The weight then means what it says:
    0.5 is an equal say.

    **Median and IQR rather than mean and standard deviation**, because the two
    sources are censored differently in the tail. A backup quarterback is
    genuinely 270 points below a superflex replacement level and the Footballers
    say so; the ADP curve cannot, because it maps positional rank onto a fitted
    curve that floors out around -30. That is one real tail and one clipped one,
    and it moves the measured scale ratio a long way: 1.76 by standard deviation,
    1.55 by IQR, 1.43 with the dozen deepest quarterbacks dropped. A standard
    deviation set by twelve players nobody will draft is the wrong number to
    calibrate the whole board's blend on, so the scale is taken from the body of
    the distribution instead.

    **Centered per position, scaled globally, and the split between those two is
    the whole correction.** An earlier version centered globally as well, on the
    argument that PAR's only job is cross-position comparison and that
    standardizing within position would rescale every position to the same spread
    — asserting the best tight end is worth the best quarterback. That argument is
    right about the *spread* and was wrongly extended to the *center*. The two are
    separable, and conflating them let this function decide where a whole position
    sits, which is not a thing three analysts' projections should be able to do.

    Measured on the 2026 board under global centering, the median `ffb_par` minus
    `par` ran **+8.7 at tight end against -16.2 at quarterback**, -9.2 at running
    back and -6.6 at receiver. A single global center leaves that offset intact,
    so it lands on the blend as a uniform per-position shift — and a shift applied
    uniformly to a whole position is by construction not an opinion about players.

    The offset survives `attach_footballers` already subtracting each system's own
    replacement level, because the two replacement levels sit at different points
    on differently shaped curves and the board holds each position to a different
    depth: 18 tight ends against 67 receivers, cut at different distances from
    their own replacement. So it is a pool-composition artifact rather than a
    considered cross-position opinion, and it does not belong in the ordering.

    So: the **center** comes from each position's own median, which hands the
    cross-position level to `par` alone — the one thing `par` is documented to be
    good at, and its only non-tautological content. The **scale** stays global, so
    a position that spreads widely in both sources still spreads widely here and
    nothing is rescaled to a common width. The original spread argument survives
    untouched; only the center moved.

    At `weight=1.0` the Footballers now set the entire ordering *within* each
    position and none of the ordering *between* them. `weight=0.0` reproduces
    `par` exactly, as it did before — the center cancels at zero, so that identity
    is a control on the arithmetic here rather than evidence about the bias.

    **What this does not fix, and the measurement is the point of saying so.**
    On the 2026 board the correction moved the median quarterback six places
    (-18.0 to -12.0 against ADP) and the median tight end **not at all**: +47.5
    before and +47.5 after.

    The tight-end promotion is therefore not this function's doing. Ranking on
    raw `par` alone already shifts tight ends **+47.0** places, and every layer
    downstream inherits it — `blend_par` +50.0, `par_env` +53.0, the final board
    +47.5. It is PAR comparing positions the board holds to unequal depth: 67
    receivers against 18 tight ends, so the deep receivers sit far below receiver
    replacement while the deep tight ends sit barely below theirs, and a
    cross-position sort on points-over-replacement interleaves them accordingly.

    That is a replacement-and-roster-demand question, not a blending one. **Do
    not re-tune this weight trying to cancel it** — doing so would trade a real
    correction for a cosmetic one and hide the cause. The fix belongs where the
    pool is cut.

    A position with fewer than `_BLEND_MIN_POSITION_ROWS` blendable players falls
    back to the global center: a median taken over two players is not a level, and
    a wrong center is worse than a shared one.
    """
    if "ffb_par" not in players.columns or "par" not in players.columns:
        return players

    both = players.filter(
        pl.col("par").is_not_null() & pl.col("ffb_par").is_not_null()
    )
    if both.height < _BLEND_MIN_ROWS:
        return players.with_columns(
            pl.col("par").alias("blend_par"),
            pl.col("par").alias("blend_par_exact"),
        )

    par_mid = float(both.get_column("par").median())
    ffb_mid = float(both.get_column("ffb_par").median())

    # Frames without a position column blend as one pool, which is the old
    # behaviour and the right answer when there is nothing to be biased between.
    if "position" not in players.columns:
        centers = None
    else:
        centers = (
            both.group_by("position")
            .agg(
                pl.len().alias("_n"),
                pl.col("par").median().alias("_par_mid"),
                pl.col("ffb_par").median().alias("_ffb_mid"),
            )
            .with_columns(
                pl.when(pl.col("_n") >= _BLEND_MIN_POSITION_ROWS)
                .then(pl.col("_par_mid"))
                .otherwise(pl.lit(par_mid))
                .alias("_par_mid"),
                pl.when(pl.col("_n") >= _BLEND_MIN_POSITION_ROWS)
                .then(pl.col("_ffb_mid"))
                .otherwise(pl.lit(ffb_mid))
                .alias("_ffb_mid"),
            )
            .select("position", "_par_mid", "_ffb_mid")
        )

    def _with_centers(frame: pl.DataFrame) -> pl.DataFrame:
        if centers is None:
            return frame.with_columns(
                pl.lit(par_mid).alias("_par_mid"), pl.lit(ffb_mid).alias("_ffb_mid")
            )
        # A position with no blendable rows never reaches `centers`, so the join
        # leaves nulls rather than dropping him. Shared center for those.
        return frame.join(centers, on="position", how="left").with_columns(
            pl.col("_par_mid").fill_null(par_mid),
            pl.col("_ffb_mid").fill_null(ffb_mid),
        )

    # **The scale is taken from position-centered residuals, not from the raw
    # columns, and that is not a refinement.** A raw global IQR is inflated by
    # exactly the between-position level differences this function exists to
    # remove, so a source that sits high at one position would shrink its own
    # z-scores everywhere — the bias leaking back in through the denominator
    # after being removed from the numerator. Pooling the residuals measures
    # within-position dispersion, which is the only thing the weight should be
    # trading off, and makes the whole blend invariant to a per-position offset.
    centered = _with_centers(both)

    def _iqr(value: str, center: str) -> float:
        residual = centered.get_column(value) - centered.get_column(center)
        return float(residual.quantile(0.75) - residual.quantile(0.25))

    par_scale = _iqr("par", "_par_mid")
    ffb_scale = _iqr("ffb_par", "_ffb_mid")

    # A degenerate spread on either side makes the standardization undefined.
    # Falling back to par alone is the honest answer; a blend nobody can scale
    # is not.
    if par_scale <= 0 or ffb_scale <= 0:
        return players.with_columns(
            pl.col("par").alias("blend_par"),
            pl.col("par").alias("blend_par_exact"),
        )

    out = _with_centers(players)
    w = float(weight)
    z_par = (pl.col("par") - pl.col("_par_mid")) / par_scale
    z_ffb = (pl.col("ffb_par") - pl.col("_ffb_mid")) / ffb_scale
    blended = pl.when(pl.col("ffb_par").is_null()).then(pl.col("par")).otherwise(
        pl.col("_par_mid") + (w * z_ffb + (1.0 - w) * z_par) * par_scale
    )

    # Rounded for display, exact for ranking, and the pair is not redundant.
    # Standardizing compresses the Footballers' scale by roughly a third, so two
    # players 0.1 apart on `ffb_par` land 0.06 apart here and collide at one
    # decimal. Ranking the rounded column then breaks that tie on **row order**,
    # which is arbitrary in a way that changes if the input is ever reordered —
    # at weight 1.0 it visibly reversed Pittman and Pollard against their own
    # ffb_par. Ranking the exact value keeps the order faithful and
    # reproducible; the tenth is a display convention, not the number.
    return out.with_columns(
        blended.alias("blend_par_exact"),
        blended.round(1).alias("blend_par"),
    ).drop("_par_mid", "_ffb_mid")


def attach_ecr(
    players: pl.DataFrame, profile: LeagueProfile | None = None
) -> pl.DataFrame:
    """Add FantasyPros expert consensus rank, from the page that prices this format.

    **The page choice is the whole trap.** FantasyPros publishes a superflex
    board (`redraft-op`) and a 1QB board (`redraft-overall`), and quarterbacks
    sit tens of ranks apart between them. Reading the wrong one is the ECR
    version of the 2026 superflex bug — a well-formed column, plausible numbers,
    and every quarterback mispriced. The page is picked off the profile's roster,
    not passed in, so it cannot drift from the format the way a loose argument
    would.

    `ecr_sd` travels with it. Most consensus products give a mean and stop; this
    one ships the dispersion, which is a free read on where a hundred rankers
    disagree — and disagreement is where a private opinion is worth having.

    Joined on the normalized name and deduped to one row per player first, the
    `attach_quality` pattern. Nulls where FantasyPros has no row, which is not
    the same as a low rank.
    """
    if not players.height:
        return players

    profile = profile or pf.resolve()
    superflex = "SUPER_FLEX" in set(profile.roster_positions)
    page = ECR_PAGE_SUPERFLEX if superflex else ECR_PAGE_STANDARD

    try:
        from src import nflverse as nv

        raw = nv.ff_rankings("draft")
    except Exception:
        raw = pl.DataFrame()

    empty = (
        pl.lit(None, dtype=pl.Float64).alias("ecr"),
        pl.lit(None, dtype=pl.Float64).alias("ecr_sd"),
    )
    if not raw.height or "page_type" not in raw.columns:
        return players.with_columns(*empty)

    board = raw.filter(
        (pl.col("page_type") == page)
        & pl.col("pos").str.replace_all(r"\d+", "").is_in(list(FANTASY_POSITIONS))
    )
    if not board.height:
        return players.with_columns(*empty)

    keys = (
        board.select(
            ids.normalize("player").alias("_norm"),
            pl.col("ecr").cast(pl.Float64),
            pl.col("sd").cast(pl.Float64).alias("ecr_sd"),
        )
        .filter(pl.col("ecr").is_not_null())
        .sort("ecr")
        .unique(subset=["_norm"], keep="first")
    )

    return (
        players.with_columns(ids.normalize("name").alias("_norm"))
        .join(keys, on="_norm", how="left")
        .drop("_norm")
    )


def blend_ecr(
    players: pl.DataFrame, weight: float = ECR_WEIGHT, base: str = "blend_par"
) -> pl.DataFrame:
    """Fold ECR into the blended value, on the rank scale.

    **Blended as a rank rather than converted to points, and that is the design
    decision.** Mapping ECR's rank through `expected.py` would push it down the
    same curve that pools the top six running backs into a single number — so it
    would contribute nothing precisely at the top of the board, which is where a
    third opinion was wanted. A rank carries one value per player all the way up.

    So ECR is negated (low rank is good), standardized on median and IQR the same
    way `blend_par` treats `ffb_par`, and mixed into the base at
    `config.ECR_WEIGHT` — far below the Footballers' weight, because ECR is a
    consensus of orderings largely reading the same market the board is already
    priced against, while the Footballers are an independent projection.

    Median and IQR rather than mean and standard deviation, for the reason
    `blend_par` gives: the two sources are censored differently in the tail and a
    scale set by players nobody drafts is the wrong one to calibrate on.

    Falls back to the base untouched wherever ECR is missing or its spread is
    degenerate. A player FantasyPros never ranked keeps his blended value rather
    than being pushed to the bottom.
    """
    if "ecr" not in players.columns or base not in players.columns:
        return players

    both = players.filter(
        pl.col(base).is_not_null() & pl.col("ecr").is_not_null()
    )
    if both.height < 4:
        return players

    def _center_scale(col: pl.Series) -> tuple[float, float]:
        return float(col.median()), float(col.quantile(0.75) - col.quantile(0.25))

    base_mid, base_scale = _center_scale(both.get_column(base))
    # Negated so that, like every other value on this board, larger is better.
    ecr_value = -both.get_column("ecr")
    ecr_mid, ecr_scale = _center_scale(ecr_value)
    if base_scale <= 0 or ecr_scale <= 0:
        return players

    w = float(weight)
    z_base = (pl.col(base) - base_mid) / base_scale
    z_ecr = ((-pl.col("ecr")) - ecr_mid) / ecr_scale
    blended = (
        pl.when(pl.col("ecr").is_null())
        .then(pl.col(base))
        .otherwise(base_mid + (w * z_ecr + (1.0 - w) * z_base) * base_scale)
    )

    return players.with_columns(blended.round(1).alias(base))


def compare_footballers(built: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """What the blend actually moved, player by player.

    The sibling of `compare_baselines`, and it exists for the same reason: a
    correction that never changes a decision is complexity that is not earning
    its keep. `rank_shift` is `board_rank - blend_rank`, so positive means the
    Footballers like him more than this project's own curve does.

    Sorted by the size of the disagreement, because the rows where two
    independent reads diverge are the only ones worth spending time on. Agreement
    is not information.
    """
    players = built.get("players", pl.DataFrame())
    if not players.height or "blend_rank" not in players.columns:
        return pl.DataFrame()

    cols = [
        c
        for c in (
            "name", "position", "team", "adp", "board_rank", "blend_rank",
            "par", "ffb_par", "blend_par", "ffb_spread", "n_analysts",
            "stalest_days",
        )
        if c in players.columns
    ]
    return (
        players.filter(pl.col("ffb_par").is_not_null())
        .select(cols)
        .with_columns(
            (pl.col("board_rank") - pl.col("blend_rank"))
            .cast(pl.Int32)
            .alias("rank_shift")
        )
        .sort(pl.col("rank_shift").abs(), descending=True, nulls_last=True)
    )


def build(
    league_id: str | None = None,
    season: int = SEASON,
    use_draft_demand: bool = True,
    gap: float = ex.TIER_GAP_POINTS,
    profile: LeagueProfile | None = None,
    footballers_weight: float = FOOTBALLERS_WEIGHT,
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
    # Attached to the *pool*, before keepers and drafted players leave, because
    # that is the frame the Footballers' own replacement level has to be read
    # off — the same pool and the same demand the board's baseline uses. See
    # `attach_footballers` for why each source keeps its own baseline.
    pool = attach_footballers(pool, summary, use_draft_demand=use_draft_demand)
    if "ffb_par" in pool.columns and not pool.get_column("ffb_par").is_not_null().any():
        warnings.append(
            "The Fantasy Footballers' projections did not load, so the board is "
            "this project's curve alone. Their page is public and unauthenticated; "
            "a failure here is a network problem or a site change, not a paywall."
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
    # A provisional ordering only. `rank_board` rewrites `board_rank` in the app
    # once quality and environment are attached, because an ordinal rank breaks
    # ties by row order and row order here is ADP — see that function. This stays
    # so `build` returns a sorted board on its own, for callers with no valuation
    # frame, and because it is the **unblended control** that makes
    # `compare_footballers` a real comparison rather than a restatement — the
    # same reason `compare_baselines` exists.
    players = players.with_columns(
        pl.col("par").rank("ordinal", descending=True).cast(pl.Int32).alias("board_rank")
    )
    players = blend_par(players, weight=footballers_weight)
    if "blend_par_exact" in players.columns:
        players = (
            players.with_columns(
                pl.col("blend_par_exact")
                .rank("ordinal", descending=True)
                .cast(pl.Int32)
                .alias("blend_rank")
            )
            .sort("blend_par_exact", descending=True, nulls_last=True)
            .drop("blend_par_exact")
        )
    else:
        # `nulls_last` matters here: polars defaults a descending sort to nulls
        # FIRST, so a player the ADP curve could not match would open the board
        # looking like its best pick.
        players = players.sort("par", descending=True, nulls_last=True)

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


# The columns `attach_vegas` carries across. `market` travels with `vegas_gap`
# because the gap's meaning depends on which prop it came from — a receiving
# yards line says something much closer to fantasy value than a passing yards
# line does. See `props.against_price` on why QB is the weakest case.
_VEGAS_COLUMNS = ["market", "line", "line_pct", "vegas_gap"]


def attach_vegas(players: pl.DataFrame, priced: pl.DataFrame) -> pl.DataFrame:
    """Add the sportsbook's read, carried across from an already-priced frame.

    **`priced` is required rather than fetched, and that is the point.** Every
    other `attach_*` here can build its own input; this one cannot, because the
    input comes from FanDuel over the network. Defaulting it to `None` and
    quietly returning a column of nulls would make "the book is down" and "the
    book has no line on him" the same value, and this module's tests would start
    depending on a bookmaker being reachable. The app composes it instead — the
    same division `attach_environment` and `props.against_price` already use.

    **The value is carried, never recomputed, and that constraint is load
    bearing.** `vegas_gap` is one percentile minus another, and `against_price`
    requires both be taken over the *same population* — the ~143 players
    `valuation.board` scores. Re-deriving it against this board's ~159 would
    change every number by shifting the denominator under one side only. It is
    also why `against_price` cannot simply be called here: the board frame
    carries `player_id`, not the `gsis_id` that function keys on.

    **Expect nulls for roughly two thirds of the board.** FanDuel posts
    season-long markets for about 92 players, top of the board only. A null is
    "no line posted", never "no edge".

    Joined on the normalized name, deduped to one row per *player* before the
    join — `ids.normalize` strips generational suffixes, so Michael Pittman Jr.
    and Sr. collapse to a single key and a join without the dedupe fans rows out
    without raising. That is not hypothetical; it took the props join from 145
    rows to 151.
    """
    if not players.height:
        return players

    have = [c for c in _VEGAS_COLUMNS if c in priced.columns]
    if not priced.height or "vegas_gap" not in have:
        return players.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("market"),
            pl.lit(None, dtype=pl.Float64).alias("line"),
            pl.lit(None, dtype=pl.Float64).alias("line_pct"),
            pl.lit(None, dtype=pl.Float64).alias("vegas_gap"),
        )

    keys = priced.select(
        ids.normalize("name").alias("_norm"),
        *[pl.col(c) for c in have],
    ).unique(subset=["_norm"], keep="first")

    return (
        players.with_columns(ids.normalize("name").alias("_norm"))
        .join(keys, on="_norm", how="left")
        .drop("_norm")
    )


def signal(
    players: pl.DataFrame,
    quality_edge: float = 15.0,
    vegas_edge: float = 10.0,
) -> pl.DataFrame:
    """Label where the two ADP-independent reads agree, oppose, or say nothing.

    **Why this is a label and not a score.** The board carries three numbers that
    look like opinions — `par`, `value_gap`, `vegas_gap` — but only two of them
    are independent of the market. `par` is a function of price: `exp_points`
    maps a player's *positional ADP rank* to what that rank historically scored,
    so within a position it reproduces ADP's ordering by construction. Averaging
    all three into a composite would count the market twice and call the result a
    second opinion.

    So the two that genuinely disagree for unrelated reasons are `value_gap`
    (per-opportunity quality measured here, weighted by how much each metric
    repeats) and `vegas_gap` (a line a bookmaker will take money on). This
    reports their relationship and leaves the ordering to `par`.

    Four levels, thresholds inherited from the agreement panel already on the
    Board tab:

    - **both up** — each past its threshold, both saying underpriced. Two
      unrelated methods agreeing is the strongest statement available here.
    - **both down** — the same, in reverse.
    - **split** — they oppose, each past threshold. **The interesting one**, and
      the reason no blend ships: an average turns this into a zero and the zero
      is indistinguishable from "nothing here".
    - **quiet** — both measured, neither past threshold.

    **A null stays null, and that is deliberate.** If either input is missing the
    label is null rather than `quiet`, because "the book posted no line" and "the
    book posted a line and it agrees with ADP" are different facts. Folding the
    first into `quiet` would overstate how much of this board has actually been
    checked — see `BIG_BOARD_SPEC.md` §8, which asked the question and is
    answered here.
    """
    if not players.height:
        return players
    if "value_gap" not in players.columns or "vegas_gap" not in players.columns:
        return players.with_columns(pl.lit(None, dtype=pl.Utf8).alias("signal"))

    quality, vegas = pl.col("value_gap"), pl.col("vegas_gap")
    measured = quality.is_not_null() & vegas.is_not_null()

    return players.with_columns(
        pl.when(~measured)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when((quality >= quality_edge) & (vegas >= vegas_edge))
        .then(pl.lit("both up"))
        .when((quality <= -quality_edge) & (vegas <= -vegas_edge))
        .then(pl.lit("both down"))
        .when((quality >= quality_edge) & (vegas <= -vegas_edge))
        .then(pl.lit("split"))
        .when((quality <= -quality_edge) & (vegas >= vegas_edge))
        .then(pl.lit("split"))
        .otherwise(pl.lit("quiet"))
        .alias("signal")
    )


def apply_env_weight(
    players: pl.DataFrame, weight: float = ENV_WEIGHT, base: str | None = None
) -> pl.DataFrame:
    """`par` adjusted for the offence a player actually plays in.

    **Until 2026-08-14 `env_swing` was computed, displayed, and given a weight of
    exactly zero.** That was never decided; it was where the column landed. And
    it is not a small omission — `env_swing` spans -31.5 to +47.0 on the 2026
    board against a PAR range of -65 to +72.6, so more than half the board's
    total spread sat in a column nothing read. De'Von Achane leads Puka Nacua by
    12.3 PAR and trails him by 67.5 points of offensive environment.

    `weight` comes from `config.ENV_WEIGHT` and is **asserted rather than
    measured** — the only such number in this project. It is below 1.0 because
    ADP already prices some of the environment (a good player on a bad offence is
    drafted later) and `env_swing` cannot tell how much, so adding it whole would
    double count. It is above 0.0 because the column is too large to keep
    ignoring by default. See the long note in `config.py`, and measure it before
    trusting the exact figure.

    **`base` defaults to `blend_par` when the Footballers blend has run**, and to
    `par` otherwise. That ordering is the point of the whole chain: the ADP curve
    gives a *slot* value blind to both player and team, the Footballers blend adds
    the player, and this adds the team. Applying the environment to raw `par`
    while a blended column sat unused would throw away the only player-level
    projection on the board.

    Adds `par_env`. Falls back to `par` where the environment join found nothing,
    so an unreachable Vegas feed costs a correction rather than a row.
    """
    if not players.height:
        return players
    base = base or ("blend_par" if "blend_par" in players.columns else "par")
    if base not in players.columns:
        return players
    if "env_swing" not in players.columns:
        return players.with_columns(pl.col(base).alias("par_env"))

    return players.with_columns(
        (pl.col(base) + weight * pl.col("env_swing").fill_null(0.0))
        .round(1)
        .alias("par_env")
    )


def roster_need(
    kept: pl.DataFrame | None = None,
    owner: str | None = None,
    profile: LeagueProfile | None = None,
) -> pl.DataFrame:
    """Which starting slots **you** still have to fill, after your own keepers.

    **The gap this closes is the difference between scarcity and need, and the
    board had only ever modelled the first.** `roster_demand` prices what the
    *league* is short of: it subtracts league-wide keepers from league-wide
    demand, which on the 2026 board leaves quarterback at 7 slots against 20 and
    therefore screaming scarce. That is true and it is not Zach's problem. Thirteen
    of those twenty are kept, and **two of them are his** — Jayden Daniels and
    Trevor Lawrence, against a roster carrying exactly two quarterback-capable
    slots, `QB` and `SUPER_FLEX`. His quarterback need is zero.

    The failure was not a bad weight. League scarcity is *maximally* misleading
    to the manager who caused it: the teams that make quarterback scarce by
    keeping one are precisely the teams that then must not draft one. So a board
    that reports only league demand tells every keeper-holder to buy the thing
    they already own, and it told him so at pick 4.

    Slots are filled **most restrictive first**, which is exactly optimal here
    for the reason `config.FLEX_SLOTS` gives: dedicated ⊂ FLEX ⊂ SUPER_FLEX
    nests, so a greedy pass cannot strand a player who had a better home. His two
    quarterbacks land in `QB` and then `SUPER_FLEX`, closing both. A roster
    carrying `REC_FLEX` *and* `WRRB_FLEX` would not nest and this would become a
    heuristic; that combination does not exist here, and `starter_demand` and the
    simulator's optimizer both say the same thing at the same volume.

    Returns one row per fantasy position: `starters` (dedicated slots at that
    position), `kept`, `open_dedicated`, `open_flex` (open flex slots this
    position is *eligible* for) and `slots_open`.

    **`slots_open` deliberately double counts a flex slot across every position
    that can fill it**, because the question it answers is "can this player still
    improve my starting lineup", not "how many of these should I draft". One open
    FLEX shows as an open slot for RB, WR and TE alike — exactly one of them will
    take it. Read `starters_left` off the frame's attrs for the honest total.
    """
    profile = profile or pf.resolve()
    owner = owner or SLEEPER_USERNAME
    kept = kept if kept is not None else kept_players(profile=profile)

    mine: list[str] = []
    if kept is not None and kept.height and "owner" in kept.columns:
        mine = (
            kept.filter(pl.col("owner") == owner)
            .get_column("position")
            .drop_nulls()
            .to_list()
        )

    slots = [s for s in profile.roster_positions if s not in NON_STARTING_SLOTS]
    dedicated: dict[str, int] = {}
    flex_open: dict[str, int] = {}
    for slot in slots:
        if slot in FLEX_SLOTS:
            flex_open[slot] = flex_open.get(slot, 0) + 1
        else:
            dedicated[slot] = dedicated.get(slot, 0) + 1

    pool: dict[str, int] = {}
    for position in mine:
        pool[position] = pool.get(position, 0) + 1

    # Dedicated slots first — the most restrictive of all, eligibility set of one.
    open_dedicated = dict(dedicated)
    for position, count in list(pool.items()):
        take = min(count, open_dedicated.get(position, 0))
        if take:
            open_dedicated[position] -= take
            pool[position] -= take

    # Then flex slots, narrowest eligibility first.
    for slot in sorted(flex_open, key=lambda s: len(FLEX_SLOTS[s])):
        for position in FLEX_SLOTS[slot]:
            while flex_open[slot] and pool.get(position, 0):
                flex_open[slot] -= 1
                pool[position] -= 1

    rows = []
    for position in FANTASY_POSITIONS:
        eligible_flex = sum(
            n for slot, n in flex_open.items() if position in FLEX_SLOTS[slot]
        )
        rows.append(
            {
                "position": position,
                "starters": dedicated.get(position, 0),
                "kept": sum(1 for p in mine if p == position),
                "open_dedicated": open_dedicated.get(position, 0),
                "open_flex": eligible_flex,
                "slots_open": open_dedicated.get(position, 0) + eligible_flex,
            }
        )

    out = pl.DataFrame(rows)
    # The un-double-counted total, which is the number that answers "how many
    # starters am I actually drafting".
    out = out.with_columns(
        pl.lit(
            sum(open_dedicated.values()) + sum(flex_open.values())
        ).cast(pl.Int32).alias("starters_left")
    )
    return out


def roster_demand(
    players: pl.DataFrame,
    replacement_frame: pl.DataFrame | None,
    value_col: str = "par_env",
) -> pl.DataFrame:
    """Mark where each position runs out of slots the league will actually fill.

    **The fix for the board's largest measured distortion, and it asserts no new
    number.** Ranking across positions on points-above-replacement promoted every
    tight end on the 2026 board — median **+47.5 places** against ADP, never fewer
    than +23 — and the cause is not any weight or blend. It is that PAR is a
    *level above a positional baseline* and the board holds each position to a
    different depth: 67 receivers against 18 tight ends. The deep receivers sit
    far below receiver replacement (median `par` -17.9) while the deep tight ends
    sit essentially at theirs (median 0.0), so a cross-position sort interleaves
    TE15 at -15 ahead of WR55 at -35.

    That comparison is arithmetically correct and decision-useless. **Below
    replacement, PAR carries no lineup value at all** — you would start the freely
    available replacement instead, so the marginal starting-lineup points from any
    of these players is zero regardless of how negative the number is. Ranking -15
    above -35 asserts a distinction PAR cannot support, which is the same house
    rule `expected.tiers` follows when it refuses to force a rank order the curve
    does not earn.

    So the board is split at the line where PAR stops meaning anything, and the
    line needs no new constant: `replacement()` already computes
    `replacement_rank` per position from the league's own roster shape and its
    keepers. A player inside it is inside the demand the draft has to fill; the
    first player past it *is* the replacement, by construction. On the 2026 board
    that is 56 players — 22 RB, 21 WR, 7 QB, 6 TE.

    **`pos_rank` is taken on `value_col`, not on ADP**, so the board still gets to
    disagree with the market about *which* players fill those slots. It only stops
    disagreeing about *how many* exist.

    A position missing from `replacement_frame` keeps every player inside the
    line, and so does a player the ranking column could not score. Both are the
    `rank_board` lesson: not measured is not worst, and a data gap must not quietly
    demote someone sixty places.

    Adds `demand`, `pos_rank` and `in_demand`. `rank_board` reads them.
    """
    if not players.height or "position" not in players.columns:
        return players
    if value_col not in players.columns:
        return players
    if replacement_frame is None or not replacement_frame.height:
        return players
    if "replacement_rank" not in replacement_frame.columns:
        return players

    # `replacement_rank` is the first player *past* demand, so demand is one less.
    demand = replacement_frame.select(
        "position",
        (pl.col("replacement_rank") - 1).cast(pl.Int32).alias("demand"),
    )
    return (
        players.join(demand, on="position", how="left")
        .with_columns(
            pl.col(value_col)
            .rank("ordinal", descending=True)
            .over("position")
            .cast(pl.Int32)
            .alias("pos_rank")
        )
        .with_columns(
            (pl.col("pos_rank") <= pl.col("demand")).fill_null(True).alias("in_demand")
        )
    )


def rank_board(players: pl.DataFrame, value_col: str = "par") -> pl.DataFrame:
    """Order the board by what the curve can actually resolve, then by quality.

    **This replaces an ordering that was, inside a tie, just ADP.** `board_rank`
    was `rank("ordinal")` over `par`, and an ordinal rank breaks ties by *row
    order* — which traces straight back to the ADP the pool arrived in. Nine
    running backs share a PAR of 72.6 on the 2026 board, so the top nine were
    presented as a strict ordering with no evidence behind a single step of it.
    Derrick Henry outranked James Cook because the market drafts Henry eleven
    picks earlier. For a tool whose entire purpose is to disagree with the
    market, a tiebreak that *is* the market is the worst available answer.

    Worse, the board already carried the number that contradicted it. Inside that
    "indistinguishable" nine, `quality_pct` ran from 13.6 to 100 — and Josh
    Jacobs (56.8) outranked Ja'Marr Chase (91.1) outright.

    So the order is now lexicographic on two keys:

      1. **The group's PAR**, not the player's. If the pooled standard error says
         the curve cannot tell nine backs apart, they share a rank key and the
         board stops pretending otherwise. `block` is that key, dense-ranked.
      2. **`quality_pct` within the block** — the read that is *not* derived from
         ADP, so a tie is broken by an independent opinion rather than a circular
         one.

    **What this is not: a blend.** PAR alone decides which block a player is in,
    and quality is consulted only where PAR is provably indifferent. Averaging
    the two would double-count the market, since `par` is a function of price.
    See `BIG_BOARD_SPEC.md` §2.

    **The honest caveat, and it must travel with this function.** `quality_pct`
    is last season's per-opportunity efficiency, and this repo has twice measured
    that such signals do not beat ADP out of sample — `breakout.py` at +0.035 AUC
    and `projection.py` at +0.002 Spearman, both intervals covering zero. The
    narrower claim made here is *not* "quality beats ADP"; it is "where the ADP
    curve has no resolution, an independent read is a better tiebreak than the
    order rows happened to arrive in." That is a judgement, not a result, and it
    is why `block` is displayed rather than a dense 1..N rank: the block is
    measured, the order inside it is not.

    `value_col` is what the blocks are cut on. The app passes `par_env` so the
    team-environment weight moves a player between blocks rather than only inside
    one — Achane and Nacua sit in different blocks, so an adjustment confined to
    within-block ordering could never have reached the comparison being asked
    about.

    **The board is cut in two at roster demand, when `roster_demand` has run.**
    Everything above is ordered as described; everything below is ordered by ADP
    and carries a null `block`, because below replacement PAR has no lineup value
    to rank on and this tool has nothing to say. Deferring there is not a
    concession — it is the only honest answer, and it is what stops eighteen tight
    ends from being interleaved into the top hundred on the strength of a number
    that means "less bad than a receiver nobody will start either". See
    `roster_demand`.

    A null `block` below the line is deliberate and should stay visible. The
    column is the board's claim to have resolved something; printing a number
    there would extend the claim past where it holds.

    **The order inside a block is three keys, not two**, and the middle one
    carries an imputation:

      1. the block's PAR, as above;
      2. `quality_pct`, with **the block's median standing in where it is
         missing** — see the note at the fill;
      3. `value_col` itself, so the final tiebreak is stated rather than
         inherited from whatever order the frame arrived in.

    Adds `block` and rewrites `board_rank`. A player with no quality score is
    placed among his block's typical members rather than beneath all of them:
    "not measured" is not "worst", and until 2026-08-14 this function said that
    and did the opposite.
    """
    if not players.height or value_col not in players.columns:
        return players

    # Without a group the block degenerates to the player's own value, which is
    # the pre-2026-08-13 behaviour minus the ADP tiebreak. Better than raising.
    group_key = (
        pl.col(value_col).max().over(["position", "indist_group"])
        if "indist_group" in players.columns
        else pl.col(value_col)
    )
    quality = (
        pl.col("quality_pct")
        if "quality_pct" in players.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    # **Unscored players take the block's median quality, not the bottom of it.**
    # `nulls_last` on this key used to demote them, which is precisely what the
    # paragraph above promises not to do: on the 2026 board all seven unscored
    # players inside roster demand sat last in their block, and Brock Purdy —
    # the *highest* `par_env` in his — sat third of three. A missing quality
    # score is a coverage gap in last season's volume floor, not a finding about
    # the player. The median says "assume he is typical here", which is the only
    # thing the absence of evidence supports.
    group_cols = [c for c in ("position", "indist_group") if c in players.columns]
    quality_filled = (
        quality.fill_null(quality.median().over(group_cols))
        if group_cols
        else quality.fill_null(quality.median())
    )
    keyed = players.with_columns(
        group_key.alias("_group_par"),
        quality_filled.alias("_q"),
        # Last key, and it only ever fires on an exact tie in all three above.
        # Imputing the median can make an unscored player *equal* to a measured
        # one, and where the board genuinely cannot separate them the measured
        # player goes first. That keeps the old guard's point — nothing unknown
        # opens a block — without its overreach of sending him to the bottom.
        quality.is_not_null().alias("_measured"),
    )

    # nulls_last on every key. Polars defaults descending sorts to nulls FIRST, so
    # an unscored player would otherwise open the board — the exact trap CLAUDE.md
    # documents, and the reason a "best N" query here returns the N rows that
    # could not be scored.
    # `value_col` is the third key so the last tiebreak is **stated rather than
    # inherited**. Polars sorts stably, so equal keys previously kept whatever
    # order the frame arrived in — which happens to be `blend_par` today and is
    # an invariant nothing enforces. `blend_par` already found the sharp edge of
    # relying on that: rounding collapsed two players into one displayed value
    # and row order then reversed them against their own projections. A board
    # whose ordering can change because an upstream join reordered rows is not
    # reproducible, and the fix is cheap.
    def _ranked(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sort(
            ["_group_par", "_q", value_col, "_measured"],
            descending=[True, True, True, True],
            nulls_last=True,
        ).with_columns(
            pl.col("_group_par").rank("dense", descending=True).cast(pl.Int32).alias("block")
        )

    if "in_demand" not in keyed.columns:
        ordered = _ranked(keyed)
    else:
        inside = _ranked(keyed.filter(pl.col("in_demand")))
        # Ascending, and ADP's own nulls last: a player the market never priced is
        # not the best pick left on the board.
        beyond = keyed.filter(~pl.col("in_demand")).with_columns(
            pl.lit(None, dtype=pl.Int32).alias("block")
        )
        if "adp" in beyond.columns:
            beyond = beyond.sort("adp", nulls_last=True)
        ordered = pl.concat([inside, beyond], how="vertical")

    return ordered.with_columns(
        pl.int_range(1, pl.len() + 1).cast(pl.Int32).alias("board_rank")
    ).drop("_group_par", "_q", "_measured")


def positional_drop(players: pl.DataFrame, spots: int = 5) -> pl.DataFrame:
    """PAR given up by falling `spots` places at the same position.

    **The column that stops PAR being read as the whole answer.** PAR is a
    *level* — how much this player is worth over replacement. The decision on the
    clock is about *shape*: what you lose by taking someone else first and
    circling back. Those come apart badly on the 2026 board, and not subtly:

        RB   72.6  72.6  72.6  72.6  72.6  72.6  70.5  67.4  60.5  53.9
        WR   58.2  52.2  46.4  45.2  39.7  31.8  21.4  21.4  20.4  16.1

    The top six running backs are the *same number*. Not close — identical,
    because the ADP curve is not monotone there and `expected.tiers` pools ranks
    it cannot order rather than inventing one. So RB1 to RB8 costs 5.2 points
    while WR1 to WR8 costs 36.8, and a board sorted on PAR alone puts six
    interchangeable backs above a receiver you genuinely cannot replace.

    Read it against PAR, never instead of it: **between two positions you intend
    to fill anyway, take the one that is more expensive to wait on.** A high drop
    on a low-PAR player is still a low-PAR player.

    `spots` defaults to 5 as a rough stand-in for how many of one position leave
    the board between picks in a 10-team league. It is a blunt instrument by
    design — `cost_of_waiting` is the exact version, using your real pick list
    and the draft-slot dispersion FFC ships beside ADP, and it should be
    preferred whenever a pick list exists. This exists because it needs neither,
    so it works on a cold clone and for a league this project cannot read.

    Null where fewer than `spots` players remain below him at the position:
    there is nothing left to fall to, which is not the same as a drop of zero.
    """
    if not players.height or "par" not in players.columns:
        return players
    if spots < 1:
        raise ValueError(f"spots must be at least 1, got {spots}")

    # The shift has to run down a PAR-sorted frame, but the caller's row order is
    # not this function's to change — `build` returns a board the app renders in
    # the order it was handed. So the sort is undone before returning.
    return (
        players.with_row_index("_row")
        # nulls_last on a descending sort, every time: polars defaults it to
        # False, so unscored players would head each position group and the drop
        # would be measured from a player who has no PAR at all.
        .sort("par", descending=True, nulls_last=True)
        .with_columns(
            (pl.col("par") - pl.col("par").shift(-spots))
            .round(1)
            .over("position")
            .alias("drop")
        )
        .sort("_row")
        .drop("_row")
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
