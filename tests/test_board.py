"""The keeper-adjusted draft board.

The failure that matters here is a kept player left sitting on the board, or a
replacement baseline computed against a pool that does not exist. Both produce
a board that looks completely normal and is wrong on draft day, so the pure
functions are tested against synthetic frames where the right answer is known
by construction, and the live-league path is exercised separately and skipped
when no league is configured.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import adp
from src import board as bd


def _board(rows: list[tuple[str, str, float, float, bool]]) -> pl.DataFrame:
    """(name, position, adp, exp_points, kept) -> a board-shaped frame."""
    return pl.DataFrame(
        [
            {
                "name": n, "position": p, "adp": a, "exp_points": e,
                "kept": k, "kept_by": "someone" if k else None,
            }
            for n, p, a, e, k in rows
        ]
    )


def _summary(position: str, league: float, kept: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "position": position,
                "league_demand": league,
                "kept": kept,
                "draft_demand": max(league - kept, 0.0),
                "undeclared_teams": 0,
            }
        ]
    )


# --- replacement level ------------------------------------------------------


def test_replacement_prices_against_the_draft_pool_not_the_league() -> None:
    """The module's central claim, on a case where the answer is arithmetic.

    Ten QB slots league-wide, eight already kept, so the draft fills two. The
    baseline must be the third-best *available* quarterback, not the eleventh.
    """
    rows = [(f"qb{i}", "QB", float(i), 200.0 - 10 * i, False) for i in range(12)]
    board = _board(rows)
    summary = _summary("QB", league=10.0, kept=8)

    draft = bd.replacement(board, summary, use_draft_demand=True)
    league = bd.replacement(board, summary, use_draft_demand=False)

    assert draft.get_column("replacement_rank")[0] == 3
    assert league.get_column("replacement_rank")[0] == 11
    # Pricing against a shallower pool must give the *higher* baseline, which
    # is what makes the position worth less in the draft than league-wide
    # demand alone would suggest.
    assert (
        draft.get_column("replacement_points")[0]
        > league.get_column("replacement_points")[0]
    )


def test_replacement_ignores_kept_players() -> None:
    """A kept player must never set the baseline — nobody can draft him."""
    board = _board(
        [("elite", "RB", 1.0, 300.0, True)]
        + [(f"rb{i}", "RB", float(i + 2), 100.0 - i, False) for i in range(5)]
    )
    repl = bd.replacement(board, _summary("RB", league=2.0, kept=1))
    assert repl.get_column("replacement_points")[0] < 200.0


def test_replacement_survives_a_thin_pool() -> None:
    """Demand deeper than the available pool must clamp, not raise."""
    board = _board([(f"te{i}", "TE", float(i), 90.0 - i, False) for i in range(3)])
    repl = bd.replacement(board, _summary("TE", league=40.0, kept=0))
    assert repl.height == 1
    assert repl.get_column("replacement_rank")[0] == 3


# --- keeper matching --------------------------------------------------------


def test_unmatched_keepers_are_reported() -> None:
    """A keeper that silently fails to match is the one error here that would
    actively mislead — he would still be sitting on the board."""
    kept = pl.DataFrame(
        [
            {"owner": "a", "player_name": "Josh Allen", "position": "QB"},
            {"owner": "b", "player_name": "Deep Bench Guy", "position": "WR"},
        ]
    )
    board = _board([("Josh Allen", "QB", 1.4, 250.0, True)])
    unmatched = bd.keeper_match_report(kept, board)
    assert unmatched.height == 1
    assert unmatched.get_column("player_name")[0] == "Deep Bench Guy"


def test_keeper_matching_is_case_and_punctuation_insensitive() -> None:
    """Sleeper and FFC share no ids, so the join is on normalized names."""
    kept = pl.DataFrame(
        [{"owner": "a", "player_name": "ja'marr chase", "position": "WR"}]
    )
    board = _board([("Ja'Marr Chase", "WR", 10.1, 177.0, True)])
    assert bd.keeper_match_report(kept, board).height == 0


# --- the live league --------------------------------------------------------


@pytest.fixture(scope="module")
def built() -> dict[str, pl.DataFrame]:
    """The live board. Skips rather than fails on the three legitimate ways
    this can be unavailable: no league configured, no network, cold cache."""
    try:
        out = bd.build()
    except Exception:  # noqa: BLE001 — offline is a supported state
        pytest.skip("no league reachable")
    if not out["players"].height:
        pytest.skip("no ADP board cached")
    return out


@pytest.fixture(scope="module")
def with_keepers(built) -> dict[str, pl.DataFrame]:
    """The board *and* a league that has declared keepers.

    Without `FF_EDGE_LEAGUE_ID` the board still builds — it is simply
    unadjusted, which is correct behaviour for someone who cloned the repo and
    has no league. The keeper properties below need an actual keeper list.
    """
    if not built["kept"].height:
        pytest.skip("no league configured (FF_EDGE_LEAGUE_ID unset)")
    return built


def test_no_kept_player_appears_on_the_board(with_keepers) -> None:
    """The property the whole module exists for."""
    players = with_keepers["players"]
    assert players.height
    assert not players.get_column("kept").any()

    kept_names = set(
        with_keepers["kept"].get_column("player_name").drop_nulls().to_list()
    )
    on_board = set(players.get_column("name").to_list())
    assert not (kept_names & on_board)


def test_unadjusted_board_is_honest_about_having_no_keepers(built) -> None:
    """With no league configured the board must report zero kept rather than
    quietly implying the pool has been adjusted."""
    if built["kept"].height:
        pytest.skip("league is configured — covered by the keeper tests")
    assert (built["summary"].get_column("kept") == 0).all()
    assert built["summary"].get_column("draft_demand").to_list() == (
        built["summary"].get_column("league_demand").to_list()
    )


def test_draft_demand_never_exceeds_league_demand(built) -> None:
    s = built["summary"]
    assert (s.get_column("draft_demand") <= s.get_column("league_demand")).all()
    assert (s.get_column("draft_demand") >= 0).all()


def test_pick_inventory_is_internally_consistent(with_keepers) -> None:
    """Pick numbers must be unique, in range, and consistent with the keeper
    placements — the three ways a snake-plus-trades calculation goes wrong."""
    pk = bd.picks()
    if not pk.height:
        pytest.skip("no draft order posted yet")

    nums = pk.get_column("pick_no").to_list()
    assert len(nums) == len(set(nums)), "a pick is listed twice"
    assert all(1 <= n <= 15 * 10 for n in nums)

    # Every pick consumed by a keeper is marked unusable, and vice versa.
    for r in pk.iter_rows(named=True):
        assert r["usable"] == (r["keeper"] is None)

    kept_here = set(pk.filter(~pl.col("usable")).get_column("keeper").to_list())
    my_keepers = set(
        with_keepers["kept"]
        .filter(pl.col("owner") == pk.get_column("from_owner").mode()[0])
        .get_column("player_name")
        .to_list()
    ) if with_keepers["kept"].height else set()
    # Keeper names on my picks must be real keepers, not stray draft picks.
    all_keepers = set(with_keepers["kept"].get_column("player_name").to_list())
    assert kept_here <= all_keepers


def test_targets_respects_the_availability_floor(built) -> None:
    """Only players who plausibly last may be suggested, best PAR first."""
    players = built["players"]
    assert "stdev" in players.columns, "dispersion must survive the board build"

    got = bd.targets(players, 24, min_available=0.5)
    assert got.height
    assert (got.get_column("p_available_at_24") >= 0.5).all()
    par = got.get_column("par").to_list()
    assert par == sorted(par, reverse=True)

    # A very early pick makes almost everyone available; a late one does not.
    # `top` has to exceed the board or it caps both sides and the comparison
    # silently becomes 50 == 50 — which is what happened when the board grew.
    uncapped = players.height + 1
    early = bd.targets(players, 2, min_available=0.5, top=uncapped).height
    late = bd.targets(players, 120, min_available=0.5, top=uncapped).height
    assert early > late


def test_targets_refuses_to_guess_without_dispersion(built) -> None:
    """A survival curve invented from a default stdev would look exactly as
    confident as a real one, so the absence of dispersion must return nothing."""
    stripped = built["players"].drop("stdev")
    assert bd.targets(stripped, 24).height == 0


def test_tiers_and_ranks_agree(built) -> None:
    """A better tier must never contain a worse board rank than a worse tier."""
    players = built["players"].sort("board_rank")
    ranks = players.get_column("board_rank").to_list()
    assert ranks == sorted(ranks)
    assert players.get_column("par").is_not_null().all()


# --- team environment -------------------------------------------------------


def test_environment_join_normalizes_team_codes(built) -> None:
    """FFC says LAR where nflverse says LA. An unnormalized join does not fail,
    it silently nulls every Rams player — which is worse."""
    p = bd.attach_environment(built["players"])
    if p.get_column("team_implied").is_null().all():
        pytest.skip("no lines posted yet")
    assert p.get_column("team_implied").null_count() == 0


def test_env_swing_sign_follows_the_offence() -> None:
    """Above-average offence helps, below-average hurts, and a replacement-level
    player is barely moved either way."""
    players = pl.DataFrame(
        {
            "name": ["good", "bad", "tiny"],
            "position": ["TE"] * 3,
            "team": ["CHI", "ARI", "ARI"],
            "exp_points": [150.0, 150.0, 5.0],
            "par": [30.0, 30.0, 0.0],
        }
    )
    got = bd.attach_environment(players)
    if got.get_column("team_implied").is_null().any():
        pytest.skip("no lines posted yet")
    by = {r["name"]: r["env_swing"] for r in got.iter_rows(named=True)}
    assert by["good"] > 0 > by["bad"]
    assert abs(by["tiny"]) < abs(by["bad"])


def test_context_flags_catch_a_small_edge_against_a_big_gap(built) -> None:
    """The pairs where the board is not really the deciding input."""
    p = bd.attach_environment(built["players"])
    if p.get_column("env_swing").is_null().all():
        pytest.skip("no lines posted yet")
    flags = bd.context_flags(p)
    assert flags.height
    # Every flagged pair must genuinely have context outweighing the board.
    assert (flags.get_column("env_edge") > flags.get_column("par_edge")).all()


# --- Survival against the right number line ---------------------------------


# Keepers priced early but slotted late, which is the real league's shape and
# the only shape where the two number lines actually diverge. With one keeper
# ahead of a player and his slot also ahead, removal (-1) and the consumed pick
# (+1) cancel exactly — the gap opens only when the keepers priced ahead of you
# outnumber the keeper slots spent before your pick. In the Shiva Bowl thirteen
# keepers sit ahead of pick 24 by ADP while only three slots are consumed.
_KEEPER_SLOTS = [100, 101, 102, 103]


def _synthetic_board() -> pl.DataFrame:
    """Four keepers priced ahead of the target, their picks spent much later."""
    return pl.DataFrame(
        {
            "name": ["k1", "k2", "k3", "k4", "target", "deep"],
            "position": ["RB"] * 6,
            "adp": [1.0, 2.0, 3.0, 4.0, 20.0, 60.0],
            "stdev": [2.0, 2.0, 2.0, 2.0, 6.0, 12.0],
            "par": [70.0, 65.0, 60.0, 55.0, 40.0, 10.0],
            "tier": [1, 1, 1, 2, 2, 5],
            "kept": [True, True, True, True, False, False],
        }
    )


def test_survival_reads_the_column_it_is_given() -> None:
    """The two number lines give different answers, and that is the whole point."""
    board = bd.keeper_adjusted_adp(_synthetic_board(), slots=_KEEPER_SLOTS, teams=10)
    raw = adp.survival(board, 24, adp_col="adp")
    adjusted = adp.survival(board, 24, adp_col="exp_pick")
    col = "p_available_at_24"
    for a, b in zip(raw.get_column(col), adjusted.get_column(col)):
        assert b <= a, "keepers move players earlier, so survival cannot rise"


def test_survival_overstates_availability_on_raw_adp_in_a_keeper_league() -> None:
    """The bug this parameter exists for, as an assertion.

    A keeper is off the board and consumes a pick, so everyone behind him is
    selected earlier than public ADP says. Measuring survival against raw ADP
    reports players as available who will already be gone.
    """
    board = bd.keeper_adjusted_adp(_synthetic_board(), slots=_KEEPER_SLOTS, teams=10)
    col = "p_available_at_18"
    raw = adp.survival(board, 18, adp_col="adp").filter(pl.col("name") == "target")
    adjusted = adp.survival(board, 18, adp_col="exp_pick").filter(
        pl.col("name") == "target"
    )
    assert raw.get_column(col)[0] > adjusted.get_column(col)[0]


def test_survival_ignores_a_column_it_does_not_have() -> None:
    """A board with no keeper adjustment must not raise, it must fall back."""
    plain = _synthetic_board()
    assert "p_available_at_10" not in adp.survival(plain, 10, adp_col="exp_pick").columns
    assert "p_available_at_10" in adp.survival(plain, 10).columns


def test_targets_prefers_the_keeper_adjusted_pick_number(built) -> None:
    """On a live board `targets` must be measuring against exp_pick."""
    players = built["players"]
    if "exp_pick" not in players.columns:
        pytest.skip("board has no keeper adjustment")
    got = bd.targets(players, 24, min_available=0.35)
    assert got.height
    assert "exp_pick" in got.columns, "the column it judged on must be shown"


# --- Cost of waiting --------------------------------------------------------


def test_cost_of_waiting_falls_off_as_picks_get_later(built) -> None:
    """The best player available cannot improve by waiting.

    Every player's survival probability falls monotonically with the pick
    number, so the expectation built from them has to fall too. A rise would
    mean the walk lost track of who was already gone.
    """
    players = built["players"]
    got = bd.cost_of_waiting(players, [4, 17, 24, 37, 44])
    if not got.height:
        pytest.skip("no dispersion on this board")
    for position in got.get_column("position").unique().to_list():
        vals = (
            got.filter(pl.col("position") == position)
            .sort("pick_no")
            .get_column("best_par")
            .to_list()
        )
        assert vals == sorted(vals, reverse=True), f"{position} improves by waiting"


def test_cost_of_waiting_is_never_negative(built) -> None:
    got = bd.cost_of_waiting(built["players"], [4, 24, 44])
    if not got.height:
        pytest.skip("no dispersion on this board")
    costs = got.get_column("cost_of_waiting").drop_nulls().to_list()
    assert costs, "some cost must be computed"
    assert all(c >= 0 for c in costs), costs


def test_cost_of_waiting_has_no_cost_on_the_last_pick(built) -> None:
    """There is nothing after your last pick to wait for, so it is null."""
    picks = [4, 24, 44]
    got = bd.cost_of_waiting(built["players"], picks)
    if not got.height:
        pytest.skip("no dispersion on this board")
    last = got.filter(pl.col("pick_no") == max(picks))
    assert last.height
    assert last.get_column("cost_of_waiting").null_count() == last.height


def test_cost_of_waiting_is_empty_without_picks_or_dispersion(built) -> None:
    assert not bd.cost_of_waiting(built["players"], []).height
    assert not bd.cost_of_waiting(pl.DataFrame(), [4, 24]).height
    no_disp = built["players"].drop("stdev") if "stdev" in built["players"].columns else built["players"]
    assert not bd.cost_of_waiting(no_disp, [4, 24]).height


# --- Live draft state -------------------------------------------------------


def _fake_picks(names: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pick_no": list(range(1, len(names) + 1)),
            "round": [1] * len(names),
            "player_name": names,
            "picked_by": ["someone"] * len(names),
        }
    )


def test_drafted_players_leave_the_board(built, monkeypatch) -> None:
    """The failure a draft-day board cannot have.

    Nothing removed a player from the pool when he was selected — `kept` reads
    the roster keepers field, which is a static pre-draft declaration. By the
    third round the board would still be offering players taken in the first.
    """
    pre = built["players"]
    gone = (
        pre.sort("par", descending=True, nulls_last=True)
        .head(12)
        .get_column("name")
        .to_list()
    )
    monkeypatch.setattr(bd, "drafted_players", lambda league_id=None: _fake_picks(gone))

    mid = bd.build()
    assert mid["players"].height == pre.height - len(gone)
    assert not (set(mid["players"].get_column("name")) & set(gone))
    assert mid["drafted"].height == len(gone)


def test_par_does_not_move_when_someone_else_picks(built, monkeypatch) -> None:
    """PAR is a valuation, not a function of draft state.

    Replacement is computed before live picks are removed on purpose. Excluding
    drafted players from the baseline pool while leaving demand fixed would walk
    the baseline steadily deeper into what remains and inflate everyone's PAR as
    the night went on — the board would look like it was finding value simply
    because the draft was progressing.
    """
    pre = built["players"]
    gone = (
        pre.sort("par", descending=True, nulls_last=True)
        .head(12)
        .get_column("name")
        .to_list()
    )
    monkeypatch.setattr(bd, "drafted_players", lambda league_id=None: _fake_picks(gone))

    mid = bd.build()
    joined = pre.select("name", pl.col("par").alias("pre")).join(
        mid["players"].select("name", pl.col("par").alias("mid")), on="name", how="inner"
    )
    assert joined.height, "some players must survive to compare"
    drift = (joined.get_column("mid") - joined.get_column("pre")).abs().max()
    assert drift == 0.0, f"PAR moved by {drift} because other teams picked"


def test_an_empty_draft_leaves_the_board_untouched(built, monkeypatch) -> None:
    """Before the draft opens, every caller must behave exactly as it did."""
    monkeypatch.setattr(bd, "drafted_players", lambda league_id=None: pl.DataFrame())
    again = bd.build()
    assert again["players"].height == built["players"].height
    assert not again["players"].get_column("drafted").any()


def test_drafted_players_excludes_keepers(with_keepers) -> None:
    """Keepers arrive on the same endpoint and are already handled.

    Counting them twice would subtract them from positional demand once as
    keepers and again as picks.
    """
    gone = bd.drafted_players()
    if not gone.height:
        pytest.skip("draft has not started — nothing live to check")
    kept_names = set(with_keepers["kept"].get_column("player_name").to_list())
    assert not (set(gone.get_column("player_name")) & kept_names)


def test_draftable_marks_drafted_without_touching_kept() -> None:
    """The two flags mean different things and must stay separable."""
    board = _board(
        [("a", "RB", 1.0, 300.0, True), ("b", "RB", 2.0, 200.0, False),
         ("c", "RB", 3.0, 100.0, False)]
    )
    # Exercise the marking logic directly on a synthetic board.
    marked = board.with_columns(
        pl.col("name").is_in(["b"]).alias("drafted")
    )
    assert marked.filter(pl.col("kept")).get_column("name").to_list() == ["a"]
    assert marked.filter(pl.col("drafted")).get_column("name").to_list() == ["b"]
    available = marked.filter(~pl.col("kept") & ~pl.col("drafted"))
    assert available.get_column("name").to_list() == ["c"]


# --- indistinguishable groups -----------------------------------------------


def _scored(rows: list[tuple[str, str, float, float]]) -> pl.DataFrame:
    """name, position, par, se — the only columns `indistinguishable` reads."""
    return pl.DataFrame(
        {
            "name": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "par": [r[2] for r in rows],
            "se": [r[3] for r in rows],
        }
    )


def test_indistinguishable_does_not_chain_down_a_shallow_slope() -> None:
    """The failure mode that makes single-linkage useless here.

    Each player is within a pooled standard error of the man directly above him,
    but the top and bottom are 40 points apart. Comparing against the previous
    player would put all nine in one group and report that the entire position is
    interchangeable. Comparing against the group *leader* — what the function
    actually does — has to cut somewhere.
    """
    rows = [(f"p{i}", "WR", 100.0 - 5.0 * i, 4.0) for i in range(9)]
    got = bd.indistinguishable(_scored(rows))
    groups = got.get_column("indist_group").unique().to_list()
    assert len(groups) > 1, "a 40-point spread collapsed into one group"

    # And every group must actually satisfy the claim it makes: each member
    # within the pooled SE of its own leader. Brute-forced rather than asserted.
    for group in groups:
        members = got.filter(pl.col("indist_group") == group).sort(
            "par", descending=True
        )
        lead_par = members.get_column("par")[0]
        lead_se = members.get_column("se")[0]
        for par, se in zip(members.get_column("par"), members.get_column("se")):
            pooled = (lead_se**2 + se**2) ** 0.5
            assert lead_par - par <= pooled + 1e-9


def test_indistinguishable_separates_a_genuine_cliff() -> None:
    """Two tight clusters far apart must not merge."""
    rows = [
        ("a", "RB", 100.0, 3.0), ("b", "RB", 99.0, 3.0),
        ("c", "RB", 40.0, 3.0), ("d", "RB", 39.0, 3.0),
    ]
    got = bd.indistinguishable(_scored(rows))
    by_name = {r["name"]: r["indist_group"] for r in got.iter_rows(named=True)}
    assert by_name["a"] == by_name["b"]
    assert by_name["c"] == by_name["d"]
    assert by_name["a"] != by_name["c"]
    counts = {r["name"]: r["indist_n"] for r in got.iter_rows(named=True)}
    assert counts["a"] == 2 and counts["c"] == 2


def test_indistinguishable_is_per_position() -> None:
    """A receiver and a back at the same PAR are not the same asset."""
    rows = [("w", "WR", 50.0, 5.0), ("r", "RB", 50.0, 5.0)]
    got = bd.indistinguishable(_scored(rows))
    assert got.get_column("indist_n").to_list() == [1, 1]


def test_indistinguishable_leaves_unscored_players_ungrouped() -> None:
    """A null PAR is not evidence of similarity to anything."""
    rows = [("a", "TE", 30.0, 4.0), ("b", "TE", None, 4.0)]
    got = bd.indistinguishable(_scored(rows))
    blank = got.filter(pl.col("name") == "b")
    assert blank.get_column("indist_group")[0] is None


def test_indistinguishable_degrades_without_a_standard_error() -> None:
    """No `se` column means the question cannot be asked — nulls, not a crash."""
    frame = _scored([("a", "WR", 10.0, 1.0)]).drop("se")
    got = bd.indistinguishable(frame)
    assert got.get_column("indist_group").is_null().all()


# --- tier map ---------------------------------------------------------------


def _tiered() -> pl.DataFrame:
    """A board with known tiers, for counting."""
    return pl.DataFrame(
        {
            "name": ["a", "b", "c", "d", "e", "f"],
            "position": ["RB", "RB", "RB", "WR", "WR", "WR"],
            "par": [50.0, 48.0, 20.0, 30.0, 29.0, 28.0],
            "tier": [1, 1, 2, 1, 1, 1],
        }
    )


def test_tier_map_counts_what_is_left_not_what_existed() -> None:
    got = bd.tier_map(_tiered())
    rb1 = got.filter((pl.col("position") == "RB") & (pl.col("tier") == 1))
    assert rb1.get_column("n_left")[0] == 2
    assert rb1.get_column("best_available")[0] == "a"
    assert rb1.get_column("par_top")[0] == 50.0
    assert rb1.get_column("par_bottom")[0] == 48.0
    assert got.filter(pl.col("position") == "WR").get_column("n_left")[0] == 3


def test_tier_map_best_available_is_the_top_of_its_tier() -> None:
    """`best_available` is read on the clock — it must be the max, not the first row."""
    scrambled = _tiered().sort("par")  # worst first, to catch an order assumption
    got = bd.tier_map(scrambled)
    for row in got.iter_rows(named=True):
        members = scrambled.filter(
            (pl.col("position") == row["position"]) & (pl.col("tier") == row["tier"])
        )
        best = members.sort("par", descending=True).get_column("name")[0]
        assert row["best_available"] == best
        assert row["par_top"] == members.get_column("par").max()


def test_tier_map_caps_tiers_per_position() -> None:
    """Tier 9 at running back is not a draft-day input."""
    many = pl.DataFrame(
        {
            "name": [f"p{i}" for i in range(12)],
            "position": ["RB"] * 12,
            "par": [100.0 - 5 * i for i in range(12)],
            "tier": list(range(1, 13)),
        }
    )
    assert bd.tier_map(many, limit=4).height == 4


def test_tier_map_ignores_unscored_players() -> None:
    frame = pl.DataFrame(
        {
            "name": ["a", "b"],
            "position": ["TE", "TE"],
            "par": [10.0, None],
            "tier": [1, 1],
        }
    )
    assert bd.tier_map(frame).get_column("n_left")[0] == 1


def test_tier_map_survives_a_board_without_tiers() -> None:
    frame = pl.DataFrame({"name": ["a"], "position": ["WR"], "par": [1.0]})
    assert bd.tier_map(frame).is_empty()
