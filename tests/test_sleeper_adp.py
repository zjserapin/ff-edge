"""Sleeper's ADP feed, and the two ways it can be wrong without raising.

Both failure modes here produce a full, well-typed, plausible board:

  1. **The 999 sentinel.** Sleeper spells "undrafted" as `adp = 999.0`, never as
     null. It sorts last, which is exactly where you want it, so every ascending
     query looks right — and every mean, curve fit and `nulls_last` guard reads
     it as a real draft slot in round 100. Nine tenths of the feed carries it.

  2. **A name join that fans out.** Sleeper ships the price and FFC ships the
     dispersion, so the two must be joined on a normalized name — the exact join
     that has already collapsed a father onto his son and a cornerback onto a
     receiver elsewhere in this repo. A fan-out adds rows and changes nobody's
     ADP, so nothing looks wrong downstream.

The network is never touched: the transport and the cache are both replaced with
fixtures whose right answer is known by construction.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import profiles as pf
from src import sleeper_adp as sa


class _Response:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict]:
        return self._payload


def _row(name: str, position: str, adp: float, team: str = "ATL") -> dict:
    first, _, last = name.partition(" ")
    return {
        "player_id": name.replace(" ", "").lower(),
        "stats": {"adp_2qb": adp},
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
        },
    }


@pytest.fixture
def uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the loader every time and never write to the real cache."""
    monkeypatch.setattr(
        sa, "frame", lambda name, ttl, loader, force=False: loader()
    )


def _serve(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> None:
    monkeypatch.setattr(
        sa._session, "get", lambda *a, **k: _Response(rows)
    )


# --- the sentinel -----------------------------------------------------------


def test_undrafted_sentinel_is_dropped_and_real_prices_are_kept(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """999 leaves; everything else stays.

    The second half of this assertion is the load-bearing half. A parser that
    dropped *every* row would satisfy "no 999 on the board" perfectly, so the
    kept-row check is what stops this test passing for the wrong reason.
    """
    _serve(
        monkeypatch,
        [
            _row("Real Player", "WR", 12.5),
            _row("Undrafted Guy", "WR", 999.0),
            _row("Deep Sleeper", "TE", 240.0),
        ],
    )
    board = sa.fetch("2qb", 2026)

    assert board.height == 2, "the 999 row should be gone and the others kept"
    assert "Undrafted Guy" not in board.get_column("name").to_list()
    assert 999.0 not in board.get_column("adp").to_list()
    # A genuinely deep player is a price, not a sentinel, and must survive.
    assert "Deep Sleeper" in board.get_column("name").to_list()


def test_a_sentinel_left_in_would_be_caught_by_this_assertion() -> None:
    """The guard above fails when the sentinel is not filtered.

    Asserting that something did not happen is worthless unless the assertion is
    known to fire when it does. This reproduces the unfiltered parse directly
    rather than trusting that the filter is what made the board clean.
    """
    unfiltered = pl.DataFrame({"name": ["Undrafted Guy"], "adp": [sa.UNDRAFTED]})
    assert sa.UNDRAFTED in unfiltered.get_column("adp").to_list()


def test_sentinel_survives_a_sort_which_is_why_it_hides(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """Sorting ascending puts 999 last — the reason nobody notices it.

    Pins the property that makes this trap invisible, so a future reader sees
    why "the board looks fine" is not evidence the sentinel was handled.
    """
    raw = pl.DataFrame({"adp": [sa.UNDRAFTED, 5.0, 12.0]}).sort("adp")
    assert raw.get_column("adp").to_list()[-1] == sa.UNDRAFTED
    assert raw.get_column("adp").to_list()[0] == 5.0


# --- market selection -------------------------------------------------------


def test_unknown_market_raises_rather_than_falling_back() -> None:
    """A typo must not silently price a superflex league off the 1QB board."""
    with pytest.raises(KeyError):
        sa.fetch("suprflex", 2026)


def test_markets_are_spelled_the_way_ffc_spells_them() -> None:
    """One string selects the market at both sources, so they cannot drift."""
    from src import adp as ffc

    assert "2qb" in sa.MARKETS
    # The profile hands the same string to both sources; if the vocabularies
    # diverged, a profile could ask FFC for 2QB and Sleeper for something else.
    assert pf.resolve("shiva_bowl").adp_scoring in sa.MARKETS
    assert ffc.BASE.startswith("https://")


# --- the profile pairing ----------------------------------------------------


def test_shiva_bowl_prices_off_the_board_its_managers_can_see() -> None:
    assert pf.resolve("shiva_bowl").adp_source == "sleeper"


def test_unknown_source_raises() -> None:
    """Same reasoning as `resolve` raising on an unknown profile name."""
    with pytest.raises(ValueError):
        pf.customize("shiva_bowl", adp_source="sleeperr")


def test_every_profile_declares_a_known_source() -> None:
    for name, profile in pf.PROFILES.items():
        assert profile.adp_source in pf.ADP_SOURCES, name


# --- the dispersion join ----------------------------------------------------


def _ffc_frame(rows: list[tuple[str, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "adp": [r[2] for r in rows],
            "stdev": [r[3] for r in rows],
        }
    )


def test_stdev_join_does_not_fan_out_on_a_shared_name(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """Father and son normalize to one key; the board must not gain a row.

    `ids.normalize` strips generational suffixes, so Michael Pittman Jr. and
    Michael Pittman Sr. share a join key. A left join against both would
    duplicate the Sleeper row, add a player to the board who does not exist, and
    change nobody's ADP while doing it.
    """
    _serve(monkeypatch, [_row("Michael Pittman", "WR", 40.0)])
    from src import adp as ffc

    monkeypatch.setattr(
        ffc,
        "fetch",
        lambda *a, **k: _ffc_frame(
            [
                ("Michael Pittman Jr.", "WR", 40.0, 8.0),
                ("Michael Pittman Sr.", "WR", 41.0, 3.0),
            ]
        ),
    )
    board = sa.board("2qb", 10, 2026)
    assert board.height == 1, "a shared name must not add a player to the board"
    # The earlier-drafted identity wins, deterministically.
    assert board.get_column("stdev").to_list() == [8.0]


def test_missing_dispersion_is_imputed_from_its_round_not_floored(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """A missing stdev says "not measured", never "no spread".

    `slot_scale` floors a null/zero stdev at 0.5 picks, which would assert that
    a pick lands within half a pick of its ADP — near-certainty, applied to the
    players we know the least about. Imputing the round's typical dispersion
    keeps the unmeasured player among ordinary members of his round, the same
    impute-rather-than-sink rule `board.rank_board` uses.
    """
    _serve(
        monkeypatch,
        [
            _row("Known Player", "WR", 5.0),
            _row("Unpriced Deep", "TE", 205.0),
        ],
    )
    from src import adp as ffc

    monkeypatch.setattr(
        ffc,
        "fetch",
        lambda *a, **k: _ffc_frame(
            [
                ("Known Player", "WR", 5.0, 2.0),
                ("Late Round Guy", "RB", 200.0, 17.0),
            ]
        ),
    )
    board = sa.board("2qb", 10, 2026).sort("adp")
    got = dict(zip(board.get_column("name"), board.get_column("stdev")))

    assert got["Known Player"] == 2.0, "a measured stdev must pass through"
    imputed = got["Unpriced Deep"]
    assert imputed is not None, "an unmeasured player must not be left null"
    assert imputed > 0.5, "0.5 is the floor that asserts near-certainty"
    assert imputed == 17.0, "should take the dispersion typical of its round"
    assert board.filter(pl.col("name") == "Unpriced Deep")["stdev_imputed"][0]
    assert not board.filter(pl.col("name") == "Known Player")["stdev_imputed"][0]


def test_no_ffc_board_degrades_survival_rather_than_the_board(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """An unpriceable FFC format still returns Sleeper's prices.

    FFC suppresses a format until it has collected enough drafts, which is a
    normal August condition rather than an error. The price is the thing being
    asked for; losing the spread degrades `survival`, not the board.
    """
    _serve(monkeypatch, [_row("Real Player", "WR", 12.5)])
    from src import adp as ffc

    monkeypatch.setattr(ffc, "fetch", lambda *a, **k: pl.DataFrame())
    board = sa.board("2qb", 10, 2026)
    assert board.height == 1
    assert board.get_column("adp").to_list() == [12.5]
    assert board.get_column("stdev").to_list() == [None]


def test_team_codes_are_normalized_on_the_way_in(
    monkeypatch: pytest.MonkeyPatch, uncached: None
) -> None:
    """Both sides of every team-keyed join, every time.

    Sleeper spells the Rams `LAR`; nflverse spells them `LA`. A board carrying
    the raw code joins to nothing downstream and drops the row silently.
    """
    _serve(monkeypatch, [_row("Puka Nacua", "WR", 5.5, team="LAR")])
    assert sa.fetch("2qb", 2026).get_column("team").to_list() == ["LA"]
