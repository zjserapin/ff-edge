"""The website renders the board the modules build — or a sentence.

TestClient drives the real FastAPI app against the real cache, so this file
runs in the main tree like every other data-touching test file. The board is
built once at module scope (it is the expensive part) and every test reads it.

What is deliberately held to here:
- An unknown profile is a 404 with the sentence, never a plausible fallback —
  the same refusal `profiles.resolve` makes.
- A blank filter means "all", never "match nothing" — the empty <option>
  submits `signals=`, and [""] as a filter would silently empty the board.
- The CSV is the same filtered frame the page shows, with raw column names,
  because it is meant to be copied out and marked up (checklist A4).
"""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

from web import data as wd
from web.server import app

client = TestClient(app)


@pytest.fixture(scope="module")
def board():
    return wd.board()


@pytest.fixture(scope="module")
def players(board):
    frame = board["players"]
    if not frame.height:
        pytest.skip("no board to render — needs an ADP board for the season")
    return frame


def test_home_renders_a_board_or_a_sentence():
    r = client.get("/")
    assert r.status_code == 200
    assert ("board-region" in r.text) or ("No board yet" in r.text)


def test_home_shows_every_build_warning(board):
    """`board.build` promises a sentence instead of an empty frame; the page
    must not swallow it. Warnings render inside role=alert banners."""
    r = client.get("/")
    for warning in board.get("warnings", []):
        # Markdown bold survives as literal text in HTML; strip the markers
        # before asking whether the sentence is on the page.
        assert warning.replace("**", "")[:60] in r.text.replace("**", "")


def test_api_board_contract(board):
    r = client.get("/api/board")
    assert r.status_code == 200
    payload = r.json()
    assert set(payload) == {"data", "warnings", "meta"}
    assert payload["meta"]["rows"] == len(payload["data"])
    assert payload["meta"]["rows"] == board["players"].height
    if payload["data"]:
        assert set(payload["meta"]["columns"]) == set(payload["data"][0])


def test_unknown_profile_is_a_sentence_not_a_fallback():
    page = client.get("/?profile=definitely_not_a_league")
    assert page.status_code == 404
    assert "unknown profile" in page.text
    api = client.get("/api/board?profile=definitely_not_a_league")
    assert api.status_code == 404
    assert "unknown profile" in api.json()["detail"]


def test_csv_respects_the_position_filter(players):
    r = client.get("/board.csv?positions=RB")
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows, "an RB-only board should not be empty"
    assert {row["position"] for row in rows} == {"RB"}
    # Prove the assert can fail: the unfiltered board must carry more than RBs,
    # otherwise the filter check above passes vacuously.
    full = list(csv.DictReader(io.StringIO(client.get("/board.csv").text)))
    assert {row["position"] for row in full} != {"RB"}
    assert len(full) == players.height


def test_blank_filter_params_mean_all_not_nothing(players):
    """The empty <option value=''> submits `signals=`; that must be 'all'."""
    blank = client.get("/board.csv?signals=&positions=")
    full = client.get("/board.csv")
    assert blank.text == full.text


def test_board_partial_is_a_fragment_not_a_page():
    r = client.get("/partials/board")
    assert r.status_code == 200
    assert "<html" not in r.text
    assert "board-region" in r.text


def test_usage_columns_appear_only_when_toggled(players):
    from src import board as bd

    present = [c for c in bd._USAGE_COLUMNS if c in players.columns]
    if not present:
        pytest.skip("no usage columns on this board")
    probe = present[0]
    without = client.get("/board.csv")
    with_usage = client.get("/board.csv?usage=1")
    header_without = without.text.splitlines()[0].split(",")
    header_with = with_usage.text.splitlines()[0].split(",")
    assert probe not in header_without
    assert probe in header_with


def test_every_page_renders():
    for path in ("/", "/draft", "/board", "/player", "/research", "/reference"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "<h1" in r.text, path


def test_no_page_silently_drops_a_section():
    """A section that fails renders a banner naming the error; it never just
    disappears. The stability panel vanished for an afternoon behind a broad
    `except` — invisible on screen and green in the suite."""
    for path in ("/draft", "/board", "/research", "/player"):
        assert "failed to build" not in client.get(path).text, path


def test_research_carries_its_sections(players):
    r = client.get("/research")
    for heading in ("Screens", "Snap trend", "Does a metric mean the same thing"):
        assert heading in r.text, heading


def test_draft_day_carries_its_sections(players):
    r = client.get("/draft")
    for heading in ("Your picks", "The board", "Who is likely to be there"):
        assert heading in r.text, heading


def test_charts_are_sized_in_pixels_not_container(players):
    """`width: "container"` renders a blank box rather than raising — vega
    resolves it against a parent that has not laid out yet. Two charts shipped
    that way. Nothing on the site may reintroduce it."""
    import json
    import re

    for path in ("/draft", "/board", "/player?name=" + players.get_column("name")[0]):
        html = client.get(path).text
        specs = re.findall(r'class="vega-spec"[^>]*>(.*?)</script>', html, re.S)
        assert specs, f"no chart on {path}"
        for raw in specs:
            spec = json.loads(raw.replace("<\\/", "</"))
            for key in ("width", "height"):
                assert spec.get(key) != "container", f"{path} {key}"


def test_player_page_finds_a_player_and_says_so_when_it_cannot(players):
    name = players.get_column("name")[0]
    hit = client.get(f"/player?name={name}")
    assert hit.status_code == 200
    assert name in hit.text
    # Prove the assert can fail: a name that is not on the board must say so
    # rather than rendering an empty player page.
    miss = client.get("/player?name=Definitely+Not+A+Player")
    assert "is on this board" in miss.text
    assert "Definitely Not A Player" in miss.text


def test_player_page_never_prints_a_zero_for_an_unmeasured_column(players):
    """Blank means not measured. A zero would read as a real red-zone role."""
    import polars as pl

    if "rz_carry_share" not in players.columns:
        pytest.skip("no usage columns on this board")
    receivers = players.filter(
        (pl.col("position") == "WR") & pl.col("rz_carry_share").is_null()
    )
    if not receivers.height:
        pytest.skip("every receiver has a red-zone carry share")
    r = client.get(f"/player?name={receivers.get_column('name')[0]}")
    assert r.status_code == 200
    assert "Red-zone carries" not in r.text


def test_value_page_carries_both_scatters():
    """Quality-against-price and the sportsbook line are separate opinions built
    from unrelated evidence. Losing either quietly is the failure to guard."""
    board = wd.valuation()
    if not board.height:
        pytest.skip("no valuation board — needs ADP and a warm feature table")
    r = client.get("/board")
    assert r.status_code == 200
    assert "Quality against price" in r.text
    assert "What the sportsbook thinks" in r.text


def _flat(html: str) -> str:
    """Collapse whitespace before matching prose.

    Templates wrap sentences across lines for readability, so a phrase that
    reads as contiguous on screen is not contiguous in the source. Asserting
    against the raw text tests the line wrapping rather than the content.
    """
    import re

    return re.sub(r"\s+", " ", html)


def test_vegas_section_says_how_thin_the_feed_is(players):
    """FanDuel prices ~92 players season-long. A position with no lines must say
    'not measured', never render an empty chart as though it were an answer."""
    import polars as pl

    board = wd.valuation()
    if not board.height or "line_pct" not in board.columns:
        pytest.skip("no book lines cached")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = board.filter(pl.col("position") == pos)
        if not sub.height:
            continue
        text = _flat(client.get(f"/board?pos={pos}").text)
        priced = sub.filter(pl.col("line_pct").is_not_null()).height
        if priced:
            assert f"{priced} of {sub.height} {pos}s have a season-long line" in text
        else:
            assert "No sportsbook lines matched" in text


def test_vegas_trust_note_is_position_specific(players):
    """The QB caveat inverts the obvious reading — passing-yards lines flatter
    exactly the quarterbacks who are mediocre fantasy quarterbacks. It must not
    be shown under a position it was not written about."""
    qb = client.get("/board?pos=QB").text
    wr = client.get("/board?pos=WR").text
    if "No sportsbook lines matched" not in qb:
        assert "Weakest here" in qb
        assert "Weakest here" not in wr


def test_every_position_renders_on_the_value_page():
    board = wd.valuation()
    if not board.height:
        pytest.skip("no valuation board")
    for pos in ("QB", "RB", "WR", "TE"):
        r = client.get(f"/board?pos={pos}")
        assert r.status_code == 200, pos
        assert "failed to build" not in r.text, pos


def test_reference_filters(players):
    everything = client.get("/reference")
    filtered = client.get("/reference?q=replacement")
    assert everything.status_code == filtered.status_code == 200
    assert len(filtered.text) < len(everything.text)


def test_vendored_assets_are_served():
    for name in ("htmx.min.js", "vega.min.js", "vega-lite.min.js", "vega-embed.min.js"):
        assert client.get(f"/static/vendor/{name}").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_dropoff_partial_carries_a_spec_or_a_sentence(players):
    r = client.get("/partials/dropoff")
    assert r.status_code == 200
    assert ("vega-spec" in r.text) or ("No curve" in r.text)
    # A player name must never be able to close the script element early.
    assert "</script>" not in r.text.partition("vega-spec")[2].rpartition("</script")[0]


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_the_page_says_which_league_it_resolved():
    """Discovery picks the *first* league Sleeper returns, and on this account
    that is The Jungle rather than the Shiva Bowl — a well-formed board for a
    draft you are not having. The header has to name it either way."""
    identity = wd.league_identity()
    r = client.get("/")
    assert r.status_code == 200
    if not identity["resolved"]:
        assert "no league" in r.text
    elif identity["discovered"]:
        # The guessed state must be visibly flagged, not just labelled.
        assert "guessed" in r.text
        assert "league-tag guessed" in r.text
    else:
        assert "league-tag set" in r.text
        if identity["name"]:
            assert identity["name"] in r.text


def test_the_league_tag_is_on_every_page():
    """A tag that only rides on the board is a tag you miss on Draft Day."""
    for path in ("/", "/draft", "/player", "/research", "/reference"):
        assert "league-tag" in client.get(path).text, path
