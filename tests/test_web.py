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


def test_placeholder_pages_render():
    for path in ("/draft", "/player", "/research", "/reference"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "arrives in" in r.text


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
