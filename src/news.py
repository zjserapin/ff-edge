"""Free news ingestion for the claims ledger: RSS, depth charts, trending.

Three sources, three trust levels, per CLAIMS_SPEC.md:

- **Google News RSS** is the beat-report firehose — headlines plus snippets,
  which is deliberately all v1 extracts from. Full-article scraping means
  paywalls, consent walls, and silent breakage; a headline+snippet is short,
  stable, and enough to carry a role-change claim with a verbatim quote.
- **nflverse depth charts** are structured — a rank change needs no LLM and
  becomes a machine-generated claim in `claims.from_depth_charts`.
- **Sleeper trending adds** are not a claim source at all: they are the
  *detector* that says "the market noticed something, find the claim." A
  player trending with no claim in the ledger is a to-do, not a signal.

Everything here returns plain frames; interpretation lives in `claims.py`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import polars as pl
import requests

from src import nflverse as nv
from src import sleeper
from src.cache import frame

GOOGLE_NEWS = "https://news.google.com/rss/search"

# Query terms that select role-change reporting over hot takes. Kept short:
# Google News treats each OR clause as a signal, and too many terms dilutes to
# generic team coverage.
ROLE_TERMS = '(depth chart OR starter OR "first team" OR snaps OR role OR injury)'

_session = requests.Session()
_session.headers.update({"User-Agent": "ff-edge/0.1 (personal research)"})

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", text or "").replace("&nbsp;", " ").strip()


def _parse_rss(xml_text: str) -> pl.DataFrame:
    """RSS items → frame. Separated from the fetch so tests feed it fixtures."""
    root = ET.fromstring(xml_text)
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source = (item.findtext("source") or "").strip()
        snippet = _strip_html(item.findtext("description") or "")
        published = (item.findtext("pubDate") or "").strip()
        try:
            pub_date = datetime.strptime(
                published, "%a, %d %b %Y %H:%M:%S %Z"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            pub_date = None
        rows.append(
            {
                "title": title,
                "snippet": snippet,
                "source": source,
                "url": (item.findtext("link") or "").strip(),
                "published": pub_date.date().isoformat() if pub_date else None,
            }
        )
    if not rows:
        return pl.DataFrame()
    # Google News commonly appends " - Source Name" to titles; the source tag
    # already carries it, so strip each row's own suffix to keep quotes clean.
    return pl.DataFrame(rows).with_columns(
        pl.col("title")
        .str.strip_suffix(pl.concat_str([pl.lit(" - "), pl.col("source")]))
        .alias("title")
    )


def team_news(team_name: str, force: bool = False) -> pl.DataFrame:
    """Role-change-flavored news for one team, from Google News RSS.

    Cached at the 1-hour "live" TTL — the daily pull is always past it, while
    re-running a pipeline twice in a row stays offline.

    Returns: title, snippet, source, url, published.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", team_name.lower()).strip("_")

    def _load() -> pl.DataFrame:
        resp = _session.get(
            GOOGLE_NEWS,
            params={
                "q": f'"{team_name}" {ROLE_TERMS}',
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return _parse_rss(resp.text)

    return frame(f"news_{slug}", "live", _load, force)


def all_team_news(force: bool = False) -> pl.DataFrame:
    """Every team's news, stacked, with the team abbreviation attached.

    One RSS request per team, sequential on purpose — 32 requests against a
    free endpoint is politeness territory, not a throughput problem.
    """
    teams = nv.teams()
    if not teams.height:
        return pl.DataFrame()

    name_col = "full" if "full" in teams.columns else "team_name"
    abbr_col = "team" if "team" in teams.columns else "team_abbr"
    parts = []
    for abbr, name in teams.select(abbr_col, name_col).unique().iter_rows():
        if not name:
            continue
        got = team_news(name, force=force)
        if got.height:
            parts.append(got.with_columns(pl.lit(abbr).alias("team")))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def depth_chart_snapshot(season: int, force: bool = False) -> pl.DataFrame:
    """Latest fantasy-position depth chart, normalized for diffing.

    The nflverse table is keyed by a `dt` snapshot timestamp, not by week, and
    WRs occupy three parallel starter slots — so `depth_rank` here is a
    player's *best* listed rank at his position: every starting WR is rank 1,
    and moving from 2 to 1 is a promotion into a starting slot, which is
    exactly the machine claim `claims.from_depth_charts` wants.

    Returns one row per (team, position, player): captured_at, team, position,
    depth_rank, player_name, gsis_id.
    """
    raw = nv.depth_charts([season], force=force)
    if not raw.height:
        return pl.DataFrame()

    latest_dt = raw.get_column("dt").max()
    return (
        raw.filter(
            (pl.col("dt") == latest_dt)
            & pl.col("pos_abb").is_in(["QB", "RB", "WR", "TE"])
        )
        .group_by("team", "pos_abb", "player_name")
        .agg(
            pl.col("pos_rank").min().cast(pl.Int32).alias("depth_rank"),
            pl.col("gsis_id").first(),
        )
        .rename({"pos_abb": "position"})
        .with_columns(pl.lit(str(latest_dt)).alias("captured_at"))
        .sort(["team", "position", "depth_rank"])
    )


def trending_adds(limit: int = 50, force: bool = False) -> pl.DataFrame:
    """Sleeper's most-added players — the market-noticed-something detector.

    Returns: sleeper_id, player_name, position, team, add_count.
    """

    def _load() -> pl.DataFrame:
        got = sleeper._get(f"players/nfl/trending/add?limit={limit}")
        if not got:
            return pl.DataFrame()
        trend = pl.DataFrame(got, infer_schema_length=None, strict=False).rename(
            {"player_id": "sleeper_id", "count": "add_count"}
        )
        players = sleeper.players_nfl()
        if not players.height:
            return trend
        keep = [
            c for c in ("player_id", "full_name", "position", "team")
            if c in players.columns
        ]
        return trend.join(
            players.select(keep).rename(
                {"player_id": "sleeper_id", "full_name": "player_name"}
            ),
            on="sleeper_id",
            how="left",
        )

    return frame("sleeper_trending_add", "live", _load, force)
