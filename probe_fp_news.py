"""One-off probe: what does the FantasyPros news feed actually give per call?

Written 2026-08-10 to settle four questions the public docs do not answer. The
only machine-readable description of this endpoint found was a third-party MCP
wrapper that self-rates its own schema as incomplete, so **everything below is
unverified until this runs.** Delete this file once the answers are recorded.

Spends **5 calls** of a 50/day budget and refuses to run without `--confirm`,
because a probe that silently burns 10% of the daily budget is exactly the kind
of quiet failure this repo keeps writing tests against.

What each call is buying:

1. baseline      `limit=25`            — row shape: is there a player id? a timestamp?
2. cap probe     `limit=100`           — does it error, or silently clamp to 25?
3. partition     `category=injury`     — are category feeds disjoint, or one feed filtered?
4. pagination    `offset=25`           — undocumented, but costs one call to rule out
5. injuries      `/nfl/injuries`       — the bulk alternative: whole league in one call?

Question 3 is the one that matters most. If the five categories partition the
feed, a sweep is 5 calls for ~125 distinct items. If they are one feed filtered,
a sweep is 25 items and the categories buy nothing.

    FF_EDGE_FP_API_KEY=... uv run python probe_fp_news.py --confirm
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

BASE = "https://api.fantasypros.com/public/v2/json"
CALL_BUDGET = 5

# The wrapper implementation was seen hitting `/json/all/news` rather than a
# sport-scoped path, so try the documented shape first and fall back.
NEWS_PATHS = ("/nfl/news", "/all/news")


def _get(path: str, key: str, **params: Any) -> tuple[int, Any]:
    """One call. Returns (status, parsed-or-text) without raising on 4xx/5xx."""
    r = requests.get(
        f"{BASE}{path}",
        headers={"x-api-key": key},
        params=params,
        timeout=20,
    )
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:400]


def _items(payload: Any) -> list[dict]:
    """Find the list of news items whatever the envelope turns out to be."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("news", "items", "data", "results", "posts"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # Unknown envelope: surface the keys so the next run knows where to look.
        print(f"    (unrecognised envelope, top-level keys: {list(payload)[:12]})")
    return []


def _ids(items: list[dict]) -> set[str]:
    """Best-effort stable identity for an item, for the disjointness test."""
    out = set()
    for it in items:
        ident = it.get("id") or it.get("news_id") or it.get("url") or it.get("headline")
        if ident is not None:
            out.add(str(ident))
    return out


def main() -> int:
    key = os.environ.get("FF_EDGE_FP_API_KEY")
    if not key:
        print("FF_EDGE_FP_API_KEY is not set. Shell exports do not reach the")
        print("Bash tool — prefix the command instead:")
        print("  FF_EDGE_FP_API_KEY=... uv run python probe_fp_news.py --confirm")
        return 2

    if "--confirm" not in sys.argv:
        print(f"This spends {CALL_BUDGET} calls of a 50/day budget. Re-run with --confirm.")
        return 1

    spent = 0

    # 1. Baseline, and settle which path works.
    news_path = None
    baseline: list[dict] = []
    for path in NEWS_PATHS:
        status, payload = _get(path, key, limit=25)
        spent += 1
        print(f"[{spent}] GET {path}?limit=25 -> {status}")
        if status == 200:
            news_path = path
            baseline = _items(payload)
            break
        print(f"    {str(payload)[:200]}")

    if news_path is None:
        print("\nNeither news path returned 200. Nothing further to probe.")
        return 1

    print(f"    items returned: {len(baseline)}")
    if baseline:
        print(f"    row keys: {sorted(baseline[0])}")
        print("\n    --- first row, verbatim ---")
        print(json.dumps(baseline[0], indent=2)[:900])
        print("    ---------------------------\n")

    base_ids = _ids(baseline)

    # 2. Does limit>25 error, or silently clamp? A silent clamp is the dangerous
    #    answer: it looks like the feed simply had 25 items today.
    status, payload = _get(news_path, key, limit=100)
    spent += 1
    got = len(_items(payload))
    print(f"[{spent}] limit=100 -> {status}, {got} items", end="  ")
    print("(clamped to 25 — cap is real and silent)" if got <= 25 else "(>25! cap is soft)")

    # 3. The decisive one: do categories partition the feed?
    status, payload = _get(news_path, key, limit=25, category="injury")
    spent += 1
    inj = _items(payload)
    inj_ids = _ids(inj)
    overlap = len(base_ids & inj_ids)
    print(f"[{spent}] category=injury -> {status}, {len(inj)} items, {overlap} shared with baseline")
    if inj_ids and overlap == 0:
        print("    DISJOINT: a 5-category sweep is ~125 distinct items for 5 calls.")
    elif inj_ids:
        print(f"    OVERLAPPING: categories filter one feed; a sweep buys < 125.")

    # 4. Undocumented pagination. If offset works, coverage stops being capped.
    status, payload = _get(news_path, key, limit=25, offset=25)
    spent += 1
    off_ids = _ids(_items(payload))
    print(f"[{spent}] offset=25 -> {status}, {len(off_ids)} items", end="  ")
    if off_ids and not (off_ids & base_ids):
        print("(PAGINATION WORKS — undocumented, and it changes everything)")
    else:
        print("(ignored — same page back, no pagination)")

    # 5. The bulk alternative to polling a 25-item window.
    status, payload = _get("/nfl/injuries", key, season=2026, week=1)
    spent += 1
    inj_rows = _items(payload)
    print(f"[{spent}] /nfl/injuries?season=2026&week=1 -> {status}, {len(inj_rows)} rows")
    if inj_rows:
        print(f"    row keys: {sorted(inj_rows[0])}")

    print(f"\nspent {spent} calls. Record the answers, then delete this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
