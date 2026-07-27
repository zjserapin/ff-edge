# ff-edge

A free-data foundation for 2026 redraft fantasy football research. No paid
subscriptions, no scraping anything behind a login. Everything here comes from
nflverse, Sleeper's public API, and Fantasy Football Calculator's open REST
endpoint.

This is a **data layer**, not a ranking product. It gets clean, joined, cached
data onto your disk so you can ask your own questions.

## Setup

```bash
uv sync
uv run python -m src.bootstrap --light   # ~2 min cold, hydrates everything but pbp
uv run python -m src.simulate --sims 4000  # ~3s, writes the strategy artifact
uv run streamlit run app.py              # the app: Landscape / Players / Strategy / Board
```

`uv run python -m src.bootstrap --light --analysis` does the cache and the
derived artifacts in one command. `uv run pytest` runs the suite (82 tests,
~6s); it reads the local cache and skips cleanly when cold.

To pull your own league data, export your Sleeper **display name** (not your
email) and re-run bootstrap. It's read from the shell rather than committed so a
personal handle doesn't end up in the repo:

```bash
export FF_EDGE_SLEEPER_USER=yourname
uv run python -m src.bootstrap --light
```

Without it the Sleeper section is skipped cleanly.

Drop `--light` to also pull play-by-play (~1GB, several minutes).

## Modules

**Data layer** — everything that touches a network, and everything that
normalizes what comes back.

| File | What it's for |
|---|---|
| `src/config.py` | Season window, league params, paths, cache TTLs. One place to roll the year. |
| `src/cache.py` | TTL-aware parquet/JSON cache. Every network pull goes through it. |
| `src/nflverse.py` | Cached wrappers over `nflreadpy` — production, opportunity, context. |
| `src/sleeper.py` | Read-only Sleeper client, including the `previous_league_id` walk. |
| `src/adp.py` | FFC ADP, daily snapshots, and pick-survival probability. |
| `src/ids.py` | The join layer. All cross-source ID and name matching lives here. |
| `src/bootstrap.py` | One command to hydrate the cache. Never dies on one bad source. |
| `src/peek.py` | Four worked examples that prove the joins hold. |
| `src/mcp_server.py` | FastMCP server so the same cache is queryable in conversation. |

**Analysis layer** — turns the cache into answers. Never touches a network
directly.

| File | What it's for |
|---|---|
| `src/scoring.py` | Sleeper scoring settings → points, replacement level, PAR. |
| `src/landscape.py` | How positional value has moved over time, in *your* scoring. |
| `src/features.py` | One row per player-season: usage shares and rates, never totals. |
| `src/archetypes.py` | Per-position clustering. Describes usage profiles; does not predict. |
| `src/breakout.py` | The beat-ADP backtest, with calibration against the ADP-only baseline. |
| `src/rookies.py` | Separate rookie model — draft capital, combine, vacated opportunity. |
| `src/simulate.py` | Monte Carlo draft + season sim comparing draft strategies. |
| `src/uncertainty.py` | Wilson, bootstrap, and season-clustered intervals. |
| `app.py` | Streamlit app: Landscape / Players / Strategy / Board. |

## MCP

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ff-edge": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/ff-edge",
        "run", "python", "mcp_server.py"
      ]
    }
  }
}
```

14 read-only tools: `my_leagues`, `league_rosters`, `draft_history`,
`transactions`, `adp`, `survival`, `adp_movement`, `points_over_expected`,
`usage_leaders`, `snap_trend`, `market_disagreement`, `player_lookup`,
`data_inventory`, `refresh`. Nothing can modify a league.

Run bootstrap once before first use so the tools answer from cache.

## What the analysis found

Four findings, two of them negative. The negative ones are the point — a
research tool that only ever confirms things isn't measuring anything.

**Prior-season usage does not predict beating ADP.** Out-of-sample AUC 0.401
against 0.472 for draft price alone, difference interval [-0.173, +0.028]. A
sub-0.5 AUC is usually a sign error, so: shuffling labels gives 0.497 across
twelve seeds (pipeline is correct), in-sample AUC is 0.630 (the fit finds
structure), and *tightening* regularization makes out-of-sample worse, which
overfitting noise does not do. The relationship reverses between training and
test seasons. Nothing inverts the model and calls it a signal.

**The rookie model works.** Out-of-sample correlation 0.568, 22% less error than
predicting the mean, consistent across all four positions. Draft capital does
nearly all of it; combine numbers are close to noise.

**Two quarterback-timing strategies beat drafting by ADP** — late-QB +3.6pp and
elite-QB +2.8pp on title rate, with Bonferroni-corrected intervals excluding
zero. Early-TE is significantly worse.

**But season choice dominates strategy choice.** Zero-RB averages 8.5 wins in
2022 and 5.1 in 2024. That 3.4-win swing is larger than any gap between
strategies within a season, and it matches the Landscape tab showing running
back value climbing after 2023.

**Usage has one dominant axis.** Silhouette peaks at two clusters for every
position and falls from there. The honest grouping is "featured" and "not" — the
one real exception is quarterback, where the split is rushing versus pocket.

## What's verified

Row counts from a real run on 2026-07-27. `uv run python -m src.bootstrap --light`
reported **38/38 pulls ok**, 57 MB cached.

Checks that run in the test suite:

- **Scoring is exact.** Our compiled half-PPR expression equals nflverse's
  independently-computed `fantasy_points_ppr - 0.5 * receptions` to 1e-15 across
  all 34,882 cached skill player-weeks. The one exception is real: RB Dare
  Ogunbowale kicked a field goal in 2023 week 9, and Sleeper scores by stat line.
- **ADP joins at 97-99%** on skill positions (the 82% figure is all-positions,
  dragged down by kickers and defenses that have no `gsis_id`).
- **The simulation hits its structural baselines**: ADP-following gets a 0.611
  playoff rate and 7.01 wins where the format guarantees 0.60 and 7.0.
- **Greedy lineup selection matches brute force** on the real slot structure.

| table | rows |
|---|---:|
| players | 25,035 |
| ff_playerids | 12,468 |
| crosswalk (deduped) | 11,904 |
| rosters 2026 | 2,930 |
| schedules 2026 | 272 |
| depth_charts 2022-25 | 666,634 |
| draft_picks | 12,927 |
| contracts | 51,741 |
| combine | 8,968 |
| ff_rankings | 6,391 |
| weekly_stats 2022-25 | 75,876 |
| season_stats (reg) | 7,967 |
| ff_opportunity weekly | 24,178 |
| ff_opportunity pbp_pass | 75,585 |
| ff_opportunity pbp_rush | 61,636 |
| snap_counts | 106,148 |
| injuries | 23,564 |
| participation | 187,421 |
| ftn_charting | 185,215 |
| adp ppr | 228 |

Also verified by running it:

- **Warm cache**: second `bootstrap.py --light` completes in 0.4s with no
  network calls.
- **Join rates**: ECR 3,933/6,391 = **61.5%**; ADP 187/228 = **82.0%**.
- **Survival math**: ADP 40 / stdev 3 → 0.0038 at pick 48; ADP 40 / stdev 14 →
  0.2839. Same price, ~75x difference in whether you can wait.
- **MCP**: all 14 tools register and return real data under `fastmcp.Client`.

The Sleeper section is verified too, against a live account (**33/33 pulls ok**):
user resolves, 1 league, 10 managers, 10 rosters. The `previous_league_id` walk
returned **470 picks across 3 prior seasons** (2023/2024/2025), joining to
nflverse at **96.0%**. `transaction_history` returned **1,984 transactions**
across 2023-2025 (1,071 waiver, 884 free agent, 32 trade). All 14 MCP tools
return real data, including the four league tools.

Note that both multi-season pulls run off the same `sleeper.league_chain()`
traversal, and that transactions deliberately span prior seasons — during the
offseason the current league is empty, so a single-season pull returns 0 and
tells you nothing about how these managers behave.

## Notes

**`nfl_data_py` is dead — use `nflreadpy`.** The old package pins `pandas<2.0`,
which fails to build on modern Python (`No module named 'pkg_resources'` during
the wheel build). `nflreadpy` is the maintained nflverse successor and returns
**polars**, not pandas. This project is polars end to end; nothing converts
internally.

**The ID dtype trap.** `load_ff_playerids()` types `sleeper_id`, `espn_id`, and
friends as `Int64`. The Sleeper API returns those same IDs as **strings**. Joining
across that mismatch doesn't reliably error — it can just match nothing and hand
back a plausible-looking frame. `ids.crosswalk()` casts every `*_id` column to
`Utf8` once so no downstream join has to think about it. This is the single most
likely way a project like this breaks quietly, which is why `ids.match_report()`
exists: print the rate, don't assume it.

The ~62% ECR match rate is expected and fine. The misses are dominated by IDP
(DL 614, DB 549, LB 187), DSTs (169, no `gsis_id` exists), and kickers — none of
which matter for redraft skill positions.

**`ff_opportunity` is the highest-value table here.** It carries `*_exp`
(expected production given the opportunity actually received) alongside actuals,
and `*_diff` = actual − expected. That's the volume-vs-efficiency split most
public projections blur together. Volume is far stickier year over year than
efficiency, so the sign of `*_diff` tells you which one you're buying.

**ADP stdev is the underused column.** Two players at ADP 40 with stdev 3 vs 14
are 0.4% and 28% likely to survive to pick 48. Identical price, completely
different decision. `adp.survival()` reframes the draft from "who's best
available" to "who do I lose if I wait" — which is the question you actually
have on the clock.

**`market_disagreement` has bias in both directions.** Raw `sd` scales with
rank, so sorting on it surfaces deep-league fliers. `rel_sd` (sd/ecr)
over-corrects and inflates the top of the board — on the real 2026 file a naive
rel_sd sort puts Ja'Marr Chase first on an sd of ~1. Neither sort is the answer.
Filter to an ECR band you'd actually draft in, or rank sd within tier.

**`previous_league_id` is the unlock.** Sleeper links each league season to its
predecessor, so `sleeper.draft_history()` walks the chain backwards and returns
years of *your specific leaguemates'* draft behavior from one current league ID.
No paid service sells that, because it's specific to your twelve people. Combined
with `all_transactions()` (FAAB aggression, hold times, positional bias), it's
the closest thing to an actual edge in this repo.

**Scoring is league-specific.** `sleeper.league()` returns `scoring_settings` and
`roster_positions`. Derive replacement level from those rather than assuming
generic PPR — a superflex league with 6-point passing TDs has a different RB2
than every public ranking assumes.

## ADP history

ADP is a moving target through August and the FFC API only serves *today*.
Nobody sells the history, so you have to accumulate it yourself. **It cannot be
backfilled** — `adp.movement()` returns empty until you have two days of
snapshots, and every day you don't run it is a day permanently missing.

```cron
0 6 * * * cd /absolute/path/to/ff-edge && uv run python -c "from src import adp; adp.snapshot(); adp.snapshot('half-ppr', 10)"
```

Start it now, not in August.
