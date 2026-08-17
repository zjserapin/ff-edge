# ff-edge

A free-data foundation for 2026 redraft fantasy football research. No paid
subscriptions, no scraping anything behind a login. Everything here comes from
nflverse, Sleeper's public API, and Fantasy Football Calculator's open REST
endpoint.

This is a **data layer**, not a ranking product. It gets clean, joined, cached
data onto your disk so you can ask your own questions.

## The question this project answers

One question, asked several ways: **beyond what the market already prices into
a player's draft slot, is there anything measurable that predicts how he does
against that price?** Not "who is good" — ADP already has an opinion on that —
but "where is the market's opinion wrong, and can that be shown rather than
argued."

Everything in `src/` is one stage of a pipeline built to answer it:

1. **Ingest** — pull production stats, opportunity shares, ADP, and news from
   free sources, and cache them so the pipeline never re-hits a network for
   data it already has (`nflverse.py`, `sleeper.py`, `adp.py`, `news.py`).
2. **Join** — put every source on the same player, by the same id, so a stat
   line and a market price refer to the same person (`ids.py`).
3. **Transform** — turn raw stats into shares and rates that describe a
   player's role rather than his box score (`features.py`, `context.py`).
4. **Measure, before modeling anything** — check whether each metric even
   repeats year over year. A metric that doesn't repeat is describing last
   season's variance, not the player, and is dropped before it can pollute a
   model (`stability.py`).
5. **Model and backtest** — ask the beat-ADP question for real, scored against
   draft price and validated on seasons the model never trained on
   (`breakout.py`, `projection.py`, `rookies.py`, `archetypes.py`).
6. **Simulate** — since a backtest on real usage says nothing about draft-day
   *strategy*, replay thousands of drafts and seasons to price each template
   against simply following ADP (`simulate.py`).
7. **Surface disagreement** — where this project's read and the market's price
   diverge, and by how much (`valuation.py`).
8. **Screen role changes** — the one place the model is provably blind
   (`HANDOFF.md` §3): it cannot predict who *gets* a bigger role. So that
   input comes from you, sourced and scored by an automated claims ledger, and
   the model only grades what a role change like that has historically been
   worth (`promotion.py`, `claims.py`).

`app.py` is a Streamlit shell over that pipeline — one tab per stage, roughly
in the order above (Landscape → Players → Screen → Strategy → Board), plus a
Glossary tab so no chart requires memorizing a column name.

**Most of what the project found is negative** — the model doesn't beat ADP,
clustering didn't hold up, most "skill" metrics don't repeat — and that's
reported as the finding rather than hidden. See "What the analysis found"
below for the numbers, or **[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)** for a
guided walkthrough of how the pieces connect if you're getting reoriented
after time away.

## Setup

```bash
uv sync
uv run python -m src.bootstrap --light   # ~2 min cold, hydrates everything but pbp
uv run python -m src.simulate --sims 4000  # ~3s, writes the strategy artifact
uv run streamlit run app.py              # the app: Big Board / Draft Day / Board / Research / Glossary
```

`uv run python -m src.bootstrap --light --analysis` does the cache and the
derived artifacts in one command. `uv run pytest` runs the suite (**361 tests,
~22s**, or 367 with a league id and Sleeper handle set); it reads the local
cache and skips cleanly when cold.

To pull your own league data, export your Sleeper **display name** (not your
email) and re-run bootstrap. Both of these are read from the shell rather than
committed, because Sleeper's API is public and unauthenticated — a league id in
a public repo lets anyone read that league's name, every manager's display name,
and its full draft and transaction history:

```bash
export FF_EDGE_SLEEPER_USER=yourname
export FF_EDGE_LEAGUE_ID=1234567890123456789   # optional; discovered if unset
uv run python -m src.bootstrap --light
```

Without them the Sleeper section is skipped and the app falls back to saved
league settings (10-team, half-PPR, FLEX + SUPER_FLEX), so everything still runs — you
just get the reference league's rules instead of your own.

The claims ledger's news extraction needs one more optional key. Without it,
the ledger still accumulates depth-chart claims and everything else runs:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # enables beat-report claim extraction
```

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
| `src/props.py` | FanDuel player prop lines — season-long today, weekly in-season. |
| `src/news.py` | Free news ingestion: Google News RSS, depth charts, Sleeper trending. |
| `src/llm.py` | The one place a model API is touched. Anthropic ↔ Bedrock swap is config-only. |
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
| `src/context.py` | Where the touches came from — red zone, goal line, TD equity, neutral script. |
| `src/stability.py` | Does a metric repeat next year? The gate every feature has to clear. |
| `src/archetypes.py` | Quality/opportunity scores and nearest-neighbour comparables. |
| `src/breakout.py` | The beat-ADP backtest, with calibration against the ADP-only baseline. |
| `src/projection.py` | The same question with a continuous target, which measures it ~5× tighter. |
| `src/rookies.py` | Separate rookie model — draft capital, combine, vacated opportunity. |
| `src/simulate.py` | Monte Carlo draft + season sim comparing draft strategies. |
| `src/valuation.py` | Quality against price — where this project disagrees with ADP. |
| `src/promotion.py` | The promotion screen: you name whose role is growing, it grades them by position-specific criteria and reports the base rate. |
| `src/claims.py` | The claims ledger: role-change claims scored by tier × specificity × novelty × recency, every flag decomposable to its quotes. See `CLAIMS_SPEC.md`. |
| `src/prompts.py` | Versioned system prompts — the claim-extraction contract lives here, not inline. |
| `src/uncertainty.py` | Wilson, bootstrap, and season-clustered intervals. |
| `src/glossary.py` | What every metric means. Feeds column tooltips and the Glossary tab. |
| `app.py` | Streamlit app: Landscape / Players / Screen / Strategy / Board / Glossary. |

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

16 read-only tools: `my_leagues`, `league_rosters`, `draft_history`,
`transactions`, `adp`, `survival`, `adp_movement`, `points_over_expected`,
`usage_leaders`, `snap_trend`, `market_disagreement`, `metric_stability`,
`touchdown_equity`, `player_lookup`, `data_inventory`, `refresh`. Nothing can
modify a league.

Run bootstrap once before first use so the tools answer from cache.

## What the analysis found

Several findings, most of them negative. The negative ones are the point — a
research tool that only ever confirms things isn't measuring anything.

**Opportunity persists year to year. Quality mostly doesn't.** The check that
should have come first: rank every qualified player within his position and
season, pair each player-season with his own next one, correlate the two
percentiles. No model, no outcome, just "does this measurement repeat".

| position | opportunity | quality |
|---|---:|---:|
| WR | 0.549 | 0.442 |
| RB | 0.526 | 0.278 |
| TE | 0.510 | 0.400 |
| QB | 0.472 | 0.394 |

This reorganized the feature sets. Six columns turned out to be noise dressed as
skill and were removed: `contested_catch_rate` (0.061 at WR, *negative* at TE),
`drop_rate` (0.096), `catch_rate` at running back and tight end, `ypt` at
running back. All four are things the fantasy literature treats as ability.
`ryoe_per_att` — rush yards over expected, added expecting it to be the running
back's yards per route run — persists at 0.202, barely above the floor. What
actually persists about a running back is whether his offense throws to him.

Selecting features this way uses no outcome data, so it is not the same as
searching for what scores well. It is now a module (`src/stability.py`) and a
test, so a noisy column cannot drift back in.

**Prior-season usage does not predict beating ADP, and this is now measured to
±0.01.** The original binary label — did he finish inside 60% of his price —
gave stratified AUC 0.528 against 0.493 for price alone, a gap of +0.035 with an
interval of [-0.014, +0.083]. That interval is too wide to conclude anything.

Keeping the outcome continuous instead (where did he finish among drafted
players at his position, scored by rank correlation) uses every pair rather than
only those spanning a threshold:

| | rank correlation |
|---|---:|
| model | 0.497 |
| ADP alone | 0.494 |
| **difference** | **+0.002, 95% CI [−0.008, +0.010]** |

Predicting next season's finish is easy — usage persists, so anything sane gets
to ~0.5. Beating the *price* is the question, and the answer is no, now with an
interval five times tighter. "No edge" measured to ±0.01 is a finding; "no edge"
measured to ±0.06 is an absence of evidence.

Things tried that did not help, recorded so they don't get re-run every August:

| approach | gap vs ADP |
|---|---:|
| play-context features (red zone, TD equity, RYOE) | −0.013 |
| quality/opportunity composite scores | −0.013 |
| Gaussian-mixture soft cluster membership | −0.022 |
| partial pooling across positions | −0.030 |
| PCA on the quality block | worse at every position |

The first is instructive. Those are real measurements that do persist — and
adding them still cost accuracy, because at two to three events per variable
another column buys less signal than it costs in variance. The binding
constraint is label seasons, not columns. Widening the window from 2020 to 2018
doubled the test folds and was worth more than every modelling change combined.

No position clears the conventional ten-events-per-variable floor (WR 6.5, RB
5.3, QB 3.3, TE 2.5), and the app reports that table above the results.

**Quarterback is the one place the model is actively harmful.** Its gap against
ADP is −0.099 with an interval of [−0.173, −0.033] — entirely below zero.
Rushing share and expected-points share are already fully priced at quarterback,
so adding them to a 77-row sample adds variance and nothing else.

**The rookie model works, and stratifying helps it too.** Out-of-sample
correlation 0.592 and 24% less error than predicting the mean. The per-position
coefficients are the clearest evidence stratification was worth doing: the RB
model keys on vacated *carries*, the WR model on vacated *targets* — signals a
pooled fit had to average together. Draft capital dominates everywhere; combine
numbers are close to noise.

**The 2026 superflex slot reprices quarterback more than any other rules change
this league has made.** One of the two FLEX slots became a SUPER_FLEX, which
puts quarterbacks into the marginal-starter comparison. League-wide QB demand
goes from 10 to 20, replacement quarterback falls from QB11 to QB21, and the
baseline drops 74 points — so every quarterback's PAR rises by that much.
Sorting 2025 by PAR, the top 14 goes from **one** quarterback to **six**, and
Josh Allen moves from 11th to 3rd overall. None of that is special-cased:
`starter_demand` fills the most restrictive slot first and asks the same
"who is worth more at the next open rank" question the FLEX slots go through.

**Two quarterback-timing strategies beat drafting by ADP** — late-QB +3.6pp and
elite-QB +2.8pp on title rate, with Bonferroni-corrected intervals excluding
zero. Early-TE is significantly worse. **These describe the 2024–25 two-FLEX
format.** The strategy simulator is deliberately pinned there (`SIM_ROSTER_POSITIONS`):
a strategy comparison needs the market and the templates to match the format,
and the board is priced from 1QB ADP while every template was written for two
FLEX slots. Unpinning it needs 2QB ADP boards (FFC has them back to 2020) and
QB-heavier templates, not new machinery — the lineup optimizer already handles
superflex and is brute-force tested against it.

**But season choice dominates strategy choice.** Zero-RB averages 8.5 wins in
2022 and 5.1 in 2024. That 3.4-win swing is larger than any gap between
strategies within a season, and it matches the Landscape tab showing running
back value climbing after 2023.

**Clustering on volume just rediscovers ADP.** The original feature set was
half volume — target share, snap share, air-yards share — which is exactly what
draft price already knows, and it dominated the distance metric so every
position collapsed to "featured vs not". Clustering on *per-opportunity quality*
instead (yards per route run, separation, yards after contact) finds real
structure — but not enough to be worth keeping. **The clustering was removed.**
Silhouette topped out at 0.19-0.29, which is a partition of a continuum rather
than a set of groups, and cluster membership added nothing downstream: fed to the
projection model as Gaussian-mixture soft memberships it scored 0.022 *below* ADP
alone. What survived is the quality/opportunity split it was built on top of,
plus nearest-neighbour comparables — which is what the clustering was a lossy
summary of.

The quality score weights each metric by how well it repeats rather than
averaging them flat, which moved its rank correlation with *next* season's
points from 0.464 to 0.502 at WR and 0.455 to 0.492 at TE.

**That split is what makes under/overvaluation detectable.** Quality and price
are each converted to a within-position percentile and subtracted. Elite,
expensive players land near the parity line — Puka Nacua is 98th percentile
quality at 100th percentile price, correctly priced. The interesting names are
off it: Dalton Kincaid at 100th-percentile TE quality and a 143 ADP; Luther
Burden III at 2.06 yards per route on a 3.8th-percentile target share. Quality
alone is not a buy signal, so a `path_score` requires somewhere to gain volume —
a 90th-percentile receiver who is already his team's alpha has none.

**What predicts a promoted player is position-specific, and at RB it is not
efficiency.** Among backups whose role then grew 10+ percentile points
(`src/promotion.py`, cohort 61 RB / 102 WR / 57 TE), prior efficiency at
running back reads as noise — YPC −0.15, yards after contact 0.00, broken
tackles −0.08 against next-season points — while the trust markers predict:
snap share 0.47, red-zone carry share 0.34, TD-equity share 0.32. At WR/TE the
efficiency logic holds (TE yards per route run 0.35). Quality terciles are a
filter, not a picker: promoted players hit a top-quartile finish 4.1% / 10.2% /
18.4% from bottom to top tercile. And nothing predicts the promotion itself
(vacated opportunity ~0), so the Screen tab takes the names from you and grades
them — it never claims to know who gets the job.

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
- **Playoff weeks and two-point tries stay out of season usage.** Both
  play-level opportunity tables carry weeks 19-22 with no `season_type` column,
  and 776 pass plays are two-point conversions snapped from the two — left in,
  they inflate red-zone share for whoever was targeted.
- **Opportunity out-persists quality at every position**, asserted rather than
  assumed. If it ever flips, the three-axis split is wrong and the valuation
  board is ranking noise.
- **No feature below the noise floor ships.** The six that failed are pinned in
  a test that fails loudly if one becomes usable.
- **Permuted labels score like a coin flip** in both backtests — 0.497 AUC and
  rank correlation under 0.2 — which is what separates "no signal" from
  "signal, inverted".

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
