# CLAUDE.md — ff-edge

Project-specific rules. Inherits `Projects/CLAUDE.md`; this file wins where they
conflict.

**Read this before touching any code, including in a subagent.** Most of what is
below describes ways to be *silently* wrong in this repo — bad joins that return
nulls instead of errors, sign conventions that are backwards from the obvious
one, columns that look like duplicates but aren't. None of it raises.

---

## The docs, and what each is for

| file | read it when |
|---|---|
| `DRAFT_CHECKLIST.md` | **What to do next.** Ordered by deadline, not by value. |
| `HANDOFF.md` | Start of a session. Current state around the checklist. |
| `RESEARCH_SPEC.md` | Product direction (2026-08-13) and the Phase 0/1/2 build order. **Read its correction block first** — three of its claims have been overtaken. |
| `BIG_BOARD_SPEC.md` | The Big Board tab. Spec, then eighteen numbered records of what each change to the board found. Why only two of the three original numbers are independent of ADP. |
| `FOOTBALLERS_SPEC.md` | The Fantasy Footballers layer. Data built, display half built. |
| `CLAIMS_SPEC.md` | Design contract for the claims ledger. Built; a 2027 asset. |
| `LEAGUE_ADP_SPEC.md` | Whether ADP can be made league-specific. **Read its resolution block first** — the answer arrived from a different direction than the measurement, and the `src/market.py` design it proposes was deliberately not built. Its value-vs-cost split is the part that still governs. |
| `HOW_IT_WORKS.md` | The pipeline's shape, stage by stage. **Its tab walkthrough describes the old six-tab app** and carries a status block saying so; the pipeline half is current. |
| `FANTASYPROS_IDEAS.md` | Brainstorm, not a plan. Its lead finding shipped (ECR is live); the seven-years-of-dispersion one has not been touched. |
| `README.md` | Setup, module map, what the analysis has verified. |
| `docs/archive/` | `ANALYSIS_SPEC.md`, `DASHBOARD_SPEC.md`, `DASHBOARD_SPEC_v2.md` — superseded on direction and layout. **Their findings sections still stand** and are worth reading before re-proposing anything they declined. |
| `src/config.py` | Any question about seasons, paths, league format, TTLs. It is heavily commented and is the source of truth. |

---

## Running things

**Shell exports do not reach the Bash tool.** `export FF_EDGE_LEAGUE_ID=...` in
the user's terminal is invisible here. Prefix every command that needs it:

```bash
uv run pytest                                          # 361 pass, 8 skip w/o a league
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run pytest   # 367 pass, 2 skip
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_LEAGUE_ID=... uv run python -c "from src import board; print(board.build())"
uv run python -m src.bootstrap --light                 # daily cache refresh
```

Env vars that matter: `FF_EDGE_LEAGUE_ID`, `FF_EDGE_SLEEPER_USER`,
`ANTHROPIC_API_KEY` (news extraction only — depth-chart claims work without it).

**`FF_EDGE_SLEEPER_USER` gates more than it looks like it does.** Without the
handle, `board.picks` resolves no owner, the Draft Day pick selector never
renders, and `tests/test_draft_day.py` skips — so the tab's most important
control goes untested while the suite reports green. Set both vars together.

### The daily bootstrap runs itself — check it, don't re-add it

A launchd agent runs `bootstrap --light` at **08:00 and 17:00 daily**. It lives
at `~/Library/LaunchAgents/com.zjserapin.ff-edge.bootstrap.plist` — **outside
the repo, deliberately**, because it carries the Sleeper handle and this repo is
public. Nothing about it is committed, so a fresh clone has no agent and a
future session should not assume one.

```bash
launchctl print gui/$UID/com.zjserapin.ff-edge.bootstrap | grep -E "runs|last exit"
tail -40 output/bootstrap-daily.log     # gitignored; last run's full inventory
launchctl kickstart -k gui/$UID/com.zjserapin.ff-edge.bootstrap   # run it now
launchctl bootout   gui/$UID/com.zjserapin.ff-edge.bootstrap      # stop it
```

**launchd rather than cron, for one specific reason:** cron silently skips a run
if the laptop is asleep at the scheduled minute, and `adp_history_*` cannot be
backfilled — a missed day is gone permanently. launchd runs a missed calendar
job when the machine next wakes. The twice-daily schedule is insurance on top of
that; `adp.snapshot` replaces the same-day row rather than appending, so the
second run refreshes the day with fresher numbers and never double-counts.

It needs **only `FF_EDGE_SLEEPER_USER`** — `bootstrap` discovers every league
from `sleeper.my_leagues()`, so no league id is stored anywhere on disk.

A run that logs anything other than `N/N ok` is a real failure. The most likely
causes are no network and an expired Sleeper handle, and both are silent in the
app rather than loud: the cache simply goes stale.

**Never use `uv run --no-sync` to work around a dependency problem.** Plain
`uv run` re-syncs from the lock on every invocation, which is the mechanism that
keeps the pyarrow exclusion below actually enforced. `--no-sync` exists here for
one purpose: deliberately installing a known-bad version to prove a guard fires.

Worktrees: `data/` and `.venv/` are gitignored, so a git worktree has neither.
14 of 16 test files touch the cache. **Anything that reads data runs in the main
tree, not a worktree.**

---

## Non-negotiables

- **The repo is public.** `data/` is gitignored and stays that way — a league id
  exposes nine other people's Sleeper data. League ids and handles come from the
  shell, never a committed file.
- **Never push without explicit approval in the current conversation.**
- **Season-forward validation only.** Never a random split. A random split leaks
  future seasons into the training window and every result becomes fiction.
- **`src/llm.py` is the only file that touches a model client.** Provider swaps
  happen in `config.py` (`LLM_PROVIDER`). Prompts live in `src/prompts.py`,
  versioned, never inline.

---

## Data traps — every one of these fails silently

These were each found the hard way. Do not reintroduce them.

**pyarrow 25.0.0 segfaults the app, and 270 tests pass against it.** Excluded in
`pyproject.toml`; fixed upstream in 25.0.1. Every table in `app.py` is polars,
and `st.dataframe` round-trips each one through pandas and back into Arrow bytes
for the browser. On 25.0.0 that corrupts memory and takes the interpreter with
it — SIGSEGV, no traceback, no Streamlit error page, the server just gone.

Worth internalising *how* it hid, because it generalises past this one release:

- It needed **repetition**, not an input. Rendering once passed. Rendering nine
  times, changing the pick each time, died — in the picks table, whose contents
  are identical on every rerun.
- It was **order dependent**. The same fifteen picks walked in reverse never
  crashed once. That is heap layout, so any test free to choose a convenient
  order would have stayed green.
- The whole unit suite passed against a build that could not survive a draft.

`tests/test_draft_day.py` is the answer to that class and exists for this
reason: it drives the real widget, forward, the whole list. **A failure there
may not look like a failure** — pytest exits 139 with no summary. A test run
that vanishes is a red result.

**A Streamlit misuse warning is invisible to all four of the obvious asserts.**
Passing a widget both a default *and* a `key` whose value is already in session
state prints a warning on every render. It is not an exception, not an
`st.warning` element, and not a `warnings.warn`, so `at.exception`, `at.warning`
and `pytest.warns` all stay green. **`caplog` does not see it either**:
`streamlit.elements.lib.policies` sets `propagate = False`, and pytest's capture
handler lives on the root logger. Only a handler added to that logger *by name*
catches it, and it has to be attached **before the first render** — the panel
seeds session state ahead of building the widget, so the warning fires on render
one rather than after a drag.

`test_driving_the_horizon_does_not_warn_about_the_default` is the worked example.
Three earlier versions of it shipped green against the broken spelling. The
general rule: **when asserting that something did not happen, prove the assert
fails when it does** — the same discipline `tests/test_nflverse.py` applies to
the preseason guard.

**Asking nflverse for a season that has not kicked off raises, and it takes the
whole app.** Eleven of the loaders wrapped in `src/nflverse.py` reject a future
season — `ValueError: Season must be between 2006 and 2025` — and two more 404.
`src/nflverse._played` now filters those seasons out *before* the cache key is
built, so they return an empty frame, which is what every caller here already
handles. **Do not remove that guard, and add it to any new wrapper** over a
game-data loader. Forward-looking feeds are exempt and must stay exempt:
schedules, rosters, depth charts and draft picks all have real 2026 rows.

Use `nfl.get_current_season()`, the games-based answer. Its `roster=True`
variant flips on March 15 and reports 2026 today — the same roster-runs-ahead
disagreement as the `AZ`/`ARI` split below.

Worth studying *how this one hid*, because nothing about it was silent:

- **The handling was already written.** `promotion.weekly_trust` guards with
  `if not rush_raw.height`, and `claims.resolve` already treats an empty week
  table as *pending, not failed*. Both sat one line after the call that raised.
  The author assumed a future season came back empty rather than throwing.
- **It was dormant until data arrived.** `claims.resolve` returns early on an
  empty ledger, so the path was unreachable until `bootstrap` pulled the first
  150 claims. The bug shipped on the day the ledger stopped being empty — nine
  days before the draft — and nothing about that day touched the code.
- **It was invisible to a green suite**, because the app-level tests were the
  only ones that render `app.py`, and an unhandled exception in a Streamlit tab
  kills the entire page rather than the section.

The general lesson, and it is the same one `tests/test_draft_day.py` teaches
from the other direction: **a code path guarded by `if not x.height` is untested
until something makes `x` non-empty.** A cache that fills, a ledger that
accumulates, or a season that starts will each do that on their own schedule.

**`config.ROOT` uses `parents[1]`, not `.parent`.** `src/config.py` lives in
`src/`. Using `.parent` repoints `DATA_DIR` at `src/data`, creates it, and
orphans the entire cache without raising.

**An unset `FF_EDGE_LEAGUE_ID` does not mean "no league" — it means "some other
league".** `scoring.resolve_league_id` falls through to `sleeper.my_leagues()`
and takes the **first row**, and on this account that row is **The Jungle** —
the dynasty startup that is out of scope by decision — not the Shiva Bowl.
Nothing about the result looks wrong: the board builds, prices, ranks and
renders, using one league's keepers and rosters under whatever profile is
selected. Found 2026-08-18 while verifying the website, where the board came up
carrying a "no keepers are declared yet" warning that was true of The Jungle
and false of the league being drafted.

It is the `adp.movement` wrong-market defect one level up, and worse: the market
bug mispriced a screen, this misprices **the whole board and every panel on
it**. The two failure modes compound — a discovered league with a mismatched
profile is two independent wrongs that each look plausible alone.

`web.data.league_identity` is the mitigation on the website: the header names
the resolved league on every page and marks it **⚠ guessed** when the id was
discovered rather than set, listing the leagues that were passed over.
**`app.py` has no equivalent** — its sidebar says "Live from Sleeper" without
saying *which*, so the Streamlit app remains blind to this. Set
`FF_EDGE_LEAGUE_ID` explicitly for anything that matters, and read the tag
before trusting a board. The id itself comes from the shell and is never
written down here — see the first non-negotiable above.

**`spread_line` in nflverse is positive when the *home* team is favoured** —
the opposite of betting convention. Read `spread` from
`expected.team_environment`, never `spread_line` raw.

**Team codes disagree across sources, and nflverse disagrees with itself.** FFC
says `LAR` where nflverse says `LA`. Worse, the **2026 roster feed alone spells
Arizona `AZ`** — the 2025 rosters, weekly stats, schedules and draft picks all
say `ARI`. Both are mapped in `ids._PFR_TEAMS`.

Run **both sides** through `ids.normalize_team`, **every time**, and note that
"both sides" includes a bare equality check, not just a `join`. The live bug was
`rookies.py` deciding whether a player changed teams with
`new_team != team` — roster on one side, weekly stats on the other. Every
Cardinal who stayed read as *gone*, so the whole team's production landed in the
vacated pool: Arizona's vacated target share came out 1.00, first in the league,
against a corrected 0.209 that is *below* the league mean. The analysis pointed
at the opposite of the truth.

The failure mode is a null row or a silently inverted flag, never an exception.
A test that normalizes one side and compares against a raw feed is testing half
the join and will pass right up until the day a feed changes.

**A traded pick's `roster_id` is whose pick it *originally* was**, not who holds
it now. Reading it the other way puts picks in entirely the wrong rounds.

**The nflverse teams table has 36 rows, not 32** — it carries relocated
franchises. Key team sheets off the season schedule instead.

**`rank(descending=True)` gives rank 1 to the *largest* value.** Check the
direction every time; ADP rank and points rank run opposite ways.

**`sort(descending=True)` puts nulls FIRST.** Polars defaults `nulls_last=False`,
so a "best N" query returns the N rows that could not be scored — deep players
with no ADP-curve match, every time. Pass `nulls_last=True` on any descending
sort meant to surface a top. `expected.tiers` drops null values before sorting,
which is why the real board is unaffected and an ad-hoc query is not.

**But `nulls_last=True` on a *tiebreak* key is a different thing, and there it
silently demotes.** The two uses look identical and are opposites. On a primary
sort it keeps unscored rows off the top, which is right. On a secondary key it
ranks "we did not measure him" as "worst of his group" — which `rank_board`
promised in its own docstring not to do while doing exactly that. All seven
unscored players inside the 2026 roster-demand line sat last in their block, and
Brock Purdy, the **highest** `par_env` in his, sat third of three.

The fix is to **impute rather than sink**: `rank_board` fills a missing
`quality_pct` with its block's median, so an unmeasured player lands among the
block's typical members, and only an exact tie in every key falls back to
preferring the measured player. Ask which claim the null is making — *"not on
the board"* belongs last, *"not measured"* belongs in the middle.

Related, same function: **a stable sort makes incoming row order the real last
tiebreak**, which is an invariant nothing enforces. `rank_board` now names
`value_col` as an explicit final key so an upstream join that reorders rows
cannot reorder the board.

**Both `ff_opportunity` pbp tables carry weeks 19-22 with no `season_type`.**
Filter weeks explicitly.

**FanDuel prop markets carry `handicap: 0` and hide the line in the runner
name.** `"Bijan Robinson Over 1150.5"` is where the number lives. Reading
`handicap` yields a full, well-typed board on which every line is 0.0. Two more
in the same feed: all 97 season-long *yardage* markets are priced -114/-114, so
de-vigging them returns exactly 0.500 — the absence of a signal, not a
probability — and `marketType` is a **bucket, not a position** (Bowers, Kittle
and Loveland are all filed under `WIDE_RECEIVERS`).

**The Shiva Bowl board is priced from Sleeper's ADP, not FFC's, and Sleeper
spells "undrafted" as `999.0` rather than null.** 8,537 of the 9,414 rows in the
2026 feed carry the sentinel. Nothing raises, and it *sorts correctly* — 999
lands last, exactly where an undrafted player belongs — so every ascending query
looks right while any mean, curve fit, or `nulls_last` guard reads it as a real
draft slot in round 100. `sleeper_adp.fetch` drops it on ingest so no caller has
to remember. **A sentinel that sorts correctly is the most dangerous kind**; the
`nulls_last` entry below is the same lesson from the other direction.

The swap itself (2026-08-20) was a source change rather than a correction, and
the argument for it is causal, not statistical: **the other nine managers are
looking at Sleeper's numbers while they pick.** FFC's board is seen by nobody in
this league, so Sleeper's ADP does not merely predict this draft better — it
partly creates it. `draft.metadata.scoring_type` is `"2qb"`, so the feed is
superflex-aware for this league by construction.

The disagreement is large and runs **both ways**, which is why no per-position
offset could have fixed it: inside FFC's top 150, TEs sit a median **25 picks
earlier** on Sleeper and QBs **27 later**. Trey McBride is FFC 65.7 against
Sleeper 19.8. That TE gap independently reproduces the one
`LEAGUE_ADP_SPEC.md` measured from three seasons of real Shiva Bowl picks, which
is why the `src/market.py` bias correction proposed there was **not built** —
two unrelated methods agreeing FFC is wrong about tight ends means stop reading
FFC, not patch it.

**What did *not* move: the historical ADP→points curve.** ADP does two jobs
(`LEAGUE_ADP_SPEC.md` is the long version). Value — "what does a market's TE1
score" — is a function of positional *rank* fit across `LABEL_SEASONS`, and FFC
covers all seven where Sleeper has no `adp_2qb` before 2020. Cost — "who occupies
that rank and what will he cost" — is what was broken and what moved. Rank is
rank, so an FFC-fit curve applies to Sleeper ranks without contradiction. Moving
the curve too is defensible; **measure it against the board, don't assert it.**

Consequence worth knowing before reading a screen: **board ranks barely moved
(TE median −1.5) while prices moved enormously.** The two sources largely agree
on ordering *within* a position and disagree on cross-position price, so the
swap fixed the cost side, which is the side that was wrong. Also note **the
movement panels in `app.py` and `web/` still read FFC history** and are
therefore a different market from the board they sit beside. Sleeper history
began accumulating 2026-08-20 via `sleeper_adp.snapshot` and, like every ADP
history here, cannot be backfilled.

**Sleeper publishes no dispersion, so `stdev` still comes from FFC** by
normalized-name join — the answer `config.py` had already written down when it
sketched this swap for FantasyPros. Where FFC has no row at all (Sleeper's board
is ~3.7x deeper) the stdev is **imputed from the round's typical dispersion, not
floored**: `slot_scale`'s 0.5 floor would assert a pick lands within half a pick
of its ADP, i.e. near-certainty about the players we know least about. The first
implementation forward-filled over the *joined board* instead of a dense round
index, which handed a round-21 player round 1's dispersion — caught by
`test_missing_dispersion_is_imputed_from_its_round_not_floored`, which is the
worked example of why that test exists.

**`player_id` means two different things, and they are different types.** The
draft board's `player_id` is **FFC's, an Int64**; `archetypes.scores`,
`features.build` and the nflverse tables key on **gsis_id, a String**
(`00-0034857`). Joining one to the other without renaming leaves two columns
sharing a name, polars silently suffixes the newcomer to `player_id_right`, and
reading `player_id` back hands you the wrong namespace. It surfaces downstream as
`[(col("player_id")) == (dyn int: 5683)]` against a String column, a long way
from the join that caused it.

**Alias on the way in** — `pl.col("player_id").alias("gsis_id")` — rather than
relying on join suffixes. `board.attach_quality` sidesteps this entirely by
joining on the normalized name and never carrying an id across; that is the
pattern to copy.

**Generational suffixes collapse father onto son.** `ids.normalize` strips
`Jr./Sr.`, so Michael Pittman Jr. (WR, 2020) and Michael Pittman Sr. (RB, 1998,
no `gsis_id`) share one key. Combined with defenders who share a name — Lamar
Jackson the CB, Justin Jefferson the rookie LB — a name-only join fanned 145
prop rows out to 151 without raising. `props.resolve_players` resolves one
identity per *player* rather than per row, which is what makes the row count
preserved structurally instead of by luck.

**Two point columns above their own replacement level are still not on the same
scale.** `board.par` and `board.ffb_par` are both league points above the
position's replacement, which makes them look directly averageable. On the 2026
board `ffb_par` carries **1.55x the dispersion** of `par` by IQR (1.76x by
standard deviation), because the ADP curve maps a whole positional rank onto one
fitted number and flattens at the top — the top four backs all price at exactly
72.6 — while three analysts projecting touches spread the same four from 84 to
157. A raw weighted average at `weight=0.5` therefore hands the Footballers
about 60% of the say, and the ratio is **2.2x at quarterback**, so the tilt is
worst at the position a superflex league is most sensitive about.
`board.blend_par` standardizes each side before blending and maps the result
back onto the `par` scale. Scale first, then weight.

**And center per position, not globally — the two are separable and conflating
them was a live bug.** Standardizing on one global center is right about the
*spread* (rescaling each position to a common width would assert the best tight
end is worth the best quarterback) and wrong about the *level*. The two sources
disagree about where a whole position sits: on the 2026 board the median
`ffb_par` minus `par` ran **+8.7 at TE against -16.2 at QB**, -9.2 at RB and
-6.6 at WR. A global center leaves that intact and it lands on the blend as a
uniform per-position shift — which is by construction not an opinion about any
player. So the center comes from each position's own median, handing the
cross-position level to `par` alone, and the scale stays global so positions
keep their own spreads.

The scale is then taken from **position-centered residuals**, not the raw
columns, or the same offset leaks back through the denominator after being
removed from the numerator.

Note the offset survives `attach_footballers` already subtracting each system's
own replacement level, because the two replacement levels sit at different
points on differently shaped curves and the board holds each position to a
different depth — 18 TEs against 67 WRs.

**This did not fix the tight-end promotion, and do not re-tune the weight trying
to.** Measured before and after: median TE shift against ADP **+47.5 either
way**; QB improved from -18.0 to -12.0. Ranking on raw `par` alone already
shifts TEs **+47.0**, so the cause was PAR comparing positions the board cuts to
unequal depth. That is fixed separately, below.

**PAR below replacement is not an ordering, and sorting on it across positions
was the board's largest distortion.** A player under his position's replacement
contributes **zero** starting-lineup points no matter how negative the number —
you would start the freely available replacement instead. So -15 and -35 are the
same decision, and ranking one above the other asserts a distinction PAR cannot
support.

It was not harmless. The board holds **67 receivers to 18 tight ends**, cut at
different distances from their own replacement: median `par` -17.9 at WR against
**0.0** at TE. A cross-position sort therefore interleaved TE15 ahead of WR55 and
promoted *every* tight end a median **+47.5 places** over ADP, never fewer than
+23, putting all 18 inside the top 100 of a league that rosters about 13.

`board.roster_demand` cuts the board at the line and **asserts no new constant**:
`replacement()` already derives `replacement_rank` per position from the roster
shape and the keepers, and the first player past it *is* the replacement. Above
the line the board ranks across positions; below it the order is **ADP's**, and
`block` is null so the reader can see where the tool stops claiming. Measured
after: TE median shift **0.0**, and the 8 tight ends inside demand land at board
ranks 22-60 with the next at 105 — "get one by pick 60 or punt", which is a
draft plan rather than a distortion.

`pos_rank` is taken on the ranking column, not ADP: the cap governs **how many**
of a position get compared across positions, never **which**.

**League scarcity is not personal need, and modelling only the first told the
user to draft the position he already owned.** `roster_demand` subtracts
league-wide keepers from league-wide demand. On the 2026 board that leaves
quarterback at 7 slots against 20 — genuinely scarce, and completely irrelevant
to Zach, who **keeps Jayden Daniels and Trevor Lawrence** against a roster
carrying exactly two quarterback-capable slots (`QB` and `SUPER_FLEX`). His
quarterback need is zero. The cost-of-waiting panel ranked quarterback first and
recommended one at **pick 4**.

The failure mode generalises and it is nastier than it looks: **league scarcity
is maximally misleading to the manager who caused it.** The teams that make a
position scarce by keeping it are exactly the teams that must not draft it, so a
tool reporting only league demand gives its worst advice to the person holding
the keepers — which in a keeper league is every user of it.

`board.roster_need` is the second quantity: starting slots filled most
restrictive first from *your* keepers, leaving `slots_open` per position. It
feeds the `need` column on the board and gates which positions the
cost-of-waiting recommendation will name. **Never gate the board's rows on it** —
a position you cannot start still carries bench and trade value, and where the
quarterbacks sit is what your leaguemates are about to spend picks on.

`slots_open` double counts a flex slot across every eligible position on purpose,
because it answers "can he start for me". `starters_left` is the un-double-counted
total. Do not "fix" either into the other.

Read the correlation honestly. Whole-board Spearman against ADP *rose* from
+0.885 to +0.963, because 98 of 158 rows are now ADP by construction. Inside the
line it is **+0.820**. The board gave up an opinion it could not support about
the bottom two thirds and kept its disagreement where picks are decided; a
falling headline correlation would have been the wrong thing to optimize.

Related, and the reason that standardization uses median/IQR rather than
mean/sd: **the two sources are censored differently in the tail.** A backup
quarterback really is ~270 points below a superflex replacement and the
Footballers say so; the ADP curve floors out near -30 and cannot. Twelve
undraftable quarterbacks were setting the scale for the whole board.

**The Fantasy Footballers' panel is not evenly fresh.** `updated_at` is per
analyst per player. On 2026-08-10 the median projection was 7 days old for
Jason, 64 for Andy and **90 for Mike**. A plain mean blends an August opinion
with a May one and prints one confident number. `footballers.panel_report()` is
the `ids.match_report` of that feed — look at it. Coverage is uneven too (302
players with all three analysts, 7 with two, 4 with one), so `n_analysts` is on
every consensus row and `FOOTBALLERS_MIN_ANALYSTS` blanks a thin panel rather
than averaging over a different set of people for every player.

More play-level traps (two-point plays inflating red-zone share, `play_id` type
mismatch between ff_opportunity and FTN, null receivers on 4% of pass plays) are
documented in the `src/context.py` module docstring. Read it before doing
anything play-level.

**The `ff_opportunity` *pbp* rush table has duplicate expectation columns** —
`rushing_td_exp`/`rush_touchdown_exp` and `rushing_yards_exp`/`rush_yards_exp`.
They disagree on 4,558 rows, and it is not noise: the `rush_*` pair carries
sentinels (0.0, -1.0) where the `rushing_*` pair has a real modeled value. **Use
the `rushing_*` columns.** See `src/context.py:48`.

Note this is the *pbp* table only. The **weekly** ff_opportunity table is a
different column namespace, and `scoring.EXPECTED_COLUMNS` correctly reads
`rush_touchdown_exp` there. Don't "fix" it.

---

## Codebase conventions

- **Polars, not pandas.** All 25 data modules use `polars as pl`. pandas is a
  transitive dependency only — do not introduce it into `src/`.
- **`config.py` is the source of truth.** Never hardcode a season, a path, a
  roster slot, or a scoring value. Rolling to a new season should be a one-line
  edit, not a grep.
- **Comments explain why, not what.** This codebase's existing comments are the
  house style — they record the reasoning and the traps, not the syntax. Match
  that density.
- **Tests brute-force claims rather than asserting them** where feasible. See
  `test_lineup_selection_is_optimal`, which verifies the greedy flex allocation
  against exhaustive search instead of trusting it.
- **Corrections stay checkable.** `board.compare_baselines` runs the keeper
  adjustment both ways on purpose — if a correction ever reads near zero, the
  complexity is not earning its keep and should come out.

### Formats come from `src/profiles.py`, never from loose arguments

A `LeagueProfile` carries a roster format **and the ADP market that prices it**
as one object, because the 2026 superflex bug was exactly what happens when
those two drift apart. Never set a roster format without setting the market;
`profiles.customize` exists for one-off variants.

`adp_source` is the third half of that same pairing: **`"ffc"` or `"sleeper"`**,
validated in `__post_init__` so a typo raises instead of falling back. The Shiva
Bowl is `"sleeper"` — see the source-swap note in the traps below for why, and
note it governs **this season's board only**. The historical curve is FFC under
either source.

```bash
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
```

Two built-ins: `shiva_bowl` (default, live from Sleeper) and `standard_12`.
`resolve()` raises on an unknown name rather than falling back — a typo that
silently returned the Shiva Bowl would price a standard league as a superflex
keeper league and look entirely plausible doing it.

`board.build()` returns `warnings` alongside the frames. An unpriceable profile
must produce a sentence, not an empty DataFrame that looks like a network blip.

**Redraft only, deliberately.** Everything here prices a player against a market
for one season. Dynasty is a different question — which young players are worth
building on — and it turns on age curves and quality signals rather than this
season's price. Nothing in this project estimates an age curve, so a dynasty
profile would answer the redraft question under a dynasty label. Don't add one.

### League format is superflex as of 2026

`QB/RB/RB/WR/WR/TE/FLEX/SUPER_FLEX/K/DEF` + 5 BN, 10 teams, 0.5 PPR.

Flex slots fill **most restrictive first** (`config.FLEX_SLOTS`), which is what
makes greedy allocation exactly optimal when the eligibility sets nest. This
league's sets nest; `REC_FLEX` and `WRRB_FLEX` would not, and both functions say
so out loud.

**`config.SIM_ROSTER_POSITIONS` is deliberately pinned to the old two-FLEX
format.** The strategy simulator's board is 1QB ADP and its templates are
two-FLEX shaped, so running it under superflex would report that QBs are nearly
free *and* start twice. That pin is intentional — do not "fix" it without also
swapping in 2QB ADP and rewriting the templates.

---

## Analysis standards

- **Report intervals, not point estimates.** The binding constraint on this
  project is label seasons (2.5-6.5 events per variable across 831 labeled
  player-seasons), not algorithms.
- **A negative result is a result.** `breakout.py` and `projection.py` are both
  honest nulls and stay in the repo as such. Do not quietly delete or re-run a
  measured null with more parameters until it flips.
- **Fitted models have repeatedly failed to beat ADP here.** Before proposing
  one, say where the extra data is coming from. Week-level and play-level tables
  are orders of magnitude larger than the season-level label set; that is where
  learning has a chance.
- **Don't invent an ordering the data doesn't support.** The ADP curve is not
  monotone; isotonic regression pools the offending ranks rather than forcing a
  rank order. Tiering here is an empirical result, not a presentation choice.

---

## Working style for this project

The user is building this to *learn*, not just to have it. Per
`Projects/CLAUDE.md`: explain before building, block by block, and lead with the
analogy. That applies with extra force to the contextual-scoring and ML threads,
which he named as things he wants to understand rather than receive.

Dashboard work (`app.py`, 1800+ lines, 6 tabs) is a **spec exercise before it is
a build exercise**. Write the spec to a file and get agreement before editing.

*This file evolves. Update it when the traps or the format change.*
