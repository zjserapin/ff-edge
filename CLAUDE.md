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
| `HANDOFF.md` | Start of a session. Current state, findings, what's next. |
| `README.md` | Setup, module map, what the analysis has verified. |
| `ANALYSIS_SPEC.md` | The analytical contract — what gets measured and how. |
| `CLAIMS_SPEC.md` | Design contract for the claims ledger. Marked built. |
| `src/config.py` | Any question about seasons, paths, league format, TTLs. It is heavily commented and is the source of truth. |

---

## Running things

**Shell exports do not reach the Bash tool.** `export FF_EDGE_LEAGUE_ID=...` in
the user's terminal is invisible here. Prefix every command that needs it:

```bash
uv run pytest                                          # 190 pass, 2 skip w/o a league
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_LEAGUE_ID=... uv run python -c "from src import board; print(board.build())"
uv run python -m src.bootstrap --light                 # daily cache refresh
```

Env vars that matter: `FF_EDGE_LEAGUE_ID`, `FF_EDGE_SLEEPER_USER`,
`ANTHROPIC_API_KEY` (news extraction only — depth-chart claims work without it).

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

**`config.ROOT` uses `parents[1]`, not `.parent`.** `src/config.py` lives in
`src/`. Using `.parent` repoints `DATA_DIR` at `src/data`, creates it, and
orphans the entire cache without raising.

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

**Both `ff_opportunity` pbp tables carry weeks 19-22 with no `season_type`.**
Filter weeks explicitly.

**FanDuel prop markets carry `handicap: 0` and hide the line in the runner
name.** `"Bijan Robinson Over 1150.5"` is where the number lives. Reading
`handicap` yields a full, well-typed board on which every line is 0.0. Two more
in the same feed: all 97 season-long *yardage* markets are priced -114/-114, so
de-vigging them returns exactly 0.500 — the absence of a signal, not a
probability — and `marketType` is a **bucket, not a position** (Bowers, Kittle
and Loveland are all filed under `WIDE_RECEIVERS`).

**Generational suffixes collapse father onto son.** `ids.normalize` strips
`Jr./Sr.`, so Michael Pittman Jr. (WR, 2020) and Michael Pittman Sr. (RB, 1998,
no `gsis_id`) share one key. Combined with defenders who share a name — Lamar
Jackson the CB, Justin Jefferson the rookie LB — a name-only join fanned 145
prop rows out to 151 without raising. `props.resolve_players` resolves one
identity per *player* rather than per row, which is what makes the row count
preserved structurally instead of by luck.

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
