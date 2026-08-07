# ff-edge — handoff

**Session date:** 2026-08-06
**Branch:** `measure-what-repeats`, 8 commits, **pushed to origin**.
**State:** 190 tests pass, 2 skip cleanly without a league. Working tree clean.
**The draft is 2026-08-22 at 19:00.** That is the deadline for anything in §1.

Read this before doing anything else. It is written to be the whole context.

---

## The one-paragraph version

Last session ended with a promotion screen proposed but unbuilt. This session
built it, then built the contextual layer behind it (an automated claims
ledger), then discovered the league had switched to **superflex** — which broke
the replacement math badly enough that quarterback demand was being counted at
half its real value — then built the ranking system on top: expected points
from an ADP curve, tiers cut by isotonic regression, and a draft board priced
against the players who can actually be drafted. The league is now wired in
live: **all 18 keepers, every traded pick, and the real draft slot come from
Sleeper**, and the board reflects them.

The most useful thing to know going in: **the user's own contextual reasoning
beat the model twice in one conversation**, and both times the data backed him
rather than the board. That is the theme of the next session.

---

## What is new, by module

| module | what it does |
|---|---|
| `src/promotion.py` | Grade a player whose role is growing, by position-specific criteria. Week-level trust markers. |
| `src/claims.py` | The claims ledger — role-change claims, scored, auditable, resolvable. |
| `src/news.py` | Free ingestion: Google News RSS, nflverse depth charts, Sleeper trending. |
| `src/llm.py` | The one place a model API is touched. Anthropic ↔ Bedrock by config. |
| `src/prompts.py` | Versioned prompts. The extraction contract lives here. |
| `src/expected.py` | Expected points by ADP rank, isotonic tiers, Vegas team environment. |
| `src/board.py` | The draft board: keepers, picks, PAR against the draftable pool, context. |

`CLAIMS_SPEC.md` is the design contract for the ledger and is marked built.

---

## Findings, in order of how much they should shape what you do next

### 1. The superflex slot was silently mispriced, and is now the largest single valuation fact

The league turned one FLEX into a SUPER_FLEX for 2026. `starter_demand` only
ever counted slots literally named `FLEX`, so the league was starting 70 players
out of an 80-slot roster and QB demand was pinned at one per team.

Fixed by giving every flex type its eligible positions (`config.FLEX_SLOTS`) and
filling **most restrictive first** — which is what makes the greedy allocation
exactly optimal when the sets nest, and `test_lineup_selection_is_optimal`
brute-forces it rather than asserting it.

Effect: QB demand 10 → 20, replacement QB11 → QB21, baseline −74 points, and the
top 14 of 2025 by PAR goes from one quarterback to six.

**The strategy simulator is deliberately pinned to the old format**
(`config.SIM_ROSTER_POSITIONS`) and says so on the tab. Its board is 1QB ADP and
its templates are two-FLEX shaped, so running it under superflex would report
that QBs are nearly free *and* start twice. FFC publishes 2QB ADP back to 2020,
so unpinning it is a board swap plus new templates, not new machinery.

### 2. The ADP curve is not monotone, and the pooling is the finding

Windowed means over 2019-2024 make the 7th running back off the board worth more
than the 1st — an 11-point inversion against standard errors of 9 to 16. Isotonic
regression finds the closest non-increasing fit and pools the offending ranks
rather than inventing an ordering.

RB ranks 1-7 collapse to a single 172.5, and the resulting tier runs to RB10.
**Gibbs at ADP 1.5 and Barkley at 17.0 are the same asset** as far as six seasons
can tell. Tiering is an empirical result here, not a presentation choice: the
spread around the curve is more than five times the step between ranks.

### 3. Keeper adjustment matters at exactly one position

Replacement in a keeper league is league demand *less what is kept*, over the
players still available:

    QB   122.3 -> 169.5   (+47.2)
    TE    82.6 ->  85.2   (+2.6)
    RB   114.3 -> 114.3     0.0
    WR   112.2 -> 112.2     0.0

13 of the 18 declared keepers are QBs. `board.compare_baselines` runs it both
ways so the correction stays checkable — if it ever reads near zero, the
complexity is not earning its keep.

### 4. Keepers being off the board shifts everyone else ~15 picks earlier

All 18 keepers have ADP inside the top 150. They are off the board *and* consume
picks, so the rest move up. Barkley 26.3 → 13.3, Hampton 36.4 → 22.4, McBride
65.3 → 49.3. **Any mock that does not model keepers is showing players ~15 picks
late.** This adjustment is currently computed ad hoc in conversation and is
**not yet a function** — see §Next, item 3.

### 5. Team environment routinely outweighs the board's own edges

Measured over 128 team-seasons: team skill-position fantasy points move **45.3
per point of mean implied team total**. Best-to-worst offence is worth 34-69
points to a player depending on his share.

`board.context_flags` names pairs where a board edge is smaller than the
environment gap. The sharpest on the live board: **Trey McBride over Colston
Loveland is a 2.4-point edge against a 42.8-point environment gap**, because
Arizona is the lowest-priced offence in the NFL (17.42 implied, −1.97 SD).

This is not double counting — the expected-points curve is *rank*-based and
therefore team-blind. But it is an upper bound, since some discount is already
in the ADP level and `env_swing` cannot know how much.

### 6. RB promotion criteria (carried forward, unchanged)

Among backups whose role then grew: snap share 0.47, red-zone carry share 0.34,
TD-equity share 0.32 — while **prior efficiency reads as noise** (YPC −0.15,
yards after contact 0.00). At WR/TE the efficiency logic holds (TE yprr 0.35).
Quality terciles are a filter not a picker: 4.1% / 10.2% / 18.4%.

The decay-rate thread is **closed**: promoted receiving-profile RBs hit 13.3% vs
3.2% for rushing-profile — the opposite of the hypothesis, with overlapping
intervals on cells of ~30. Measured, reported, not a feature.

---

## The league, as the code now sees it

Pulled live. Display name `zaCHattack15`, but **the shell exports do not reach
the Bash tool's environment** — pass ids directly in scripts, or prefix the
command (`FF_EDGE_LEAGUE_ID=... uv run ...`).

- **Shiva Bowl**, 10 teams, 0.5 PPR, `QB/RB/RB/WR/WR/TE/FLEX/SUPER_FLEX/K/DEF`
  + 5 BN, 15 rounds, 2 keepers, reverse-standings waivers.
- **Keepers cost the drafted round, minus one the first year kept, minus two the
  second. Waiver pickups cost ADP round plus one. Two-year maximum.**
- His keepers: **Jayden Daniels (R6) and Trevor Lawrence (R9)** — both QB slots
  filled. **Daniels is in year two, so 2026 is his last.**
- Draft slot 4. **17 picks owned, 15 usable**, including five in an eight-pick
  window: `R5#44, R5#47, R5#48, R5#50, R6#51`. No R10, no R13.
- One team (`bordum`) has not declared; the board reports `undeclared_teams`
  rather than assuming the pool is final.

The structural edge: 6 teams need 7 QB-capable starters from ~3 startable
quarterbacks. He needs zero.

---

## Where to pick up — four threads, in the user's own priority order

### 1. Dashboard refactor — spec first, and expect a long interview

**The user's words: too many metrics he is not following, or that were proved
useless. Some graphics are solid and he wants more of those; most is not where
the project ended up.**

This is the big one and it is a **spec exercise before it is a build exercise**.
Do not start editing `app.py` (1800+ lines, 6 tabs). Interview first. Questions
worth asking, at minimum:

- Which existing charts does he actually use? (The stability chart and the
  landscape PAR-over-time charts were praised; most tables were not.)
- What decisions should the dashboard support — draft day, weekly start/sit,
  trades, waivers? Those are four different products.
- What survives from `breakout.py` / `projection.py`? Both are honest negative
  results. A negative result deserves a paragraph, not a tab.
- Does the k-means removal note still need its own callout?
- Should the Screen and Board tabs merge into one draft-day view?
- Who else looks at this? He mentioned "myself and others" — that changes the
  explanation burden a lot.

Write the spec to a file and get agreement before touching the app.

### 2. Contextual scoring — the user wants to learn it, so build it slowly

He wants to **trial on an easy-to-ingest free source (Reddit or X/Twitter API),
on a subset — WR, 5-10 players to start.** Also a spec exercise.

What already exists to build on: `claims.py` has the whole scoring spine (tier ×
specificity × novelty × recency), the verbatim-quote guard, resolution against
`weekly_trust`, and per-source grading. `news.py` has RSS. The ledger currently
runs on Google News RSS plus depth charts.

Honest notes for that spec:
- **Reddit's API is the friendlier of the two** — free tier, no cost, decent
  rate limits, and `r/fantasyfootball` game threads are dense with role talk.
  X/Twitter's free tier is severely limited and effectively read-only-tiny.
- The volume/value inversion he identified in the original scoping still holds:
  Reddit will be *high* volume and *low* value per item. The novelty dedupe and
  tier weighting exist for exactly this, but expect to retune both.
- A 5-10 player WR subset is a good call because it makes the **resolution**
  step tractable — that is what turns the ledger into labeled data, and it is
  the part that cannot be shortcut.
- Season one is still labeled-data collection. Do not promise predictive
  validation before there is a corpus.

### 3. Move the conversational findings into the dashboard

**The user explicitly wants the conclusions reachable without an LLM in the
loop.** Concretely, these exist only as ad-hoc analysis in this session's
transcript and should become functions plus views:

- **Keeper-adjusted ADP** (§4 above) — currently computed inline. Should be a
  function in `board.py` and a column on the board. Highest value, lowest
  effort; do this one first.
- **Availability at each of your picks** — `board.targets` exists but is not
  surfaced anywhere in the app.
- **Environment vs board edge** — `board.context_flags` exists, not surfaced.
- **Quality/opportunity percentile lookup for a target list** — was done ad hoc
  against `archetypes.scores`; deserves a proper "compare my shortlist" view.
- **Pick inventory** — `board.picks` exists, not surfaced.

Most of this is *surfacing what was just built*, not new analysis.

### 4. ML / deep learning — brainstorm before building

He wants something with real impact. **Be careful here**: the project has
measured, repeatedly, that fitted models do not beat ADP, and that the binding
constraint is label seasons (2.5-6.5 events per variable), not algorithms. A
neural net on the same 831 labeled player-seasons will do worse, not better, and
that is a prediction the project's own evidence supports.

Where the data volume actually is, and therefore where learning has a chance:

- **Week-level rather than season-level.** `weekly_stats` and the play-level
  `ff_opportunity` tables are orders of magnitude larger than the season-level
  label set. A start/sit or weekly-usage model has a real sample.
- **Play-level opportunity models.** Expected points per play given down,
  distance, field position, personnel — nflverse has every play.
- **Claim extraction / NLP.** Genuinely a language problem, genuinely has
  volume once the ledger accumulates, and it is the piece he wants to learn.
- **Sequence models over weekly usage.** Role changes are a change-point
  detection problem, which is a real ML framing rather than a bolt-on.

The honest framing for that conversation: **pick the problem where more data
exists than the season-level board has**, and where the answer is not already
priced by ADP. Anything else re-runs a measured null with more parameters.

---

## Things to be careful about

**`config.ROOT` uses `parents[1]`, not `parent`.** `src/config.py`. Using
`.parent` silently repoints `DATA_DIR` at `src/data` and orphans the cache.

**`data/` is gitignored and must stay that way.** The repo is public. League ids
and Sleeper handles come from the shell for that reason — a league id exposes
nine other people's data.

**Shell exports do not reach the Bash tool.** Prefix commands instead.

**Never push without asking.** `Projects/CLAUDE.md` requires explicit approval in
the current conversation.

**Season-forward validation only.** Never a random split.

**Traps found and fixed — do not reintroduce:**
- `spread_line` in nflverse is **positive when the home team is favoured**,
  opposite to betting convention. Read `spread` from `expected.team_environment`,
  never `spread_line` raw.
- FFC says `LAR`, nflverse says `LA`. Both sides of any team join must go
  through `ids.normalize_team` — the failure is a silent null, not an error.
- A traded pick's `roster_id` is whose pick it *originally* was, not who holds
  it. Reading it the other way puts picks in entirely the wrong rounds.
- The teams table carries relocated franchises (36 rows, not 32). Key team
  sheets off the season schedule.
- `rank(descending=True)` gives rank 1 to the *largest* value.
- Both `ff_opportunity` pbp tables carry weeks 19-22 with no `season_type`.
- The rush table has `rushing_td_exp` *and* `rush_touchdown_exp`; they disagree
  on 4,558 rows and the short-named one is a sentinel.

---

## Where to pick up

```bash
uv run pytest                                     # 190 tests, ~10s
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py # Landscape/Players/Screen/Strategy/Board/Glossary
FF_EDGE_LEAGUE_ID=... uv run python -c "from src import board; print(board.build()['players'].head(20))"
```

The claims ledger fills going forward only — `uv run python -m src.bootstrap
--light` daily during camp is what accumulates it, and news extraction needs
`ANTHROPIC_API_KEY` in the shell (depth-chart claims work without it).

Win totals: `data/win_totals_2026.csv` is written and blank. Filling it is the
one manual input that would improve the preseason environment layer, and it is
the best available proxy for the "will this team still be trying in December"
question that game lines cannot answer preseason.
