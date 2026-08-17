# Draft checklist — 2026-08-16

**Shiva Bowl drafts 2026-08-22 at 19:00. Six days.**
828 Omegle Chat drafts 09-06 (21 days). The Jungle 08-30 is out of scope by
decision — dynasty startup, and nothing here estimates an age curve.

This file replaces the "where to pick up" section that `HANDOFF.md` used to
carry. It is ordered by **deadline first, then value** — everything in Block A
becomes impossible or worthless after the 22nd, and everything in Block D can
wait until after all three drafts.

Audit method, so a future session knows what was actually checked rather than
read: every claim below was verified against the code or git, not against the
docs. Where a doc disagreed with the repo, the repo won and the doc was
corrected.

---

## Block A — before 08-22. Do these in order.

> **Merge order, verified 2026-08-16.** Three branches need to land and all
> three are safe. A1 and A2 are clean fast-forwards from `origin/main` with zero
> divergence. The docs branch from this audit merges cleanly against A2 —
> test-merged, no conflicts — but is a regular merge rather than a
> fast-forward once A2 is in, since both branch off `origin/main`.
>
> ```bash
> cd /Users/zacharyserapin/basecamp/Projects/ff-edge
> git merge --ff-only origin/main                        # A1
> git merge --ff-only worktree-promote-cost-of-waiting   # A2
> git merge worktree-doc-cleanup-preflight               # the docs
> uv run pytest
> ```
>
> **Nothing here is pushed.** `CLAUDE.md` forbids pushing without approval in
> the conversation, so the remote is untouched and `worktree-promote-cost-of-waiting`
> in particular exists **only on this machine** — it has no upstream and no
> remote branch contains it.

### A1. Fast-forward local `main`. Nothing else is safe until this is done.

**Status: blocking. Two minutes.**

Local `main` sits at `78cfc3c`, three commits behind `origin/main` at `31c2445`.
Strictly behind, zero divergence, clean working tree — a fast-forward with no
possible conflict. The three commits are not cosmetic; they are the last three
changes to how the board ranks:

| commit | what it changes |
|---|---|
| `024f8a7` | Centers the Footballers blend **per position**, so three analysts stop setting a whole position's level |
| `94f9e6f` | `board.roster_demand` — cuts the board at the replacement line. Fixes tight ends being promoted a median **+47.5 places** over ADP |
| `31c2445` | `rank_board` imputes a missing `quality_pct` to the block median rather than sinking it last |

**The board you open right now is the one that ranks all 18 tight ends inside
the top 100 of a league that rosters about 13.** That is the single largest
distortion the project has measured, and the fix is already written and pushed.

```bash
git -C /Users/zacharyserapin/basecamp/Projects/ff-edge merge --ff-only origin/main
uv run pytest          # expect 361 pass / 8 skip, or 367 / 2 with both env vars
```

### A2. Merge the two stranded commits in the `promote-cost-of-waiting` worktree

**Status: blocking-ish. Draft-critical work that exists nowhere else.**

`.claude/worktrees/promote-cost-of-waiting` holds two commits that are on **no**
branch that has been merged and are **not pushed anywhere**:

| commit | what |
|---|---|
| `8e8727e` | Put the recommendation above the board, not underneath its own evidence — `app.py` +136, `tests/test_big_board.py` +73 |
| `43c8cd4` | Tell him what he needs, not what the league is short of — `app.py` +78, `src/board.py` +110, `src/glossary.py` +18, `tests/test_big_board.py` +153 |

They sit directly on top of `origin/main`, so after A1 this is a fast-forward
too. Both ship with tests. This is roughly 575 lines of tested, draft-day
surface — the recommendation panel and roster-need logic — that would otherwise
be lost when the worktree is cleaned up.

```bash
git -C /Users/zacharyserapin/basecamp/Projects/ff-edge merge --ff-only worktree-promote-cost-of-waiting
uv run pytest
```

**Verify in the main tree, not the worktree.** `data/` is gitignored so a
worktree has no cache, and 14 of 16 test files read it.

### A3. Sit with the app yourself. This is the item that expires.

**Status: never done. Cannot be done after the draft.**

Carried unfinished through every handoff since 08-02. The machine dry run proves
Draft Day does not crash and answers in 0.3s. **It cannot tell you whether the
answer arrives in ten seconds when you are on the clock.** That is a human test
and there is no substitute for it.

```bash
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run streamlit run app.py
```

Do this **after A1 and A2**, so you are reading the board you will actually
draft from. Walk the pick selector forward through a realistic sequence and
answer three questions:

1. On the clock at your real pick, how many seconds to a decision?
2. Does the Big Board's `block` ordering read as sensible, or does it still
   surprise you the way Henry-above-Cook did?
3. Is anything on screen you never look at? Cut it before the 22nd, not after.

### A4. Export the board and mark it up by hand

The Big Board has a CSV download for exactly this. "Review a lot of the
analysis, add some, drop some" was the original ask and it is not a thing you do
in a Streamlit table. **A preliminary board you have argued with is worth more
than a final one you have only read.**

### A5. Run bootstrap by hand on the morning of 08-22

The launchd agent fires at 08:00 and 17:00. The 17:00 run lands two hours before
a 19:00 draft and **ADP moves all day** on draft day.

```bash
FF_EDGE_SLEEPER_USER=... uv run python -m src.bootstrap --light
```

---

## Block B — cheap, additive, zero risk to Draft Day. Do if A finishes early.

Each is genuinely additive — a new expander on an existing tab. None touches the
ranking path, so none can break the board six days out.

### B1. Surface the `peek` screens

**`app.py` imports 16 modules and `peek` is not one of them.** Verified — zero
references in the file. So three finished tools have never been reachable:

- `peek.regression_candidates` — points over expected, the canonical buy-low
  screen. This repo has its own evidence for it: `stability.year_over_year`
  measured that opportunity persists better than efficiency at every position.
  The screen and its justification were built separately and never introduced.
- `peek.market_disagreement` — where expert consensus is least settled
- `peek.snap_trend` — role change before the box score

This is the same failure mode as `board.py` sitting unreachable two weeks before
a draft. Cheapest win available.

**Ship the caveat with it:** a screen is a question generator, not a ranking
input. Promoting `pts_over_exp` to a ranking requires a measurement (M3) that
has not been made and may come back null like the last two.

### B2. Surface `adp.movement` — it returns real rows now

`adp` is also not imported. When this was last considered on 08-13 the history
held two snapshots per market and would have rendered an empty table, which
`CLAUDE.md` warns reads as a network blip rather than as "no data."

**That has changed.** The daily agent has been running since 08-13 and all three
`adp_history_*` files were written this morning. August ADP drift is camp news
made numeric, and the fortnight before a draft is the highest-information
stretch of the year — this is the window where the column is worth the most it
will ever be worth.

### B3. The Footballers staleness caption

`FOOTBALLERS_SPEC.md` §4 marks this **"required, not optional"** and it is the
one display element in that spec that is not cosmetic. It is not built —
`stalest_days` appears nowhere in `app.py`.

The panel is not evenly fresh. On 2026-08-10 the median projection was 7 days
old for Jason, 64 for Andy and **90 for Mike**. `ffb_par` now carries weight
0.5 on the board, so a May opinion is currently being presented as a current one
at half the board's say.

Also unbuilt from that spec: the disagreement panel (§2) and `ffb_spread` as an
uncertainty read (§3). `ffb_spread` is the thing no other source in this project
can do — two players at the same ADP with panel spreads of 12 and 68 points are
different bets at identical price.

---

## Block C — before 09-06, not before 08-22

### C1. The profile selector

**`profiles` is not imported into `app.py` either.** Verified — zero references.
The sidebar reads one live league and exposes teams, points-per-reception and
passing-TD only. **Roster shape is not selectable**, which means the one input
that moves replacement level cannot be changed in the UI.

This has a concrete customer now: **828 Omegle Chat drafts 09-06 and cannot be
priced without it.** It is `standard_12` with `rec: 1.0`, and FFC `ppr`/12 ADP is
already cached for all eight seasons, so both the board and the historical curve
come from a matching market. The work is wiring, not design — `profiles.py`
already binds a roster format to the ADP market that prices it.

Deliberately **not** in Block A: it touches the sidebar, which every tab reads.
That is the wrong shape of risk six days out and there is no 08-22 deadline on
it.

---

## Block D — after all three drafts

Nothing here changes a pick this season.

- **`app.py` is 3,286 lines** and growing — violates the `Projects/CLAUDE.md`
  standard. The Big Board tab is the natural first lift-out; it has no
  dependants.
- **Split `board.py`** (1,792 lines). ~600 lines are the format-general ranking
  spine and ~500 are keeper/pick plumbing. **Split it, do not delete it** —
  deleting wholesale takes the good half.
- **The four measurements, M1-M4.** M1 (outcome spread) is the highest-value
  open thread in the project and has been called that for three sessions. Note
  three of the four may well come back null, and **a null ships as a null.**
- **Weekly Fantasy Footballers rankings.** Re-probe in September; both plausible
  URLs 404'd on 08-10. Weekly answers "who do I start", which wants a per-week
  replacement rather than a season one.
- **Cross-market price disagreement**, building on `adp.multi_format`.
- **The claims ledger.** 150+ rows, every one resolving *pending* because the
  2026 games do not exist yet. That is correct and documented behaviour, not a
  bug. It starts producing labels once the season begins and gets judged next
  August.

---

## Housekeeping — noticed during the audit, none of it urgent

- **The daily bootstrap is not running clean.** The last six runs logged 58/58,
  57/58, 60/60, 56/60, 58/60 and **59/60**. `CLAUDE.md` says a run that logs
  anything other than `N/N ok` is a real failure. The observed causes are
  transient network on nflverse historical tables (`participation`,
  `pfr_advstats rec`) and one Sleeper read timeout — **none of them touch the
  ADP, Footballers or claims snapshots**, which are the three things that cannot
  be backfilled, and all three were written this morning. Low severity, but the
  log should be clean enough that a real failure stands out.
- **`BIG_BOARD_SPEC.md` §11 sits after §15.** Numbering ran 9, 10, 12, 13, 14,
  15, 11. Fixed in this pass.
- **The three newest commits documented themselves in `CLAUDE.md` but not in
  `BIG_BOARD_SPEC.md`**, which otherwise records every board change as a
  numbered section. Added as §16-18 in this pass.
- **Six stale branches.** `dropoff-and-landscape`, `measure-what-repeats`,
  `one-board-and-environment`, `rank-by-what-the-curve-can-resolve` and
  `weight-the-distance` are all fully merged into `origin/main` and safe to
  delete. `docs-orientation` is one commit ahead and holds `HOW_IT_WORKS.md`,
  a 254-line orientation doc written against the **old** architecture — it
  describes a Screen tab that no longer exists and a four-tab layout. Either
  rewrite it against the current five tabs or drop the branch; do not merge it
  as-is.
- **`worktree-fantasypros-brainstorm`** holds `FANTASYPROS_IDEAS.md`. Its lead
  finding already graduated into the board (ECR is live in `board.py`,
  `config.py`, `glossary.py`). What has **not** been used is the rest of it:
  `nflverse.ff_rankings("all")` is an archive, not a snapshot — 1.8M rows across
  359 scrape dates back to 2019, carrying `ecr_type = rsf` (redraft superflex,
  this league's actual format). **That is seven years of expert dispersion
  sitting in the cache**, and it is the only asset found in this audit that
  could put a measured weight behind the blend instead of a judgment call. Block
  D, but do not lose it.

---

*Written 2026-08-16 from an audit of the repo rather than of the docs. Where a
doc disagreed with the code, the code won.*
