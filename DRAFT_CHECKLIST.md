# Draft checklist — updated 2026-08-17

**Shiva Bowl drafts 2026-08-22 at 19:00. Five days.**
828 Omegle Chat drafts 09-06. The Jungle 08-30 is out of scope by decision —
dynasty startup, and nothing here estimates an age curve.

Ordered by **deadline first, then value.** Block A becomes impossible or
worthless after the 22nd; Block D can wait until after all three drafts.

Verified against the code, not against the docs. Where a doc disagreed with the
repo, the repo won and the doc was corrected.

---

## ~~Block A1/A2 — the merges~~ **DONE 2026-08-17**

All nine branches merged into `main` and deleted; all four worktrees removed.
`main` is now the only branch. **385 tests pass, 11 skip** — up from 361/8,
because the merges brought 24 tests with them.

**What the merge actually fixed, and it was not cosmetic.** The stranded
`promote-cost-of-waiting` commits carried `board.roster_need`, which closes the
gap between **scarcity** and **need** — a distinction the board had never made:

> `roster_demand` prices what the *league* is short of, which on the 2026 board
> leaves quarterback at 7 slots against 20 and therefore screaming scarce. That
> is true and **it is not your problem.** Thirteen of those twenty are kept and
> **two of them are yours** — Jayden Daniels and Trevor Lawrence, against a
> roster carrying exactly two QB-capable slots. **Your quarterback need is
> zero.**

The failure mode is worth keeping in mind because it inverts: *league scarcity
is maximally misleading to the manager who caused it.* The teams that make
quarterback scarce by keeping one are precisely the teams that must not draft
one — so a board reporting only league demand tells every keeper-holder to buy
the thing they already own. **It was telling you that at pick 4.**

Also landed: the recommendation panel now sits above the board rather than
underneath its own evidence, and a new `CLAUDE.md` trap about Streamlit misuse
warnings being invisible to all four of the obvious asserts.

---

> **State as of 2026-08-17, end of day.** Every cheap item is done. **437 tests
> pass, 2 skip.** What remains before the 22nd is A3, A4, A4b, A4c and A5 —
> **none of them are code**, they are you reading the board and the machine
> staying fed. Blocks C, D and E3/E4 are all post-draft by design.

## Block A — still before 08-22

### A0. Export `FF_EDGE_LEAGUE_ID` on the 22nd. **Found 2026-08-18. Read this first.**

**An unset league id does not give you "no league" — it gives you The Jungle.**
`scoring.resolve_league_id` falls through to `sleeper.my_leagues()` and takes the
**first row**, which on this account is the dynasty startup that is out of scope
by decision. The board then builds, prices, ranks and renders normally against
the wrong league's keepers and rosters.

It surfaced on the new website as a warning that read *"the league resolved but
no keepers are declared on it yet"* — true of The Jungle, false of the league
being drafted, and the only visible symptom. With the id set, the same board
resolves **20 keepers, 162 draftable, 60 inside roster demand** and the warning
disappears.

**The Streamlit app cannot tell you this is happening.** Its sidebar says "Live
from Sleeper" without saying which league. The website names the league on every
page and flags a guessed one; `app.py` was deliberately not touched, so on the
22nd the protection is you, exporting the variable:

```bash
FF_EDGE_LEAGUE_ID=<shiva bowl id> FF_EDGE_SLEEPER_USER=<handle> uv run streamlit run app.py
```

Sanity check before drafting from any board: **keepers should read 20, not 0.**

### A3. Sit with the app yourself. **This is the item that expires.**

**Status: still never done.** Carried unfinished since 08-02.

The machine dry run proves Draft Day does not crash and answers in 0.3s. **It
cannot tell you whether the answer arrives in ten seconds when you are on the
clock.** No substitute for a human doing it.

```bash
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run streamlit run app.py
```

You are now reading a materially different board than you were two days ago —
the roster-demand cut, the per-position blend centering, `roster_need`, and the
relocated recommendation panel all landed since. **Budget real time for this
one; the thing you last looked at is not the thing that will be on screen.**

Three questions to answer while you are in there:

1. On the clock at your real pick, how many seconds to a decision?
2. Does the `block` ordering read as sensible, or does it still surprise you the
   way Henry-above-Cook did?
3. Is anything on screen you never look at? Cut it before the 22nd, not after.

### A4. Export the board and mark it up by hand

The Big Board has a CSV download for exactly this. **A preliminary board you
have argued with is worth more than a final one you have only read.**

### A4b. Build per-block decision artifacts for the draft table

**Requested 2026-08-17.** The block-4 running-back breakdown done that day is
the template: one artifact per block you expect to be choosing inside, published
and open beside you on the clock. **Do this for the blocks around your actual
picks, not for the whole board** — the value is having the argument already made
where a 90-second decision is coming.

The method, because the answer is only as good as this and it is repeatable:

1. **Rank the block on the signals that repeat, not the ones the board sorted
   on.** `rank_board` breaks a within-block tie on `quality_pct`, and
   `stability.year_over_year` measures that as the *weak* half — at RB,
   opportunity runs `target_share` 0.65 / `carry_per_game` 0.588 / `rush_share`
   0.57 against quality's `tprr` 0.402 / `yprr` 0.361 / `ypc` 0.278. The block
   itself is measured; **the order inside it is not**, and `BIG_BOARD_SPEC.md`
   §12 says so. That gap is the whole reason these artifacts are worth making.
2. **Check regression debt** with `peek.regression_candidates`. Large positive
   `pts_over_exp` is efficiency that has to be given back — it repeats at
   r_yoy 0.283. Large negative is the buy-low.
3. **Read `rz_carry_share` / red-zone share**, not just target share. TD equity
   is what `src/context.py` calls "the largest thing target share cannot see",
   and it is computed but never displayed (Block E2).
4. **Check `env_swing`** for the offence, and **`ecr` / `ecr_sd`** for where the
   crowd sits and how settled it is.
5. **Run `archetypes.neighbors(..., restrict_to=<the block>)`.** The block is a
   claim about *value* and is silent on *type*; the comparables split it. An
   outlier whose nearest neighbour is far away is a different bet at the same
   price.
6. **Say the age out loud.** Nothing in `par_env` discounts it — see A4c.

Worked example from 08-17, block 4: the board ranked Achane first in the block
on a `quality_pct` of 100, while he carried **+56.1 points over expected** and
the worst environment of the seven; Jeanty ranked 15th while leading the block
in both signals that actually repeat. **The artifact disagreed with the board
and the board's own docstrings explain why it should.**

### A4c. Know that age is not priced anywhere on this board

**Not a build item — a thing to hold in your head on the 22nd.** Zach, reading
the board: *"very surprised Derrick Henry is that high, might just be his age is
not a built-in context risk marker."* Correct. `age` is computed in
`features.py`, carried into `valuation.py` and defined in the glossary, and it
is **an input to no ranking**. Nothing in `par` or `par_env` discounts a
31-year-old back.

On the 2026 board that silently inflates Henry (block 2), and Barkley at 28.6
on 17.5 carries a game and Jacobs at 27.6 (both block 4). A redraft age
coefficient is measurement **M4** in `RESEARCH_SPEC.md` §2.3 — cheap, in scope,
and unbuilt. Until it exists, apply the discount yourself.

### A5. Run bootstrap by hand on the morning of 08-22

The launchd agent fires at 08:00 and 17:00. The 17:00 run lands two hours before
a 19:00 draft and **ADP moves all day** on draft day.

```bash
FF_EDGE_SLEEPER_USER=... uv run python -m src.bootstrap --light
```

---

## ~~Block B — surface the unreachable tools~~ **DONE 2026-08-17**

`peek` and `adp` are now imported by `app.py` and there is a **Screens**
expander on Research, first and open by default. **425 tests pass, 2 skip.**

| item | shipped |
|---|---|
| **B1** `peek.regression_candidates`, `market_disagreement`, `snap_trend` | yes — with the caption saying a screen is a question generator, not a ranking input |
| **B2** `adp.movement` | yes — **read in the profile's market**, see below |
| **B3** Footballers `ffb_spread` + `stalest_days` on the big board | yes. The disagreement panel (`compare_footballers`) is still unbuilt |

**Two silent defects turned up in the wiring, and both are the reason this was
worth doing rather than just displaying something.**

*The season was a literal.* All four `peek` entry points defaulted to
`season=2025` while `config.SEASON` said 2026 — right today, wrong the first
August nobody re-reads them, and invisible either way because a screen against
the wrong year returns a full plausible table. Now `LAST_PLAYED = SEASON - 1`,
and the test asserts against that rather than a number so it cannot rot back
into the literal it forbids.

*The market was the wrong league's.* `adp.movement` defaults to
`config.ADP_SCORING`/`ADP_TEAMS` — **ppr/12, which prices 828, not the Shiva
Bowl's 2qb/10**. Rendered bare it would have shown a well-formed table of
another league's drift. The market is now taken from the board's own profile
rather than resolved a second time, so the screen cannot disagree with the board
about which market it is in.

Also fixed: `season_snap_pct` and `last4_snap_pct` both rendered as "Snap
share", which turns the one comparison the snap screen exists for into a guess
about which column moved. Only visible in rendered output, not in the code.

### ~~B3 leftover — the Footballers disagreement panel~~ **DONE 2026-08-17**

On Research, as `compare_footballers` sorted by `|rank_shift|`. **The spec's
`board_rank <= 60` cut turned out to be load-bearing rather than cosmetic** —
unfiltered, every largest disagreement is a backup quarterback or a receiver
priced past ADP 110, which is a real disagreement about players nobody in this
league drafts. The cut is a slider so it can be widened deliberately.

Held to `compare_baselines`' standard, and the panel says so on screen: if these
shifts ever read near zero, `FOOTBALLERS_WEIGHT` goes to 0.0 and the blend comes
out. `FOOTBALLERS_SPEC.md`'s proposed surface is now fully built.

## Block C — before 09-06, not before 08-22

### C1. The profile selector

`profiles` still not imported. Roster shape is not selectable in the UI, which
means the one input that moves replacement level cannot be changed.

**828 drafts 09-06 and cannot be priced without it.** It is `standard_12` with
`rec: 1.0`, and FFC `ppr`/12 ADP is already cached for all eight seasons.
`profiles.py` is genuinely finished — it has its own test file — so this is
wiring, not design.

Deliberately not in Block A: it touches the sidebar, which every tab reads.
Wrong shape of risk five days out, and no 08-22 deadline on it.

---

## Block E — research readiness *(new — this is what "detailed research" needs)*

Depth is fine. **Assembly is the problem.** Nothing here is draft-week work.

### E1. There is no player view — the structural gap

To research one player you use **five separate selectboxes across four tabs**,
each with its own session-state key: position/player in Players, `comp_player`
in comparables, one in Board, `who_claims` in claims, `big_board_anchor` on the
Big Board. The one that looks like a player detail is really a comparables view
— neighbours and position-wide tables, not one player's full picture.

`RESEARCH_SPEC.md` §6 flagged this ("answering *what do I think about this guy*
means visiting four tabs and joining by eye") and it is **still exactly true.**
For detailed research this costs you more than anything else in this file.

### ~~E2. `context.py`'s metrics never reached the screen~~ **DONE 2026-08-17**

`board.attach_usage` puts last season's opportunity profile on the big board
behind a **"Show last season's usage"** toggle — `age`, `snap_pct`,
`target_share`, `exp_td_share`, `neutral_target_share`, `rz_target_share`,
`rz_carry_share`.

**The selection rule is the design.** Every column is opportunity, because
`stability.year_over_year` measured that axis at **0.52-0.65** year over year at
RB against a quality axis topping out at **0.402** — and `quality_pct` was
already both a display column *and* the within-block sort key while the sturdier
half was invisible. `exp_td_share` covers all four positions at ~100% and is the
thing `context.py` names as invisible to target share.

**Attached after `rank_board`, and that ordering is the point.** Appending
downstream is what makes "usage cannot move a rank" a fact about the call graph
rather than a docstring claim;
`test_attaching_usage_never_reorders_or_reranks_the_board` fails if anyone moves
the call upstream.

The red-zone and neutral pairs are position-lopsided and stay that way — a
receiver has no `rz_carry_share` and filling it with a zero would read as "no
red-zone role". **Blank means not measured, never zero.**

`age` rides along for a different reason — see A4c. It is still an input to no
ranking; displaying it is the mitigation, not the fix. **M4 remains unbuilt.**

### E3. The measurements — M1 first

Nothing estimates **per-player outcome spread**. The argument is the sharpest
thing in the docs: *ADP is an efficient estimate of the mean and a non-estimate
of the variance*, and the spread around the curve is worth roughly **five times**
more than getting the tier order right. It inverts advice by round — you want
low variance early and high variance late — and the market has no mechanism to
say which is which even when it knows.

M2 (availability / games played) and M4 (age, controlling for price) are also
unbuilt. **Any of them may come back null, and a null ships as a null.**

### E4. Seven years of expert dispersion, sitting unused in the cache

From `FANTASYPROS_IDEAS.md`: `nflverse.ff_rankings("all")` is an archive, not a
snapshot — **1.8M rows across 359 scrape dates back to 2019**, carrying
`ecr_type = rsf` (redraft superflex, this league's actual format).

This is the only asset found in either audit that could put a **measured** weight
behind the blend instead of a judgment call. `FOOTBALLERS_WEIGHT = 0.5` and
`ENV_WEIGHT = 0.35` are currently the only asserted numbers in `config.py`.

---

## Block D — after all three drafts

- **`app.py` is 3,477 lines** and grew ~190 in the merge. Big Board is the
  natural first lift-out; it has no dependants.
- **Split `board.py`** (now ~1,900 lines). ~600 lines are the format-general
  ranking spine. **Split it, do not delete it.**
- **Weekly Fantasy Footballers rankings.** Re-probe in September; both plausible
  URLs 404'd on 08-10.
- **Cross-market price disagreement**, building on `adp.multi_format`.
- **The claims ledger.** Every row resolves *pending* because the 2026 games do
  not exist yet. Correct and documented, not a bug. Judged next August.

---

## Housekeeping

- **`probe_fp_news.py` is at repo root, written but apparently never run.** Its
  own docstring says *"Delete this file once the answers are recorded"* — and
  the answers are **not** in `FANTASYPROS_IDEAS.md`. It costs 5 calls of a
  50/day budget and refuses to run without `--confirm`. Either run it and record
  the answers, or delete it; leaving it is the one state its author ruled out.
- ~~**Two remote branches survive**~~ **Resolved by 2026-08-18.** Neither
  `origin/measure-what-repeats` nor `origin/worktree-fantasypros-brainstorm`
  exists any more, and `origin/main` is level with `main` — `git branch -r`
  shows only `origin/main`. Both branches were fully merged locally before they
  went, so nothing was lost. The push gate is unchanged: `CLAUDE.md` still
  requires explicit approval per push.
- **The daily bootstrap is not running clean** — recent runs logged 56-59 of 60.
  Causes are transient network on nflverse historical tables and one Sleeper read
  timeout. **None touch the ADP, Footballers or claims snapshots**, the three
  things that cannot be backfilled. Low severity; the log should still be clean
  enough that a real failure stands out.
- `HOW_IT_WORKS.md` was merged from `docs-orientation` and describes the **old
  six-tab** app. Given a status block and a where-did-the-tabs-go mapping rather
  than a rewrite — its pipeline half is still accurate and is why it was kept.

---

*Updated 2026-08-17 after merging and deleting every branch. Audit re-run
against the merged code; Blocks B, C and E were all re-verified as still open.*
