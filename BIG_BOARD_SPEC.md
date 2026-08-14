# big board spec — one ranked list, before the draft

**Written 2026-08-13. Status: built the same day — see §9 for what shipped and
what the build changed. §1-7 are the spec as agreed, left as written.**
Pulls `RESEARCH_SPEC.md` §5.2 forward from Phase 1 into Phase 0, because it is
what Zach actually asked for:

> mainly the need is to get a preliminary big board of my rankings based on all
> the work we have done. From there I can review a lot of the analysis we have
> done, add some, drop some so I can figure out my targets and draft strategies.

Three drafts inside 24 days — Shiva Bowl 08-22, The Jungle 08-30 (out of scope,
dynasty), 828 Omegle Chat 09-06. This serves the first and the third.

---

## 1. What exists, and why it isn't a big board yet

The app already computes everything. It computes it in **two frames that never
meet**, each holding two of the three columns the board needs.

| frame | app.py | carries | missing |
|---|---|---|---|
| `_draft_board()` | 2334 | `board_rank`, `par`, `tier`, `indist_n`, `adj_adp`, `env_swing`, `value_gap` | `vegas_gap` |
| `_valuation()` | 1610 | `quality_pct`, `market_pct`, `value_gap`, `vegas_gap`, `age` | `par`, `tier`, `adj_adp`, keepers |

So today, answering "who should I take" means reading two tabs and joining by
eye — which is exactly the thing that does not work on a clock.

## 2. The correction this spec makes to RESEARCH_SPEC §5.2

The spec says the app holds **three opinions** — `par`, `value_gap`, `vegas_gap`
— and proposes reconciling them. That framing is wrong in a way that matters for
any blend:

> **Only two of the three are independent of ADP.**

`par` is a function of price. `exp_points` maps a player's *positional ADP rank*
to what players at that rank have historically scored, so within a position PAR
reproduces the market's ordering by construction — `board.attach_quality`'s own
docstring says so. PAR's real content is **cross-positional**: what a QB1 is
worth against an RB1 at this format's replacement levels.

Consequences, both binding:

- **A composite that averages all three double-counts ADP.** Two of the inputs
  would be the market wearing different hats.
- The two genuinely independent reads are `value_gap` (per-opportunity quality
  this project measured, stability-weighted) and `vegas_gap` (a number a
  bookmaker will take money on). They disagree **for unrelated reasons**, which
  is the whole reason both are worth having.

So the board is: **PAR orders it, and the two independent opinions annotate
it.** Not a three-way average.

## 3. Design decision: side by side, no blend

`RESEARCH_SPEC.md` §9 Q4 asked whether the composite needs a stated weighting or
should ship showing inputs side by side. **Side by side, and here is the
argument rather than the preference:**

A weighting cannot be agreed to before the inputs have been seen disagreeing. A
blended number would hide precisely what Zach said he wants to do — "review a
lot of the analysis, add some, drop some." Averaging is how you stop being able
to see which input drove a rank.

It is also the honest option under this repo's own standard. A blend implies the
weights were measured. They have not been. None of M1–M4 is built, and the last
two fitted models here both came back null.

**Revisit after the drafts, with a measurement behind any weight.**

## 4. What ships

A new **first tab, "Big Board"**. Purely additive — it changes no existing tab,
so Draft Day carries zero risk nine days out. It is also the first brick of the
`RESEARCH_SPEC.md` §6 Rankings surface, so Phase 1 inherits it rather than
replacing it.

### Columns, left to right

| column | source | why it is on a draft board |
|---|---|---|
| `board_rank` | board | the default order — cross-positional slot value |
| `name`, `position`, `team`, `bye` | board | identity |
| `tier`, `indist_n` | board | **the targets column.** `indist_n > 1` means the curve cannot separate him from his group — take the cheaper one |
| `adp`, `adj_adp` | board | price, and keeper-adjusted price where they differ |
| `par` | board | slot value, printed at the precision the curve supports |
| `value_gap` | valuation | independent read #1: quality percentile − price percentile |
| `vegas_gap` | props | independent read #2: book's line percentile − price percentile |
| `signal` | derived | see below |
| `env_swing` | board | team environment |

### The `signal` column

Not a score. A four-way label over the two independent reads, reusing the
thresholds already live at `app.py:1834`:

- **`both up`** — `value_gap >= 15` and `vegas_gap >= 10`. Two unrelated methods
  say underpriced. The strongest thing this app can say.
- **`both down`** — `value_gap <= -15` and `vegas_gap <= -10`.
- **`split`** — they oppose, each past threshold. **This is the interesting
  one**, and it must not be averaged away: it is where a private read is worth
  having.
- **`quiet`** / null — no line posted, or nothing past threshold.

**Nulls are the common case and must read as "not measured", never "no edge".**
FanDuel prices ~92 players season-long, so `vegas_gap` is null for roughly two
thirds of the board by construction. Quality is null for thin seasons under the
volume floor. The UI says this in a caption, not a footnote.

### Export

`st.download_button` → CSV of the current filtered view. There is no export
anywhere in `app.py` today, and "review it, add some, drop some" is not a thing
you do in a Streamlit table. This is the part that makes it a *preliminary*
board rather than a final one.

## 5. The join, and the trap it walks into

Reuse `board.attach_quality`'s pattern exactly (`src/board.py:849`) — it is
already the solved version of this problem:

```python
keys = valued.select(
    ids.normalize("name").alias("_norm"), ...
).unique(subset=["_norm"], keep="first")
```

**One identity per player, not per row**, then a left join. That `.unique()` is
load-bearing: `ids.normalize` strips generational suffixes, so Michael Pittman
Jr. and Sr. collapse to one key, and a name-only join without the dedupe fans
rows out silently — the props join went 145 → 151 that way.

**Do not recompute `vegas_gap` on the board population.** `against_price`
requires both percentiles be taken over *the same set of players*; it is
computed over valuation's 143 and is only meaningful there. Carry the value
across, never re-derive it. (This is also why `against_price(board)` cannot
simply be called — the board frame has no `gsis_id`.)

Applicable traps from `CLAUDE.md`, all of which fail silently:

- `sort(descending=True)` puts **nulls first**. Every descending sort here needs
  `nulls_last=True`, or the top of the board is the players who could not be
  scored.
- `rank(descending=True)` gives rank 1 to the largest value. ADP rank and points
  rank run opposite ways.
- Both sides of any team comparison go through `ids.normalize_team`, including
  bare equality checks.

## 6. Tests

- **Row count is preserved across both joins.** The failure mode is fan-out, and
  it does not raise.
- **Null `vegas_gap` does not drop a player** from the board or change his rank.
- **`signal` never fires on a null input.**
- **Sorts put nulls last** — assert directly, per the trap above.
- **Drive the tab forward, repeatedly**, on the `tests/test_draft_day.py`
  pattern. That file exists because rendering once passed and rendering nine
  times segfaulted. Any new table with a control gets the same treatment; a
  vanished run (exit 139, no summary) is a red result.

## 7. Not in scope

- **No blended composite score.** §3.
- **No new analysis.** This assembles what is measured. M1–M4 stay Phase 2.
- **No dynasty profile.** The Jungle is out; `CLAUDE.md`'s redraft-only rule
  stands and a startup draft is the worst case for breaking it.
- **No restructure.** One added tab. `board.py`'s split is Phase 1.

## 8. Open — both resolved in the build

1. ~~Does `signal` need a fifth level for "book posted a line, quality is
   null"?~~ **Resolved: a missing input yields a null label, not `quiet`.**
   Folding "never measured" into "measured, nothing there" would report a board
   far better checked than it is, and with two thirds of it unpriced that is not
   a rounding error. `test_signal_stays_null_when_either_read_is_missing` pins
   it.
2. ~~Default sort?~~ **Resolved: `board_rank`, ADP one click away.** Rank is the
   app's opinion; ADP is the market you already have.

## 9. What shipped, and what it cost

Built 2026-08-13, same day as the spec. 303 tests pass with a league (up from
272), 298 pass and 7 skip without one.

| where | what |
|---|---|
| `src/board.py` | `attach_vegas` (carries the line across, never recomputes it) and `signal` |
| `src/glossary.py` | `board_rank`, `adj_adp`, `tier`, `vegas_gap`, `signal`, `line`, `line_pct`, `bye` — `vegas_gap` had been on the Board tab with no definition |
| `app.py` | `_tab_big_board`, first in the strip |
| `tests/test_big_board.py` | 9 unit + 2 driven-tab |

On the real board today: 159 draftable, 60 with a posted line, and of those —
5 **both up**, 6 **both down**, 7 **split**, 42 **quiet**. The splits are where
to spend reading time.

**Two things the build found that the spec did not anticipate.**

**The superflex edge is mostly already spent in this league.** 13 of the top
quarterbacks are keepers, so QB draft demand is 7 rather than 20 and replacement
lands at QB8 *among the available*. The first quarterback on the board is Brock
Purdy at **rank 19** (an earlier draft of this file said rank 30, which was the
first *split*, not the first QB).
`RESEARCH_SPEC.md` §0 is still right that the format is genuinely superflex —
that was confirmed live — but the actionable edge in the 08-22 draft is much
smaller than the format implies, because the other managers already took it.
The board handles this correctly and has been doing so; it is the *reading* of
§0 that needed the qualifier.

**An app-killing crash was live and dormant.** Building this surfaced
`ValueError: Season must be between 2006 and 2025`, thrown from
`promotion.weekly_trust(2026)` by way of the claims ledger, killing the entire
page. It had been unreachable until `bootstrap` pulled the ledger's first rows
earlier the same day. Fixed in `src/nflverse._played`; written up in
`CLAUDE.md`. **The relevant lesson for this spec is that the Big Board tab is
now the app's front door, so anything that raises inside it takes the front door
with it.**

## 10. The level/shape split — added 2026-08-13, second pass

Zach, reading the shipped board: *"A lot of RBs have higher PAR than the WRs
however it is a little contradictory as I do need to start 2 WRs and will likely
flex another… I'd rather have a top WR like Chase, Puka instead of CMC or Taylor
after the top 2 RBs go."*

**Measured, and he is right by a wide margin.** The PAR curve on the live board:

```
RB   72.6  72.6  72.6  72.6  72.6  72.6  70.5  67.4  60.5  53.9
WR   58.2  52.2  46.4  45.2  39.7  31.8  21.4  21.4  20.4  16.1
```

The top six backs carry *identical* PAR — the ADP curve is not monotone there and
`expected.tiers` pools ranks it cannot order rather than inventing one. RB1→RB8
costs 5.2 points; WR1→WR8 costs 36.8. Seven times steeper.

`cost_of_waiting`, built long ago and never surfaced, already priced his exact
scenario at his real picks:

| at pick 4 | best PAR | cost of waiting to 17 |
|---|---|---|
| RB | 72.6 | 6.6 |
| WR | 54.9 | 15.6 |

RB@4 + WR@17 = 111.9. WR@4 + RB@17 = 120.9. **Taking the receiver first is worth
+9.0 PAR**, and the margin is exactly the difference between the two waiting
costs.

**The generalisation, which is the point:** PAR is a *level* and the decision is
a *shape*. Between two positions you intend to fill anyway, take the one that is
more expensive to wait on. A board sorted on PAR alone recommends the wrong pick
whenever the curves have different slopes, which on this board they violently do.

Shipped: a `drop` column beside `par` (pick-independent, works with no league and
no handle), a cost-of-waiting panel driven by the real pick list, and a "what
this board assumes" panel that states draft demand rather than the format label.

**Not shipped, and declined with evidence: switching the board to 1QB.** See
`HANDOFF.md`. The short version is that it would raise QB demand from 7 to 10 and
make quarterbacks *more* valuable, which is the reverse of the intent, while
silently returning 20 undraftable keepers to the board.

## 12. The tiebreak was ADP — 2026-08-14

Zach, on the shipped board: *"why is Henry above Cook? Why are the elite WRs so
far down over low end RB1s? replacement value can't be the sole driver here?"*

Both correct, and the first is a defect.

**Henry above Cook had no justification.** Identical PAR (72.6), identical
replacement, same indistinguishable group. `board_rank` was
`rank("ordinal", descending=True)` over `par`, and an ordinal rank breaks ties by
**row order** — which traces straight back to the ADP the pool arrived in. Henry
led because the market drafts him at 12.8 and Cook at 18.8. *A tool whose entire
purpose is to disagree with the market was resolving every tie by deferring to
it.*

And the board already carried the number that contradicted it. Inside that
nine-player "indistinguishable" group `quality_pct` ran from 13.6 to 100:

| old rank | player | par | quality_pct | value_gap |
|---|---|---|---|---|
| 4 | McCaffrey | 72.6 | 27.3 | −65.9 |
| 7 | Achane | 70.5 | **100.0** | +13.6 |
| 8 | Barkley | 67.4 | **13.6** | −70.5 |
| 11 | Josh Jacobs | 53.9 | 56.8 | −22.7 |
| 12 | Ja'Marr Chase | 52.2 | **91.1** | −7.1 |

A 57th-percentile player ranked above a 91st-percentile one.

**On the second question, the decomposition backs him up.** The RB1-vs-WR1 PAR
gap is 14.4, of which **9.8 (68%) is replacement level** and only 4.6 is
projected points (181.8 against 177.2 — nearly a wash). Replacement is not the
sole driver but it is the dominant one. That said it is not arbitrary: flex
demand comes from `scoring._greedy_flex`, which assigns each flex slot to
whichever eligible position is best at its margin. **PAR is working as designed;
the design answers a narrower question than a big board implies**, and `drop`
is the corrective rather than a change to PAR.

### Shipped: `board.rank_board`

Lexicographic on two keys — **not a blend**, because PAR alone decides the block
and quality is consulted only where PAR is provably indifferent:

1. **`block`** — dense rank of the *group's* PAR. If the pooled standard error
   cannot separate nine backs, they share one block and the board stops
   pretending otherwise.
2. **`quality_pct` within a block** — an independent read rather than a circular
   one.

`par` no longer falls monotonically down the board, which is the fix rather than
a bug, and the UI says so.

**The caveat ships with it, in the docstring, the glossary and on screen:**
`quality_pct` is last season's per-opportunity efficiency, and this repo has
twice measured that such signals do not beat ADP out of sample (`breakout.py`
+0.035 AUC, `projection.py` +0.002 Spearman, both covering zero). The claim made
here is *not* "quality beats ADP" — it is "where the ADP curve has no resolution,
an independent read beats the order rows happened to arrive in." **The block is
measured; the order inside it is not**, and displaying the block rather than a
dense 1..159 rank is what keeps that distinction visible.

Also fixed while in there: `build` ended with `sort("par", descending=True)` and
no `nulls_last` — the documented trap, latent only because no PAR is null today.

## 13. Comparables, weighted and pointed at blocks — 2026-08-14

Zach: *"the metrics on the research board are great, are some of those factored
into the clustering exercise and who does player x look like?"*

The same metrics, yes. **The same weighting, no** — and that was the finding.

`quality_score` is a stability-weighted mean of the standardized features
(`archetypes._weighted`): each column is weighted by its year-over-year
correlation, computed rather than hardcoded, and measured to help — rank
correlation with next season's points went 0.464 → 0.502 at WR and 0.455 → 0.492
at TE. Every **distance** built from that identical matrix was a bare
`np.linalg.norm`, giving each column an equal vote.

The spread being ignored is real. RB quality features run from `tprr` at 0.402
down to `ryoe_per_att` at 0.202 — twofold — so two backs could read as similar
largely on a fluky efficiency season neither would repeat, while the score
computed from the same matrix already knew better.

Fixed in `archetypes._distance`, used by `neighbors` and `valuation.comparables`.
Normalised to mean weight 1, so it reduces **exactly** to the old unweighted
number when weights are flat: a reweighting, not a change of units. It changes
answers — the neighbours of James Cook reorder and the sixth comp changes
identity outright.

**One correction to the question: there is no clustering.** k-means was retired
for topping out at a silhouette of 0.29, "a partition of a continuum rather than
real groups". What survives is the distances, without hard labels.

### Pointed at the block problem

`neighbors` takes `restrict_to`, so the pool can be one of the board's blocks
rather than the whole position. This answers a question §12 *created*: the board
now says nine backs are one asset, which is a claim about **value** and silent on
**type**.

It splits them immediately. Inside block 1, Cook's nearest are Taylor (2.01) and
Henry (2.27); McCaffrey's nearest is Gibbs (3.37) — and Cook and McCaffrey are
each other's **furthest** at 5.87. Same block, same price, two different players.

Note the standardization happens over the restricted pool, so "alike among these
nine" is the question asked. **Those distances do not compare with the Research
tab's**, which standardize over the whole position, and the UI says so.

Found while building: the board's `player_id` is FFC's Int64 and `scores` keys on
the String gsis_id. Joining them without an alias let polars suffix the newcomer
and handed back the wrong namespace. Written up in `CLAUDE.md`.

## 11. Landscape, cut — 2026-08-13

Zach: *"leaning towards canning it as it's not really bringing any valuable info
to the table. The one thing that is kind of interesting is the drop off
visuals."*

Agreed, and split three ways rather than deleted wholesale — the tab had four
sections and exactly one was shaped like a decision.

| section | went | why |
|---|---|---|
| The shape of the dropoff | **Big Board** | The picture behind `drop`. The only decision-shaped thing in the tab. |
| PAR per starting slot | Research | PAR's derivation. Settled once, revisited rarely, and useful for auditing a surprise on the board. |
| Who occupies the top of the board | Research | Context. **Counts level only** — it would have counted all six identical RBs as top-24 players and missed the receiver behind them. |
| Are the top players taking a bigger slice? | **deleted** | A good article. Changes no pick. |

The deletion is the part to defend, and the finding is preserved rather than
lost: over 2018-2025 concentration is flat for QB/RB/WR, and **tight end moved
the other way** — top five from 31.7% of the position to 23.9%. That is a real
result and it is recorded in `landscape.concentration`'s docstring, which also
says out loud that the function is deliberately unreachable so a future session
neither re-surfaces it nor deletes it as an orphan.

Five tabs, down from six. `app.py` fell ~90 lines despite gaining three panels.

---

*Record where this turns out to be wrong, inline, the way the dashboard specs
did.*
