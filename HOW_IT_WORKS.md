# How ff-edge works

A guided walkthrough, not a reference — for when you've been away from this
project and need to rebuild the mental model before touching code again.
README.md has the numbers; this file has the shape.

> ### Status — merged 2026-08-17, and the tab half is out of date
>
> Written against the **six-tab** app (Landscape / Players / Screen / Strategy /
> Board / Glossary). That layout no longer exists. **The pipeline half of this
> file — the ingest → join → transform → measure → model chain, and every
> argument about why each stage exists — is still accurate and is the reason the
> file was kept.** The tab-by-tab walkthrough below is history.
>
> Where the old tabs went:
>
> | this file says | where it lives now |
> |---|---|
> | Landscape | **cut 08-13.** Dropoff curves → Big Board; PAR-per-slot and positional mix → Research; concentration-over-time deleted, finding preserved in `landscape.concentration`'s docstring |
> | Players | folded into **Research** as expanders |
> | Screen | folded into **Board** / Research; the claims ledger still backs it |
> | Strategy | folded into **Research**, labelled as pinned to the old format |
> | Board | still **Board** — `valuation.py`, quality against price |
> | — | **Big Board** and **Draft Day** are both new since this was written, and Big Board is now the front door |
>
> Current strip: **Big Board / Draft Day / Board / Research / Glossary.**

## The map of the docs, so you know where to look

Knowing which doc is which saves you from reading a dated decision log when you
wanted the current picture, or vice versa. **Corrected 2026-08-17** — the list
below used to name four files, three of which have since moved or been renamed.

| File | What it's for | When to open it |
|---|---|---|
| `DRAFT_CHECKLIST.md` | What to do next, ordered by deadline. | Start here if you are picking work up. |
| `README.md` | The reference. Setup, the module table, findings with real numbers, verified checks. | "What does module X do" / "what did the analysis actually find." |
| `HOW_IT_WORKS.md` | This file. Why the pipeline is shaped the way it is and how the stages hand off. | You've lost the thread and need to reorient before changing anything. |
| `CLAUDE.md` | Every silent-failure trap in the repo. | **Before touching code.** Not optional. |
| `BIG_BOARD_SPEC.md` | The Big Board, plus eighteen numbered records of what each change to the board found. | Working on ranking, blending, or the board's ordering. |
| `RESEARCH_SPEC.md` | Direction and the Phase 0/1/2 build order. | Deciding what to build. Read its correction block first. |
| `FOOTBALLERS_SPEC.md` | The Fantasy Footballers layer. Data built, display half built. | Working on the blend or its display. |
| `CLAIMS_SPEC.md` | The build contract for the claims ledger: schema, scoring formula, failure modes. | Working on `claims.py`, `news.py`, `llm.py`, or `prompts.py`. |
| `FANTASYPROS_IDEAS.md` | Brainstorm. Its lead finding shipped; the seven-years-of-ECR-history one has not. | Looking for the next measurement. |
| `HANDOFF.md` | Current state around the checklist. | Start of a session. |
| `docs/archive/` | `ANALYSIS_SPEC.md` and both dashboard specs — superseded on direction, findings sections still stand. | Before re-proposing something they declined. |

## The one-sentence version

Beyond what the draft market already prices into a player's ADP, is there
anything measurable — usage, efficiency, a reported role change — that
predicts how he does *against that price*? Not "who is good," since ADP
already has an opinion on that; specifically, "where is the market's opinion
wrong, in a way that can be shown rather than argued."

## The analogy

This is the same shape as testing a trading signal against a market price.
ADP is the market. A player's fantasy output relative to his draft cost is
his return. The project:

1. builds features the way a quant builds factors — from raw data, not from
   opinion,
2. checks whether each factor is even stable before trusting it with money
   (here: before trusting it in a model),
3. backtests the signal against the price it's supposed to beat, out of
   sample, never on a random split,
4. separately backtests *strategy* — not "which player" but "which overall
   approach to acquiring players" — because a good signal and a good strategy
   are different questions,
5. keeps a running scorecard of where the signal and the price currently
   disagree,
6. and hands off the one thing none of this can see — *who gets picked for a
   bigger role* — to a human, with automated research support instead of a
   guess.

If you think in ingest → transform → model → output, that maps directly:
`nflverse.py` / `sleeper.py` / `adp.py` / `news.py` are ingest, `ids.py` is
the join that makes ingest usable, `features.py` / `context.py` are
transform, `stability.py` / `breakout.py` / `projection.py` / `rookies.py` /
`archetypes.py` / `simulate.py` are model, and `app.py` is output.

## Why "data layer, not a ranking product"

The README says this on line one and it's worth understanding *why*, because
it explains every design choice downstream. A ranking product hides its
inputs and sells you a number. This project's stance is the opposite: every
chart shows its formula, every model shows its calibration against a naive
baseline, and every negative result ships in the app instead of getting
quietly dropped. That posture is what makes "we tried six things and none of
them worked" a legitimate, useful section of the README instead of something
to bury — it's evidence the measurement apparatus is honest, which is the
only thing that makes the positive findings (the rookie model, the two
QB-timing strategies) worth believing either.

## Stage by stage, tied to what you actually click on

### Before any tab: ingest and join

Nothing in the app touches a network. `bootstrap.py` does that once, up
front, pulling nflverse tables, Sleeper league data, and ADP snapshots
through `cache.py` (a TTL-aware parquet/JSON cache — the TTL is chosen per
source by how fast that source actually changes, from `static` at 30 days to
`live` at 1 hour). Every ID from every source gets cast and crosswalked in
`ids.py` before anything downstream joins on it — this is the single most
likely place a project like this breaks quietly (see README's "ID dtype
trap"), so it's centralized rather than repeated per module.

### Landscape tab — the descriptive baseline

Powered by `scoring.py` + `landscape.py`. Answers "how has positional value
moved" with no model involved: points above replacement by position and
season, the shape of the drop-off at each position (scarcity), who actually
occupies the top of the board, and how concentrated the points are among the
top players. This tab exists first in the pipeline's logic even though it's
first in the tab order for the same reason: you need to know what a position
is *worth* before any later section's claim about a *player* means anything.

### Players tab — does the measurement even mean anything, then does it predict

This tab is ordered on purpose, and the order is the argument:

1. **Stability first** (`stability.py`) — before fitting anything, rank every
   qualified player within his position and season, pair each player-season
   with his own next one, and correlate the two percentiles. This asks
   nothing about outcomes or models, only "does this number repeat." Six
   metrics failed this and were removed from every feature set before a
   backtest ever touched them — this is a gate, not a footnote.
2. **Quality and opportunity** (`features.py`, `archetypes.py`) — the
   metrics that survived, split onto two axes because they persist at very
   different rates (opportunity ~0.5, quality ~0.3-0.4). `archetypes.py`
   produces nearest-neighbor comparables rather than clusters — clustering
   was tried, measured, and removed (silhouette 0.19-0.29, and it made the
   downstream model worse); the decision and the numbers are in `README.md`
   under "Clustering on volume just rediscovers ADP."
3. **The backtest** (`breakout.py`, `projection.py`) — the actual beat-ADP
   question, first as a binary label then, because the binary version's
   confidence interval was too wide to mean anything, as a continuous rank
   correlation that measures the same gap about five times more precisely.
   Both say no, with the interval to prove it rather than assert it.
4. **Rookies** (`rookies.py`) — a deliberately separate model, since a rookie
   has no prior-season usage, the only input the veteran model uses. This is
   the one model that works (0.592 out-of-sample correlation).

### Screen tab — the model's blind spot, handed to a human

`promotion.py` and `claims.py`. The measured reason this tab exists: nothing
in the data predicts *who gets a bigger role* (vacated targets r = -0.04,
prior quality 0.02 — see `HANDOFF.md` §3). So the division of labor is
explicit: you name whose role is growing, from camp news or a depth chart,
and `promotion.py` grades that player against the base rate for
historically-promoted players with a similar profile — trust markers (snaps,
red-zone share) at running back, efficiency (yards per route run) at
receiver and tight end, because those are the criteria that actually
survived the same stability-style check applied to this specific cohort.

The **claims ledger** exists to make "you name whose role is growing" less of
a cold start. It's a separate, smaller pipeline inside this one — see below.

### Strategy tab — a different question, deliberately

`simulate.py`. The backtest above asks whether a *signal* beats price on
*real* usage that already happened. This tab asks a structurally different
question: given how format value has moved historically, does a *draft
strategy* (zero-RB, late-QB, etc.) beat simply following ADP? It's a Monte
Carlo over thousands of simulated drafts and seasons, not a backtest on one
league's real history — the app says explicitly that it simulates drafting,
not in-season management, so the claim is narrower than "what wins."

### Board tab — where the two disagree, today

`valuation.py`. Takes the quality/opportunity split from the Players tab and
compares it directly to current draft price, both converted to
within-position percentiles. The gap is `value_gap`, and it is explicitly a
disagreement score, not a projection — "this project's read differs from the
market's" is a reason to look closer, not a claim of being right.

### Glossary tab — so no other tab requires memorization

`glossary.py`. Every column that appears anywhere in the app has a definition
here, surfaced as a hover tooltip on the table it appears in (`app.py`'s
`table()` helper routes every dataframe through it) and searchable in full in
this tab.

## The claims ledger, close up

This is the newest piece and the most likely place to lose the thread, so it
gets its own section. Full contract: `CLAIMS_SPEC.md`. The pipeline, in the
same ingest → transform → model → output shape as the project overall, just
at a smaller scale and pointed at news instead of stats:

```
ingest          news.py           Google News RSS (per team), nflverse depth
                                   charts, Sleeper trending adds (a detector,
                                   not a claim source — it flags where to look)

extract         llm.py +          Depth-chart diffs need no LLM (a rank
                prompts.py        change is a self-quoting machine claim).
                                   Beat-report text does — Haiku-class model,
                                   prompt in prompts.py, contract: role-change
                                   claims only, verbatim quote required

guard           claims.py         Re-checks that every extracted quote is
                (extract())       really a substring of the source text.
                                   Fails the claim, not the pipeline, if not —
                                   this is the hallucinated-specificity guard

append          claims.py         Append-only parquet, deduped on
                (append())        (date, player, type, direction, source,
                                   quote). Never edited — the ledger doubles
                                   as the labeled dataset the layer gets
                                   judged against later

score           claims.py         tier x specificity x novelty x recency
                (score())         decay, every factor a visible column so a
                                   score is never a mystery

flags           claims.py         Per-player grade (A/B/C/watch/shrinking),
                (flags())         which is what pre-fills the Screen tab's
                                   text box — the user can still override

resolve         claims.py         Checks resolvable claims against what
                (resolve())       weekly usage actually did, which is what
                                   builds source_grades() and is the only
                                   way this layer ever gets validated
```

Three things worth internalizing before touching this code:

- **It cannot be backtested.** There is no historical claims corpus — season
  one *is* the labeled-data collection. The score weights (tier weight,
  specificity weight, the half-life) are hand-set priors, named as such, not
  fitted parameters. Don't mistake them for something more rigorous than they
  are.
- **The atom is a claim, not a player.** Every number the ledger produces has
  to decompose back into the quoted claims that produced it (the Screen tab's
  "decompose a player's flag into its claims" table does exactly this). The
  moment a flag can't be traced to a quote, the layer has become vibes with
  extra steps and the spec says to delete it.
- **Absence of claims is not absence of signal.** A boring veteran starter
  with no news about him is bullish stability, not a data gap. The app says
  this explicitly rather than treating silence as a missing value.

## The findings that actually drive the design (numbers in README.md)

These aren't trivia — each one changed how the code is structured, so
knowing them explains *why* things are built the way they are, not just what
they do:

- **Opportunity persists, quality mostly doesn't** → this is why they're kept
  on two separate axes everywhere instead of blended into one score.
- **Nothing beats ADP on prior-season usage**, measured to ±0.01 → this is
  why the project stopped spending sessions on model architecture and why
  every backtest result ships with its confidence interval front and center.
- **Clustering rediscovers ADP when built on volume, and doesn't survive
  testing even on quality alone** → this is why `archetypes.py` has no
  cluster function anymore, only nearest neighbors.
- **Nothing predicts who gets a bigger role** → this is the entire reason the
  Screen tab and the claims ledger exist instead of one more model.
- **The rookie model works and the QB-timing strategies beat ADP** → the two
  places the project found a real, if modest, edge — everything else is
  scaffolding that makes those two findings trustworthy rather than lucky.

## A short checklist for not getting lost again

- Before adding a new feature column, run it through `stability.py` first.
  If it doesn't repeat year over year, it doesn't belong in a model no matter
  how plausible it sounds.
- Before trusting any backtest number, check the adequacy/events-per-variable
  table next to it. Small samples are the project's binding constraint, not
  model choice.
- Validation is always season-forward, never a random split — `breakout.py`'s
  `season_forward_splits` is the pattern to copy.
- `data/` stays gitignored, always — it holds league data (nine other real
  people's names and history) and the claims ledger.
- `config.ROOT` uses `parents[1]`, not `parent` — see the comment in
  `config.py` for why that one line is the highest-risk line in the project.
- If a result seems too clean, check whether it would survive a permuted-label
  test — `breakout.py` keeps one (shuffled labels score 0.497 AUC, chance) as
  a standing check, specifically because a plausible-looking positive result
  turned out to need one.
