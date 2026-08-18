# WEB_SPEC.md — the ff-edge website

**Drafted 2026-08-17. Agreed 2026-08-18 (FastAPI + htmx; parallel build).**
**W0 and W1 shipped 2026-08-18. 451 tests pass, 2 skip.**

> ## State
>
> | block | status |
> |---|---|
> | **W0** skeleton, design system, header, memo layer | **done** |
> | **W1** Big Board page | **done** — full parity incl. warnings, demand line, usage toggle, CSV, cost-of-waiting, block similarity, dropoff charts |
> | W2 Draft Day | not started — placeholder page ships |
> | W3 Player page (E1) | not started — placeholder page ships |
> | W4 Research | not started — placeholder page ships |
> | W5 Reference, polish | not started |
>
> ```bash
> FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run uvicorn web.server:app --reload
> ```
>
> **Measured:** cold board render 7.3s (the `board.build` chain, same as
> Streamlit's first paint), **warm 0.04s for 250 rows**, htmx fragment swap
> **0.004s**. That is the answer to A3's "how many seconds to a decision" —
> filtering the board no longer re-runs the pipeline.
>
> **What building it found:** an unset `FF_EDGE_LEAGUE_ID` silently boards a
> *different league* — see `DRAFT_CHECKLIST.md` A0. The website names the
> resolved league on every page and flags a guessed one; `app.py` cannot.
> This is the second time surfacing something in a UI exposed a defect that a
> green test suite could not see, after Block B's season literal and wrong
> ADP market.

Replace the Streamlit prototype with a modern, professional, responsive website.
Backend stays Python — every number on every page keeps coming from `src/`.
This spec is the agreement gate required by `CLAUDE.md` before any dashboard
build.

---

## 1. The one constraint that shapes everything: the calendar

**Shiva Bowl drafts 08-22 — five days out.** A3 (sit with the app) is still
undone and expires on the 22nd.

So the website is a **parallel build**. Rules:

- `app.py` is not touched. Not one line. Streamlit remains the draft tool for
  08-22 unless the new site is *obviously* better by the 20th and you say so.
- The new site reads the same cache, the same env vars, the same `src/`
  functions. Nothing about the data layer forks.
- Target: **the new site is primary by the 828 draft (09-06)** — which aligns
  with checklist item C1, because the profile selector 828 needs ships as a
  first-class control here rather than being retrofitted into the Streamlit
  sidebar.
- Streamlit is retired only after the new site has been used through a real
  draft.

## 2. Architecture — the analogy first

Streamlit is a restaurant where the chef also waits every table: each
interaction re-runs the whole kitchen top to bottom. The new site splits the
jobs the way real restaurants do — **the kitchen (`src/`, unchanged) cooks; a
thin waiter (FastAPI) carries plates; the dining room (HTML/CSS) is designed
once and stays put.** When you change a filter, only your plate is re-made, not
the whole menu.

```
ff-edge/
├── app.py              # untouched Streamlit prototype, retired later
├── web/
│   ├── server.py       # FastAPI app factory + uvicorn entry — thin
│   ├── deps.py         # profile/league resolution from env + query params
│   ├── memo.py         # in-process TTL memo (replaces st.cache_data)
│   ├── api.py          # JSON routes: /api/board, /api/player/{id}, ...
│   ├── pages.py        # HTML routes rendering Jinja templates
│   ├── charts.py       # Altair chart builders → Vega-Lite JSON (Python-authored)
│   ├── format.py       # column pretty-naming, number formatting (port of _header/table)
│   ├── templates/      # Jinja2 — base.html, one file per page, partials/
│   └── static/         # styles.css, app.js, vendored htmx + vega libs, fonts
└── src/                # unchanged — the entire brain
```

### The stack, and why each piece

| piece | choice | why |
|---|---|---|
| server | **FastAPI + uvicorn** | the standard Python web layer; you already know Pydantic; JSON API falls out for free (and the MCP server / future tools can share it) |
| templating | **Jinja2** | server-rendered HTML — the page you see is produced by Python you can edit |
| interactivity | **htmx** (vendored, ~14KB) | clicks/selects swap page *fragments* rendered by the server. No Node, no build step, no npm — every behavior is a Python route returning a template. This is the load-bearing choice: it keeps the whole site greppable and tweakable by you |
| charts | **Altair in Python → Vega-Lite JSON → vendored vega-embed renders it** | charts stay *authored in Python* (already a dep, already the house chart library); the browser is only a renderer |
| CSS | **hand-written design system**, one `styles.css` with custom properties | no Tailwind/node toolchain; the design tokens are readable and yours |
| assets | **everything vendored/local** — htmx, vega, fonts | the site works with no CDN and no network beyond the Sleeper/nflverse calls `src/` already makes. On draft day, nothing on the wire but data |

Explicitly rejected: a React/Vite SPA. It is the "modern" default, but it moves
half the codebase into a language and toolchain you aren't trying to learn right
now, requires a Node build step, and buys nothing this app needs — every
interaction here is "recompute a table/chart from polars and show it," which is
exactly the request/fragment shape htmx serves. If a real-time draft sync ever
demands it, the JSON API built here is the foundation a SPA would sit on anyway.

### What this quietly fixes

The **pyarrow segfault class dies structurally**. `st.dataframe` round-trips
every polars frame through pandas into Arrow bytes — that's where 25.0.0
killed the interpreter. The web layer serializes polars straight to JSON/HTML;
pandas and Arrow leave the render path entirely. (The `!=25.0.0` pin stays
while `app.py` lives.)

The **Streamlit misuse-warning trap** and session-state key sprawl (five player
selectboxes across four tabs, E1's root cause) also don't exist here — state is
the URL. `?player=...`, `?pos=RB`, `?profile=standard_12`. Bookmarkable,
shareable, no hidden session dict.

## 3. Information architecture — five pages, one new

Ported surfaces keep their content and their captions — every "what this board
assumes" warning, every honesty note. `board.build()['warnings']` renders as a
prominent banner, never a footnote: an unpriceable board must say so in a
sentence, on screen, above the fold.

### `/` — Big Board *(port, leads for the same reason it leads now)*
The one list. Rank, block, `par_env` and its layers (`par`, `ffb_par`, `ecr`,
`env_swing`), `need`, tiers, the roster-demand line drawn visibly across the
table — above it the board ranks, below it ADP's order and muted styling where
`block` is null, so where the tool stops claiming is *visual*, not a caption.
Usage columns (`age`, snap/target/rz shares) as a toggle, blank-means-unmeasured
preserved. Sticky header, position filter chips, CSV download (A4 lives here).
Cost-of-waiting and block-similarity panels ride along.

### `/draft` — Draft Day *(port; the tab with the deadline)*
Pick selector, targets at your pick, the recommendation panel above its
evidence, live drafted-player removal on refresh. Same
`FF_EDGE_SLEEPER_USER` gating; if the handle is missing the page says exactly
what is missing and what it unlocks.

### `/player/{name}` — Player *(new — this closes checklist E1)*
The structural gap the checklist says costs more than anything else: one page,
one player, everything the repo knows. Board row + rank/block/tier, the four
layers as an argument left to right, usage profile vs position, `archetypes`
neighbors, `peek` screen flags (regression debt, market disagreement, snap
trend), claims touching him, ECR + `ecr_sd`, Footballers panel with
`n_analysts` and staleness, age said out loud (A4c) with the caption that
nothing prices it. Search-first: a fast typeahead in the site header from
anywhere. This page is the reason to rebuild rather than reskin — it's the A4b
block-artifact method, automated per player.

### `/research` — Research *(port)*
The current Research tab's sections as an anchored, sub-navigable page:
Screens (first, open), Footballers disagreement, valuation Board ("who the
market has wrong" folds in here as a section — it's read in the same sitting),
stability, breakout/projection (the honest nulls, kept as such), rookies,
strategy sim (with its deliberate two-FLEX pin caption), claims ledger,
positional value/dropoff/mix.

### `/reference` — Glossary *(port)*
Plus a "how the board is built" pipeline diagram distilled from
`HOW_IT_WORKS.md`'s current half.

### Site-wide controls *(replaces the sidebar)*
A compact header bar: **profile selector** (`shiva_bowl` / `standard_12` /
custom — C1), teams/scoring overrides in a popover, league status ("Live from
Sleeper · 2026" or the set-your-env-vars warning). Every control is a query
param; `profiles.resolve()` still raises on unknown names — a typo 404s with
the sentence, never falls back.

## 4. Design direction (the creative-freedom part)

Identity: **"the war room."** Not a SaaS dashboard template — a broadcast-desk
board. Dark-first (draft rooms are dark; a light theme via `prefers-color-scheme`
using the same tokens), one hot accent used only for *your* picks and
actionable rows, tier bands as subtle background steps, tabular numerals
everywhere, dense tables that earn their density: sticky first column, row hover
that lights the whole argument for that player, nulls rendered as an em-dash
with a "not measured" tooltip — never a zero, never blank-that-looks-broken.

Typography: a vendored grotesk for headings, system stack for body, a vendored
mono for numbers. Responsive: three-pane on desktop, single column on a phone —
Draft Day is explicitly designed to work on a phone beside the laptop on the
clock. Motion: fragment swaps fade in 150ms; nothing bounces. The dataviz skill
gets loaded before any chart code is written; charts follow one palette shared
with the table tier bands.

## 5. Data & caching

- `web/memo.py`: a small TTL memo (dict + monotonic clock, ~40 lines) replacing
  `st.cache_data`, TTLs from `config.py` where they exist. Per-process; a
  server restart is a cache clear, same as Streamlit rerun semantics.
- API routes return `{data: [...], warnings: [...], meta: {...}}` — warnings are
  first-class in the contract, matching `board.build()`.
- Env vars unchanged: `FF_EDGE_LEAGUE_ID`, `FF_EDGE_SLEEPER_USER`,
  `FF_EDGE_PROFILE` as the default profile, overridable per-request by query
  param.
- Run: `FF_EDGE_LEAGUE_ID=... uv run uvicorn web.server:app --reload`
- New deps: `fastapi`, `uvicorn`, `jinja2`. Nothing else.

## 6. Testing — inheriting the repo's discipline

- FastAPI `TestClient` tests per page: 200, warnings surface when the board
  can't price, empty-frame paths render sentences not blank tables. Run in the
  main tree (they read the cache), like the other 14 data-touching test files.
- **The `test_draft_day.py` lesson carries over**: one test walks the Draft Day
  page forward through every pick via the real route, in order, because
  render-once-passes is exactly what hid the last crash.
- The player page gets a test that requests a player with *no* usage data and
  asserts the page says "not measured" rather than 0 — proving the assert can
  fail, per the repo rule for did-not-happen tests.
- JSON API responses snapshot column names, so a silent polars suffix
  (`player_id_right`) breaks a test instead of a page.

## 7. Build order — block by block, each one shippable

| block | what | done means |
|---|---|---|
| **W0** | skeleton: FastAPI app, base template, design system, header/profile bar, memo layer | site boots, looks like itself, profile switch works |
| **W1** | **Big Board page** | full parity with the tab incl. warnings banner, demand line, usage toggle, CSV |
| **W2** | **Draft Day page** + the pick-walk test | dry-runnable end to end |
| **W3** | **Player page + typeahead** (E1) | one player, whole picture, one URL |
| **W4** | Research page (sections ported in checklist order: screens → footballers → valuation → the rest) | parity |
| **W5** | Reference, polish pass, phone pass | — |

W0–W1 are this-week sized without touching draft prep. W2 before 08-22 is a
stretch goal, not a promise — Streamlit stays the Shiva Bowl tool unless W2
lands early and you bless it. W3–W5 target 09-06.

## 8. Out of scope, on purpose

- Deployment/hosting. It runs on `localhost` like the Streamlit app. The repo
  is public and `data/` is not — nothing here changes the containment rules.
- Auth, multi-user, websockets/live-sync. The JSON API leaves the door open.
- Any change to `src/` beyond *additive* helpers if a port reveals a gap —
  and those get their own tests.
- Deleting `app.py`. That happens after 09-06, as its own decision.

---

*Agreement gate: build starts when Zach signs off on this file.*
