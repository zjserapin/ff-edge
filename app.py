"""ff-edge — draft research for the Shiva Bowl.

    uv run streamlit run app.py

Lives at the repo root rather than in src/ because `streamlit run` puts the
script's own directory on sys.path, and from src/ the `from src import ...`
imports below would not resolve.

Caching convention, applied everywhere in this file: a scoring dict is not
hashable, so every cached function takes `scoring_key` — a tuple of sorted
(key, weight) pairs — and rebuilds the dict inside. Nothing that reads
st.session_state is ever cached; that is the standard way a Streamlit board goes
stale while looking fine.
"""

from __future__ import annotations

from typing import Any, Mapping

import altair as alt
import polars as pl
import streamlit as st

from src import landscape as ls
from src import scoring as sc
from src import theme
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    FEATURE_SEASONS,
    LEAGUE_ID,
)

st.set_page_config(page_title="ff-edge", page_icon="🏈", layout="wide")


# --- Cache boundaries -------------------------------------------------------


def _key(scoring: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Make a scoring dict hashable for @st.cache_data."""
    return tuple(sorted((k, float(v)) for k, v in scoring.items()))


@st.cache_data(show_spinner=False)
def _league(league_id: str) -> dict[str, Any]:
    return sc.league_settings(league_id)


@st.cache_data(show_spinner=False)
def _scoring_history(league_id: str) -> pl.DataFrame:
    return sc.scoring_history(league_id)


@st.cache_data(show_spinner="Scoring history…")
def _season_points(scoring_key: tuple[tuple[str, float], ...]) -> pl.DataFrame:
    return sc.score_season(scoring=dict(scoring_key))


@st.cache_data(show_spinner=False)
def _par(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return ls.par_by_position(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _concentration(
    scoring_key: tuple[tuple[str, float], ...], shares: tuple[int, ...]
) -> pl.DataFrame:
    return ls.concentration(
        scoring=dict(scoring_key), shares=shares, season_points=_season_points(scoring_key)
    )


@st.cache_data(show_spinner=False)
def _scarcity(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
    max_rank: int,
    basis: str,
) -> pl.DataFrame:
    return ls.scarcity_curve(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        max_rank=max_rank,
        basis=basis,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _cross(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return ls.cross_positional_value(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _replacement(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return sc.replacement_level(
        _season_points(scoring_key),
        list(roster),
        teams,
        dict(flex_split) if flex_split else None,
    )


# --- Sidebar ----------------------------------------------------------------


def _sidebar() -> dict[str, Any]:
    st.sidebar.title("League")

    league = _league(LEAGUE_ID)
    if league["source"] != "sleeper":
        st.sidebar.caption("⚠️ Sleeper unreachable — using saved 2026 settings.")
    else:
        st.sidebar.caption(f"Live from Sleeper · {league['season']} season")

    teams = st.sidebar.number_input(
        "Teams", min_value=4, max_value=16, value=int(league["teams"]), step=1
    )

    st.sidebar.subheader("Scoring")
    scoring = dict(league["scoring"])
    rec = st.sidebar.slider(
        "Points per reception", 0.0, 1.0, float(scoring.get("rec", 0.5)), 0.25
    )
    pass_td = st.sidebar.slider(
        "Points per passing TD", 3.0, 6.0, float(scoring.get("pass_td", 4.0)), 1.0
    )
    scoring["rec"] = rec
    scoring["pass_td"] = pass_td

    st.sidebar.subheader("FLEX allocation")
    mode = st.sidebar.radio(
        "How FLEX slots are filled",
        ["Computed from the data", "Set manually"],
        help=(
            "Computed assigns each FLEX slot to whichever of RB/WR/TE is worth "
            "more at its next open rank, using that season's real scoring. "
            "This moves replacement level by several ranks, so it is worth "
            "seeing both."
        ),
    )
    flex_split: dict[str, float] | None = None
    if mode == "Set manually":
        rb = st.sidebar.slider("Share of FLEX going to RB", 0.0, 1.0, 0.5, 0.05)
        te = st.sidebar.slider("Share going to TE", 0.0, 1.0 - rb, 0.05, 0.05)
        flex_split = {"RB": rb, "WR": max(0.0, 1.0 - rb - te), "TE": te}
        st.sidebar.caption(
            f"RB {flex_split['RB']:.0%} · WR {flex_split['WR']:.0%} · TE {flex_split['TE']:.0%}"
        )

    unmapped = sc.unmapped_keys(scoring)
    if unmapped:
        with st.sidebar.expander(f"Not modeled ({len(unmapped)} scoring rules)"):
            st.caption(
                "weekly_stats has no team-defense rows, so DST and IDP scoring "
                "cannot be computed from this data at all. Every chart here "
                "covers QB/RB/WR/TE only."
            )
            st.code("\n".join(unmapped), language=None)

    return {
        "league_id": LEAGUE_ID,
        "teams": int(teams),
        "scoring": scoring,
        "scoring_key": _key(scoring),
        "roster_positions": tuple(league["roster_positions"] or DEFAULT_ROSTER_POSITIONS),
        "flex_split": tuple(sorted(flex_split.items())) if flex_split else None,
        "dark": theme.is_dark(),
    }


# --- Landscape tab ----------------------------------------------------------


def _tab_landscape(p: dict[str, Any]) -> None:
    dark = p["dark"]
    pos_scale = theme.position_scale(dark)

    st.subheader("How positional value has moved")
    st.caption(
        f"Seasons {FEATURE_SEASONS[0]}–{FEATURE_SEASONS[-1]}, every season "
        "rescored under the settings in the sidebar. Weeks 1–14 only — the "
        "fantasy regular season."
    )

    hist = _scoring_history(p["league_id"])
    if hist.height > 1 and hist.get_column("rec").n_unique() > 1:
        changes = " · ".join(
            f"{r['season']}: {r['rec']} PPR, {r['flex_slots']} FLEX"
            for r in hist.iter_rows(named=True)
            if r["season"]
        )
        st.info(
            f"**This league changed its own rules.** {changes}. History below is "
            "recomputed under today's settings, which is the right basis for a "
            "2026 draft — but those seasons were not played this way.",
            icon="ℹ️",
        )

    par = _par(p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"])
    if not par.height:
        st.warning("No scored seasons found. Run `uv run python -m src.bootstrap --light`.")
        return

    # --- Value above replacement ---
    st.markdown("#### Value above replacement, per starting slot")
    st.caption(
        "Points per game a starter at this position gives you over the best "
        "player you could have had for free. The only unit that compares a "
        "quarterback to a tight end."
    )

    par_pd = par.to_pandas()
    line = (
        alt.Chart(par_pd)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("par_mean_starter:Q", title="PAR per game"),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("par_mean_starter:Q", title="PAR/game", format=".2f"),
                alt.Tooltip("demand:Q", title="Starters league-wide", format=".0f"),
                alt.Tooltip("replacement_rank:Q", title="Replacement rank"),
                alt.Tooltip("replacement_ppg:Q", title="Replacement PPG", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    labels = (
        alt.Chart(par_pd[par_pd["season"] == par_pd["season"].max()])
        .mark_text(align="left", dx=8, fontSize=11, fontWeight=600)
        .encode(
            x=alt.X("season:O"),
            y=alt.Y("par_mean_starter:Q"),
            text="position:N",
            color=alt.Color("position:N", scale=pos_scale, legend=None),
        )
    )
    st.altair_chart(theme.base_chart(line + labels, dark), use_container_width=True)

    with st.expander("Replacement level behind these numbers"):
        st.caption(
            "Replacement is the first player past the starting pool. Per-game "
            "replacement is ranked on per-game among players with 8+ games — "
            "ranking on season totals lets an injured star who happened to land "
            "on the slot set the baseline."
        )
        repl = _replacement(
            p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"]
        )
        st.dataframe(repl.to_pandas(), use_container_width=True, hide_index=True)

    # --- Scarcity curves ---
    st.markdown("#### The shape of the dropoff")
    st.caption(
        "The average above tells you how much a position is worth; this tells "
        "you *where*. A cliff is worth reaching for, a gentle slope is worth "
        "waiting on, and both can produce the same average."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        basis = st.radio(
            "Measured as",
            ["Points per game", "Season total"],
            horizontal=False,
            key="scarcity_basis",
            help=(
                "Per game compares seasons fairly and asks what a healthy player "
                "at this tier is worth (8+ games). Season total counts "
                "availability as value, which is what you actually draft."
            ),
        )
    basis_key = "ppg" if basis == "Points per game" else "total"
    with c2:
        max_rank = st.slider("Ranks to show", 12, 60, 36, 6, key="scarcity_rank")

    scarcity = _scarcity(
        p["scoring_key"],
        p["roster_positions"],
        p["teams"],
        p["flex_split"],
        max_rank,
        basis_key,
    )
    seasons = sorted(scarcity.get_column("season").unique().to_list())
    chosen = st.multiselect(
        "Seasons", seasons, default=seasons[-3:], key="scarcity_seasons"
    )
    if chosen:
        y_field, y_title = (
            ("ppg", "Points per game")
            if basis_key == "ppg"
            else ("fantasy_points", "Season points")
        )
        sc_pd = scarcity.filter(pl.col("season").is_in(sorted(chosen))).to_pandas()
        curves = (
            alt.Chart(sc_pd)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("pos_rank:Q", title="Positional rank"),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color(
                    "season:N",
                    scale=alt.Scale(
                        domain=sorted(chosen),
                        range=theme.ordinal_steps(len(chosen), dark),
                    ),
                    title="Season",
                ),
                tooltip=[
                    alt.Tooltip("player_display_name:N", title="Player"),
                    alt.Tooltip("season:O", title="Season"),
                    alt.Tooltip("pos_rank:Q", title="Rank"),
                    alt.Tooltip("ppg:Q", title="PPG", format=".1f"),
                    alt.Tooltip("fantasy_points:Q", title="Season pts", format=".0f"),
                    alt.Tooltip("games:Q", title="Games"),
                ],
            )
            .properties(height=200, width=200)
            .facet(column=alt.Column("position:N", title=None, sort=list(theme.position_colors())))
        )
        st.altair_chart(theme.base_chart(curves, dark), use_container_width=True)

    # --- Cross-positional mix ---
    st.markdown("#### Who actually occupies the top of the board")
    st.caption(
        "Every position ranked together on value over replacement. This is the "
        "early-RB argument as a countable fact rather than an opinion."
    )
    cross = _cross(p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"])
    mix = ls.positional_mix(cross)
    cutoff = st.select_slider(
        "Top N by value over replacement", options=[12, 24, 36, 48], value=24
    )
    mix_pd = mix.filter(pl.col("cutoff") == cutoff).to_pandas()
    bars = (
        alt.Chart(mix_pd)
        # A 2px stroke in the surface color, not a border: it reads as a gap
        # between stacked segments so adjacent positions stay countable.
        .mark_bar(cornerRadiusEnd=4, stroke=theme.surface(dark), strokeWidth=2)
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("n:Q", title=f"Players in the top {cutoff}", stack="zero"),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("n:Q", title="Players"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(theme.base_chart(bars, dark), use_container_width=True)

    # --- Concentration ---
    st.markdown("#### Are the top players taking a bigger slice?")
    st.caption(
        "Share of positional points held by the top N, among a pool of roughly "
        "three times the number of starters. An uncapped pool would make this a "
        "measure of how many replacement bodies the league cycled through."
    )
    conc = _concentration(p["scoring_key"], (5, 15, 30))
    conc_pd = conc.to_pandas()
    conc_chart = (
        alt.Chart(conc_pd)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("share:Q", title="Share of positional points", axis=alt.Axis(format="%")),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("top_n:Q", title="Top N"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
                alt.Tooltip("pool_size:Q", title="Pool"),
            ],
        )
        .properties(height=180)
        .facet(column=alt.Column("top_n:N", title="Top N players"))
    )
    st.altair_chart(theme.base_chart(conc_chart, dark), use_container_width=True)

    with st.expander("Table view"):
        st.caption("Every chart above, as numbers.")
        st.markdown("**Value above replacement**")
        st.dataframe(par.to_pandas(), use_container_width=True, hide_index=True)
        st.markdown("**Positional mix of the top of the board**")
        st.dataframe(mix.to_pandas(), use_container_width=True, hide_index=True)
        st.markdown("**Concentration**")
        st.dataframe(conc.to_pandas(), use_container_width=True, hide_index=True)


def _placeholder(name: str, phase: str) -> None:
    st.subheader(name)
    st.info(f"Not built yet — {phase}.", icon="🚧")


def main() -> None:
    st.title("ff-edge")
    p = _sidebar()

    landscape, players, strategy, board = st.tabs(
        ["Landscape", "Players", "Strategy", "Board"]
    )
    with landscape:
        _tab_landscape(p)
    with players:
        _placeholder("Players", "phase 2 (clusters) and phase 3 (breakout backtest)")
    with strategy:
        _placeholder("Strategy", "phase 4 (draft simulation)")
    with board:
        _placeholder("Board", "phase 5 (draft-day view)")


if __name__ == "__main__":
    main()
