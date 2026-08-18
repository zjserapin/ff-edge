"""Chart builders — authored in Python with Altair, rendered by vendored vega.

The browser is only a renderer: each function returns a Vega-Lite spec (as a
dict) that static/app.js hands to vega-embed. Colors come from src/theme.py,
the one validated palette in this repo — substituting a hue here without
re-running the palette validator is how a chart two leaguemates can't read
ships quietly.
"""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import polars as pl

from src import theme

# **An explicit pixel width, deliberately, rather than `width="container"`.**
# Container sizing renders an empty box here rather than an error: vega-embed
# resolves the width against the parent at embed time, and inside a card that
# has not finished laying out it resolves to nothing and draws a blank chart
# the same size as the space it was given. Two charts shipped that way and
# looked like a rendering bug rather than a sizing one.
#
# A fixed width is honest about the trade: the chart is this wide, and on a
# narrow screen `.chart` scrolls it, which is the same rule every wide table on
# the site follows.
_WIDTH = 720


def _spec(chart: alt.TopLevelMixin, dark: bool) -> dict[str, Any]:
    """Recessive chrome from the shared theme, transparent over the page CSS."""
    spec = json.loads(theme.base_chart(chart, dark).to_json())
    spec["background"] = "transparent"
    return spec


def tiers_left(tiers: pl.DataFrame, dark: bool) -> dict[str, Any] | None:
    """How many of each asset are still on the board.

    A tier is a group the board cannot separate, so the useful question is not
    *who is next* but *how many more of this are there*. Reaching for the last
    player in a tier buys something; reaching for the fourth of nine does not.
    Color follows the position, never the count.
    """
    if not tiers.height:
        return None
    frame = tiers.with_columns(
        (pl.col("position") + " T" + pl.col("tier").cast(pl.Utf8)).alias("band")
    )
    tips = [
        alt.Tooltip("position:N", title="Position"),
        alt.Tooltip("tier:Q", title="Tier"),
        alt.Tooltip("n_left:Q", title="Left"),
        alt.Tooltip("best_available:N", title="Best available"),
    ]
    for col, title in (("par_top", "PAR top"), ("par_bottom", "PAR bottom")):
        if col in frame.columns:
            tips.append(alt.Tooltip(f"{col}:Q", title=title, format=".1f"))
    bars = (
        alt.Chart(alt.Data(values=frame.to_dicts()))
        # 4px rounded data-end anchored to the baseline; the bar grows from zero.
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("n_left:Q", title="Players still available"),
            y=alt.Y("band:N", title=None, sort=None),
            color=alt.Color("position:N", scale=theme.position_scale(dark), legend=None),
            tooltip=tips,
        )
        .properties(height=max(240, 24 * frame.height), width=_WIDTH)
    )
    return _spec(bars, dark)


def quality_against_price(
    scored: pl.DataFrame, position: str, dark: bool
) -> dict[str, Any] | None:
    """Quality percentile against price percentile, inside one position.

    The diagonal is agreement, and distance from it is the whole message —
    top-left is where this project disagrees with the market in your favour.
    Only players more than 25 percentile points off the line are labelled;
    labelling every point turns a scatter into a wall of names.
    """
    need = {"quality_pct", "market_pct", "value_gap", "name"}
    if not scored.height or not need.issubset(scored.columns):
        return None
    sub = scored.filter(pl.col("position") == position)
    if not sub.height:
        return None

    ink = theme.ink(dark)
    base = alt.Chart(alt.Data(values=sub.to_dicts()))
    fair = (
        alt.Chart(alt.Data(values=[{"x": 0, "y": 0}, {"x": 100, "y": 100}]))
        .mark_line(strokeDash=[4, 4], color=ink["axis"])
        .encode(x="x:Q", y="y:Q")
    )
    pts = base.mark_circle(size=150, opacity=0.85).encode(
        x=alt.X("market_pct:Q", title="Draft price percentile (within position)"),
        y=alt.Y("quality_pct:Q", title="Quality percentile (within position)"),
        # Diverging: two poles with a neutral middle, because the sign of the
        # gap is the message. Never a rainbow.
        color=alt.Color(
            "value_gap:Q",
            scale=alt.Scale(scheme="redblue", domainMid=0),
            legend=alt.Legend(title="Value gap"),
        ),
        tooltip=[
            alt.Tooltip("name:N", title="Player"),
            alt.Tooltip("quality_pct:Q", title="Quality pct", format=".0f"),
            alt.Tooltip("market_pct:Q", title="Price pct", format=".0f"),
            alt.Tooltip("value_gap:Q", title="Value gap", format=".0f"),
        ],
    )
    labels = base.mark_text(align="left", dx=9, fontSize=10, color=ink["primary"]).encode(
        x="market_pct:Q", y="quality_pct:Q", text="name:N",
        opacity=alt.condition(
            "abs(datum.value_gap) > 25", alt.value(0.9), alt.value(0.0)
        ),
    )
    return _spec(
        (fair + pts + labels).properties(height=460, width=_WIDTH), dark
    )


def player_usage(row: dict[str, Any], league: pl.DataFrame, dark: bool) -> dict[str, Any] | None:
    """One player's opportunity profile against his position's distribution.

    A share means nothing without the field it was taken from — 18% of targets
    is a WR2 role or a lead role depending on the position. So each metric is
    drawn as this player's value against his position's median, and a metric
    he was not measured on is **absent rather than zero**.
    """
    metrics = [
        ("snap_pct", "Snap share"),
        ("target_share", "Target share"),
        ("rz_target_share", "Red-zone targets"),
        ("rz_carry_share", "Red-zone carries"),
        ("exp_td_share", "Expected TD share"),
    ]
    rows: list[dict[str, Any]] = []
    for col, label in metrics:
        value = row.get(col)
        if value is None or col not in league.columns:
            continue
        peers = league.filter(
            (pl.col("position") == row.get("position")) & pl.col(col).is_not_null()
        )
        if not peers.height:
            continue
        rows.append({
            "metric": label,
            "value": float(value),
            "median": float(peers.get_column(col).median()),
        })
    if not rows:
        return None

    data = alt.Data(values=rows)
    bar = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=14, color=theme.position_colors(dark).get(
            row.get("position"), theme.position_colors(dark)["RB"]
        ))
        .encode(
            x=alt.X("value:Q", title="Share", axis=alt.Axis(format="%")),
            y=alt.Y("metric:N", title=None, sort=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Him", format=".1%"),
                alt.Tooltip("median:Q", title="Position median", format=".1%"),
            ],
        )
    )
    # The position's median as a tick, so the bar is read against the field it
    # came out of rather than against zero.
    ref = alt.Chart(data).mark_tick(
        thickness=2, size=20, color=theme.ink(dark)["primary"], opacity=0.8
    ).encode(x="median:Q", y=alt.Y("metric:N", sort=None))
    return _spec((bar + ref).properties(height=26 * len(rows) + 60, width=_WIDTH), dark)


def dropoff(players: pl.DataFrame, dark: bool) -> dict[str, Any] | None:
    """The curves behind the `drop` column: `par_env` by positional rank.

    Small multiples rather than one shared plot, because each position's
    demand line sits at a different rank and a rule only means anything drawn
    on its own panel. One series per panel, so the panel title carries
    identity and no legend is needed.
    """
    needed = {"position", "pos_rank", "par_env"}
    if not players.height or not needed.issubset(players.columns):
        return None
    keep = [c for c in ("position", "pos_rank", "par_env", "name", "in_demand") if c in players.columns]
    view = players.select(keep).drop_nulls(subset=["pos_rank", "par_env"])
    if not view.height:
        return None

    colors = theme.position_colors(dark)
    panels: list[alt.LayerChart | alt.Chart] = []
    for position, color in colors.items():
        rows = view.filter(pl.col("position") == position).sort("pos_rank")
        if not rows.height:
            continue
        base = alt.Chart(alt.Data(values=rows.to_dicts()))
        line = base.mark_line(strokeWidth=2, color=color).encode(
            x=alt.X("pos_rank:Q", title="Positional rank"),
            y=alt.Y("par_env:Q", title="PAR + environment"),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("pos_rank:Q", title="Rank"),
                alt.Tooltip("par_env:Q", title="PAR+env", format=".1f"),
            ],
        )
        # Hover targets bigger than the mark: invisible points carry the tooltip.
        hover = base.mark_point(size=90, opacity=0, color=color).encode(
            x="pos_rank:Q",
            y="par_env:Q",
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("pos_rank:Q", title="Rank"),
                alt.Tooltip("par_env:Q", title="PAR+env", format=".1f"),
            ],
        )
        layers: list[alt.Chart] = [line, hover]
        if "in_demand" in rows.columns:
            inside = rows.filter(pl.col("in_demand"))
            if inside.height:
                # The roster-demand line, drawn where this position's demand
                # runs out — the same cut the board's blank `block` column marks.
                edge = int(inside.get_column("pos_rank").max())
                layers.append(
                    alt.Chart(alt.Data(values=[{"edge": edge}]))
                    .mark_rule(strokeDash=[4, 3], color=theme.ink(dark)["axis"])
                    .encode(x="edge:Q")
                )
        panels.append(
            alt.layer(*layers).properties(width=290, height=170, title=position)
        )

    if not panels:
        return None
    rows_of_two = [alt.hconcat(*panels[i : i + 2]) for i in range(0, len(panels), 2)]
    return _spec(alt.vconcat(*rows_of_two), dark)
