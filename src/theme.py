"""Chart colors and shared Altair chrome.

The position palette was not chosen by taste. Four positions means four series
on screen at once, often in forms where any pair can end up adjacent (scatter,
small multiples), so it has to clear the all-pairs colorblind and normal-vision
separation floors in *both* light and dark mode. Enumerating every 4-hue subset
of the eight-slot categorical palette leaves exactly two that do; this is one of
them. Substituting a hue here without re-running that check will quietly produce
a chart two of your leaguemates can't read.

Color follows the position, never its rank — RB is green whether it's first or
fourth in the legend, so filtering a chart never repaints the survivors.
"""

from __future__ import annotations

import altair as alt

# Slot assignments are fixed. Do not reorder to "match the legend".
POSITION_COLORS_LIGHT: dict[str, str] = {
    "QB": "#2a78d6",  # blue
    "RB": "#008300",  # green
    "WR": "#e87ba4",  # magenta
    "TE": "#eda100",  # yellow
}

POSITION_COLORS_DARK: dict[str, str] = {
    "QB": "#3987e5",
    "RB": "#008300",  # mode-invariant by design
    "WR": "#d55181",
    "TE": "#c98500",
}

# Sequential ramp for continuous magnitude (single hue, light -> dark). Never a
# rainbow — hue carries identity, lightness carries amount.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

# Ordinal steps for discrete ordered marks — seasons on a scarcity curve. These
# are truncated ranges of the same ramp, not the whole thing: the step nearest
# the surface still has to be visible against it. On light, nothing lighter than
# step 250; on dark, nothing darker than step 600. Using the full ramp puts one
# end of the series a hair away from the background.
ORDINAL_BLUE_LIGHT = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_BLUE_DARK = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]


def ordinal_steps(n: int, dark: bool | None = None) -> list[str]:
    """`n` evenly spaced steps of the ordinal ramp, darkest (or lightest) last.

    Spread across the whole legal range rather than taking the last n, so three
    seasons are three clearly different blues instead of three neighbours.
    """
    dark = is_dark() if dark is None else dark
    ramp = ORDINAL_BLUE_DARK if dark else ORDINAL_BLUE_LIGHT
    if n <= 1:
        return [ramp[-1]]
    if n >= len(ramp):
        return ramp[:n]
    step = (len(ramp) - 1) / (n - 1)
    return [ramp[round(i * step)] for i in range(n)]

INK_LIGHT = {"primary": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
INK_DARK = {"primary": "#ffffff", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}

# Streamlit's own chart surfaces, not the reference palette's. Used as the stroke
# color that separates stacked segments, so the gap reads as background rather
# than as a gray outline.
SURFACE_LIGHT = "#ffffff"
SURFACE_DARK = "#0e1117"


def is_dark() -> bool:
    """Whether Streamlit is actually rendering dark. Defaults to light.

    `st.context.theme.type`, not `st.get_option("theme.base")`. The config option
    reports what was *configured*, which is usually nothing; the context reports
    what the browser is *rendering*, which is what the chart has to sit on. With
    the config option an OS-dark viewer gets light-mode gridlines — bright white
    hairlines on a near-black surface, louder than the data.
    """
    try:
        import streamlit as st

        return str(getattr(st.context.theme, "type", "light")).lower() == "dark"
    except Exception:  # noqa: BLE001 — outside Streamlit, or no active session
        return False


def position_colors(dark: bool | None = None) -> dict[str, str]:
    dark = is_dark() if dark is None else dark
    return POSITION_COLORS_DARK if dark else POSITION_COLORS_LIGHT


def position_scale(dark: bool | None = None) -> alt.Scale:
    """A fixed domain/range scale so a position keeps its color across charts."""
    colors = position_colors(dark)
    return alt.Scale(domain=list(colors), range=list(colors.values()))


def ink(dark: bool | None = None) -> dict[str, str]:
    dark = is_dark() if dark is None else dark
    return INK_DARK if dark else INK_LIGHT


def surface(dark: bool | None = None) -> str:
    dark = is_dark() if dark is None else dark
    return SURFACE_DARK if dark else SURFACE_LIGHT


def base_chart(chart: alt.Chart, dark: bool | None = None) -> alt.Chart:
    """Recessive grid and axes, so the data is the loudest thing on screen."""
    c = ink(dark)
    return (
        chart.configure_axis(
            grid=True,
            gridColor=c["grid"],
            gridWidth=1,
            domainColor=c["axis"],
            tickColor=c["axis"],
            labelColor=c["muted"],
            titleColor=c["muted"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=c["muted"],
            titleColor=c["muted"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            # Filled circles with a visible stroke. Zeroing symbolStrokeWidth
            # renders nothing at all for line marks, which leaves the legend as
            # bare text and makes color the only carrier of identity on the
            # chart itself — the exact thing a legend exists to prevent.
            symbolType="circle",
            symbolFillColor=None,
            symbolStrokeWidth=3,
            symbolSize=90,
        )
        .configure_view(strokeWidth=0)
        .configure_title(color=c["primary"], fontSize=13, fontWeight=600, anchor="start")
    )
