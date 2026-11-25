"""Small Streamlit helpers shared by the pages: hero, section titles, cards and KPI tiles."""

from collections.abc import Sequence
from html import escape

import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from hotel_dashboard import config, modeling, theme


@st.cache_resource(show_spinner="Cargando modelos…")
def get_models() -> modeling.Models:
    """Load the trained pipelines once per server process."""
    return modeling.load_models()


def icon(name: str) -> str:
    """Inline Material Symbols glyph for raw HTML blocks."""
    return f'<span class="msr">{name}</span>'


def chip(text: str, glyph: str | None = None) -> str:
    """Rounded pill used in the hero band (``text`` may contain trusted ``<b>`` markup)."""
    return f'<span class="chip">{icon(glyph) if glyph else ""}{text}</span>'


def header(title: str, subtitle: str, glyph: str, chips: Sequence[str] = ()) -> None:
    """Tinted hero band with icon, title, one-line subtitle and optional chips."""
    st.markdown(
        f'<div class="hero"><div class="hero-icon">{icon(glyph)}</div>'
        f'<div class="hero-text"><div class="hero-title">{escape(title)}</div>'
        f'<div class="hero-sub">{escape(subtitle)}</div></div>'
        f'<div class="hero-chips">{"".join(chips)}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, glyph: str, note: str | None = None) -> None:
    """Compact section heading with a coloured icon."""
    extra = f"<small>{escape(note)}</small>" if note else ""
    st.markdown(f'<div class="section">{icon(glyph)}{escape(title)}{extra}</div>', unsafe_allow_html=True)


def card(key: str, height: str = "content") -> DeltaGenerator:
    """White rounded container with a soft shadow."""
    return st.container(border=True, key=f"card-{key}", height=height)


def kpi(
    target: DeltaGenerator,
    key: str,
    label: str,
    value: str,
    tone: str,
    glyph: str,
    help_text: str | None = None,
    delta: str | None = None,
    chart: Sequence[float] | None = None,
    arrow: bool = True,
) -> None:
    """Metric inside a tinted card whose accent colour encodes the KPI's meaning."""
    color = theme.TONES[tone]["streamlit"]
    with target, st.container(border=True, key=f"kpi-{tone}-{key}", height="stretch"):
        st.metric(
            label,
            value,
            delta,
            delta_color=color,
            delta_arrow="auto" if arrow else "off",
            icon=f":material/{glyph}:",
            help=help_text,
            chart_data=list(chart) if chart is not None and len(chart) > 1 else None,
            chart_type="area",
        )


def tile(number: str, label: str, tone: str, glyph: str) -> str:
    """HTML for a big-number insight tile (``label`` may contain trusted ``<b>`` markup)."""
    accent, tint = theme.TONES[tone]["accent"], theme.TONES[tone]["tint"]
    badge = (
        f'<div class="tile-badge" style="flex:none;width:44px;height:44px;border-radius:13px;display:grid;place-items:center;'
        f'background:{tint};color:{accent}"><span class="msr" style="font-size:26px">{glyph}</span></div>'
    )
    return (
        f'<div class="tile">{badge}<div><div class="tile-num" style="color:{accent}">{number}</div>'
        f'<div class="tile-label">{label}</div></div></div>'
    )


def segment_color(key: str) -> str:
    """Categorical colour of a K-Means segment."""
    return theme.CATEGORICAL[config.SEGMENT_ORDER.index(key)]


def segment_card(key: str, stats: Sequence[tuple[str, str]]) -> str:
    """HTML for a segment card: name, short tagline and a few big numbers."""
    info = config.SEGMENTS[key]
    color = segment_color(key)
    cells = "".join(
        f'<div class="seg-stat"><div class="v">{value}</div><div class="l">{escape(name)}</div></div>'
        for name, value in stats
    )
    return (
        f'<div class="seg-card" style="--seg:{color}">'
        f'<div class="seg-head"><span class="seg-dot"></span>{escape(info["name"])}</div>'
        f'<div class="seg-tag">{escape(info["tagline"])}</div><div class="seg-stats">{cells}</div></div>'
    )


def plot(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the shared template and toolbar config."""
    st.plotly_chart(fig, width="stretch", theme=None, config=theme.PLOTLY_CONFIG, key=key)


def segment_chip(color: str) -> str:
    """Inline HTML colour swatch for a segment name."""
    return f'<span class="seg-dot" style="--seg:{color};display:inline-block"></span>'
