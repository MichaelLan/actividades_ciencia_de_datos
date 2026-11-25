"""Shared light visual identity: palette, semantic tones, Plotly template and page CSS."""

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#ffffff"
PAGE = "#f6f8fc"
TEXT = "#1b2333"
TEXT_SECONDARY = "#4e5668"
MUTED = "#8a91a0"
GRID = "#edf0f5"
AXIS = "#cdd3de"
BORDER = "#e3e8f0"

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BRAND = CATEGORICAL[0]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_RAMP = ["#fbd5d3", "#f6aeab", "#ef8784", "#e8605e", "#e34948", "#c43534", "#a02525"]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f1f2f4"], [1.0, "#e34948"]]
STATUS = {"good": "#1baf7a", "warning": "#eda100", "serious": "#eb6834", "critical": "#e34948"}

TONES: dict[str, dict[str, str]] = {
    "brand": {"accent": "#2a78d6", "tint": "#eaf2fc", "streamlit": "blue"},
    "risk": {"accent": "#e34948", "tint": "#fdeeee", "streamlit": "red"},
    "positive": {"accent": "#1baf7a", "tint": "#e7f7f0", "streamlit": "green"},
    "warm": {"accent": "#eda100", "tint": "#fdf5df", "streamlit": "yellow"},
    "loss": {"accent": "#eb6834", "tint": "#fdf0e9", "streamlit": "orange"},
    "violet": {"accent": "#6250d6", "tint": "#efedfb", "streamlit": "violet"},
}

HOTEL_COLORS = {"City Hotel": CATEGORICAL[0], "Resort Hotel": CATEGORICAL[1]}
OUTCOME_COLORS = {"Mantenida": CATEGORICAL[0], "Cancelada": CATEGORICAL[1]}
SEGMENT_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star"]

FONT = '"Source Sans", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'


def rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a CSS ``rgba()`` string."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def fade(hex_color: str, alpha: float = 0.2) -> dict:
    """Vertical gradient fill for area charts: transparent at the baseline, tinted at the line."""
    return {
        "type": "vertical",
        "colorscale": [[0, rgba(hex_color, 0.0)], [1, rgba(hex_color, alpha)]],
    }


def _template() -> go.layout.Template:
    """Build the single Plotly template used by every chart."""
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": AXIS,
        "showline": True,
        "ticks": "",
        "tickfont": {"color": TEXT_SECONDARY, "size": 13},
        "title": {"font": {"color": MUTED, "size": 13}, "standoff": 10},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT, "color": TEXT, "size": 14},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": CATEGORICAL,
            "margin": {"l": 8, "r": 16, "t": 56, "b": 8},
            "title": {
                "x": 0,
                "xref": "container",
                "xanchor": "left",
                "y": 1,
                "yref": "container",
                "yanchor": "top",
                "pad": {"t": 12, "l": 4},
                "font": {"size": 18, "color": TEXT, "weight": 600},
            },
            "xaxis": axis,
            "yaxis": {**axis, "showline": False, "ticklabelstandoff": 6},
            "legend": {
                "orientation": "h",
                "x": 0,
                "xref": "paper",
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
                "font": {"color": TEXT_SECONDARY, "size": 13},
                "title": {"text": ""},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "hoverlabel": {
                "bgcolor": SURFACE,
                "bordercolor": AXIS,
                "font": {"family": FONT, "color": TEXT, "size": 13},
            },
            "hovermode": "closest",
            "bargap": 0.35,
            "colorscale": {"sequential": [[i / 6, c] for i, c in enumerate(BLUE_RAMP)]},
            "separators": ",.",
        }
    )


pio.templates["hotel"] = _template()
pio.templates.default = "hotel"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "locale": "es",
}


def _tone_css() -> str:
    """Accent and tint rules for keyed KPI containers (``st-key-kpi-<tone>-…``)."""
    rules = []
    for name, tone in TONES.items():
        sel = f'[class*="st-key-kpi-{name}"]'
        rules.append(
            f"{sel} {{background: linear-gradient(160deg, {tone['tint']} 0%, {SURFACE} 62%) !important;"
            f"border-top: 4px solid {tone['accent']} !important;}}"
            f'{sel} [data-testid="stMetricLabel"] [data-testid="stIconMaterial"] {{color: {tone["accent"]};}}'
            f"{sel} .mark-area path {{fill: {rgba(tone['accent'], 0.16)} !important;}}"
            f"{sel} .mark-line path {{stroke: {tone['accent']} !important; stroke-width: 2px;}}"
            f"{sel} .mark-symbol path {{stroke: {tone['accent']} !important;}}"
        )
    return "\n".join(rules)


CSS = f"""
<style>
.stApp {{background: {PAGE};}}
.block-container {{padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1440px;}}
[class*="st-key-card"], [class*="st-key-kpi"] {{background: {SURFACE}; border-radius: 18px !important;
  border-color: {BORDER} !important; box-shadow: 0 1px 2px rgba(27,35,51,0.04), 0 4px 14px rgba(27,35,51,0.04);}}
[class*="st-key-kpi"] {{padding: 0.8rem 0.9rem 0.55rem !important;}}
[class*="st-key-kpi"] [data-testid="stMetricValue"] {{font-size: 2.1rem;}}
[data-testid="stMetricChart"] svg {{background: transparent !important;}}
[class*="st-key-kpi"] [data-testid="stMetricLabel"] p {{font-size: 0.9rem;}}
[class*="st-key-krow"] [data-testid="stHorizontalBlock"] {{flex-wrap: wrap; row-gap: 1rem;}}
}}
[data-testid="stMetricValue"] {{font-weight: 700; letter-spacing: -0.01em;}}
[data-testid="stMetricLabel"] p {{color: {TEXT_SECONDARY}; font-size: 0.95rem; font-weight: 600;}}
[data-testid="stCaptionContainer"] {{color: {MUTED};}}
[data-testid="stTab"] p {{font-size: 1.02rem; font-weight: 600;}}
[data-testid="stExpander"] details {{background: {SURFACE}; border-radius: 14px;}}
{_tone_css()}
.msr {{font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
  -webkit-font-feature-settings: "liga"; font-feature-settings: "liga"; -webkit-font-smoothing: antialiased;}}
.hero {{display: flex; align-items: center; gap: 1.1rem; padding: 1.15rem 1.5rem; margin-bottom: 1rem;
  border-radius: 20px; border: 1px solid #dfe7f5;
  background: linear-gradient(115deg, #e6f0fc 0%, #eeebfb 55%, #fdeee7 100%);}}
.hero-icon {{flex: none; width: 52px; height: 52px; border-radius: 15px; display: grid; place-items: center;
  background: {BRAND}; color: #fff; box-shadow: 0 6px 16px {rgba(BRAND, 0.28)};}}
.hero-icon .msr {{font-size: 30px;}}
.hero-text {{flex: 1; min-width: 0;}}
.hero-title {{font-size: 2.05rem; font-weight: 750; line-height: 1.15; color: {TEXT}; letter-spacing: -0.015em;}}
.hero-sub {{font-size: 1.05rem; color: {TEXT_SECONDARY}; margin-top: 0.15rem;}}
.hero-chips {{display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end;}}
.chip {{display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.75rem; border-radius: 999px;
  background: rgba(255,255,255,0.85); border: 1px solid #dfe5f0; font-size: 0.9rem; color: {TEXT_SECONDARY};
  white-space: nowrap;}}
.chip .msr {{font-size: 18px; color: {BRAND};}}
.chip b {{color: {TEXT};}}
.section {{display: flex; align-items: center; gap: 0.5rem; margin: 0.9rem 0 0.1rem; font-size: 1.2rem;
  font-weight: 700; color: {TEXT};}}
.section .msr {{font-size: 22px; color: {BRAND};}}
.section small {{font-size: 0.9rem; font-weight: 500; color: {MUTED}; margin-left: 0.25rem;}}
.tile {{border-radius: 16px; padding: 0.95rem 1.15rem; background: {SURFACE}; border: 1px solid {BORDER};
  display: flex; align-items: center; gap: 1rem; height: 100%; box-shadow: 0 1px 2px rgba(27,35,51,0.04);}}
.tile-num {{font-size: 2.2rem; font-weight: 750; line-height: 1; letter-spacing: -0.02em; white-space: nowrap;}}
.tile-label {{font-size: 0.98rem; color: {TEXT_SECONDARY}; line-height: 1.3;}}
.tile-label b {{color: {TEXT};}}
.seg-card {{border-radius: 16px; background: {SURFACE}; border: 1px solid {BORDER}; padding: 1rem 1.15rem 0.95rem;
  margin-bottom: 0.9rem; box-shadow: 0 1px 3px rgba(27,35,51,0.05); position: relative; overflow: hidden;}}
.seg-card::before {{content: ""; position: absolute; inset: 0 0 auto 0; height: 5px; background: var(--seg);}}
.seg-head {{display: flex; align-items: center; gap: 0.55rem; font-weight: 700; font-size: 1.08rem; color: {TEXT};}}
.seg-dot {{width: 12px; height: 12px; border-radius: 4px; background: var(--seg); flex: none;}}
.seg-tag {{color: {MUTED}; font-size: 0.92rem; margin: 0.1rem 0 0.7rem 1.4rem;}}
.seg-stats {{display: flex; gap: 1.3rem; align-items: flex-end; flex-wrap: wrap; row-gap: 0.5rem;}}
.seg-stat {{white-space: nowrap;}}
.seg-stat .v {{font-size: 1.75rem; font-weight: 750; line-height: 1.05; color: {TEXT}; letter-spacing: -0.01em;}}
.seg-stat .l {{font-size: 0.82rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.04em;}}
.risk-badge {{display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.1rem; border-radius: 999px;
  font-size: 1.2rem; font-weight: 700; color: {TEXT}; border: 2px solid var(--risk);
  background: color-mix(in srgb, var(--risk) 14%, white);}}
.risk-badge .msr {{font-size: 24px; color: var(--risk);}}
.seg-inline {{display: flex; align-items: center; gap: 0.7rem; padding: 0.2rem 0;}}
.seg-inline .seg-dot {{width: 18px; height: 18px; border-radius: 6px;}}
.seg-inline .name {{font-size: 1.25rem; font-weight: 700; color: {TEXT};}}
.seg-inline .tag {{color: {MUTED}; font-size: 0.95rem;}}
@media (max-width: 1599px) {{
  [class*="st-key-krow6"] [data-testid="stColumn"] {{flex: 1 1 calc(33.333% - 1rem) !important;
    min-width: calc(33.333% - 1rem) !important;}}
  [class*="st-key-krow5"] [data-testid="stColumn"] {{flex: 1 1 calc(33.333% - 1rem) !important;
    min-width: calc(33.333% - 1rem) !important;}}
}}
@media (max-width: 1400px) {{
  .tile-badge {{display: none !important;}}
  .hero-chips {{flex-direction: column; align-items: flex-end;}}
  .seg-stats {{gap: 0.8rem;}}
  .seg-stat .v {{font-size: 1.35rem;}}
  .seg-stat .l {{font-size: 0.72rem; letter-spacing: 0.02em;}}
}}
</style>
"""
