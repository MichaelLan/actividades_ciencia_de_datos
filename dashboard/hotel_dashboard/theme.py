"""Shared light visual identity: palette, Plotly template and page CSS."""

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#ffffff"
TEXT = "#1f1e1c"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#ebeae4"
AXIS = "#c3c2b7"

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
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

HOTEL_COLORS = {"City Hotel": CATEGORICAL[0], "Resort Hotel": CATEGORICAL[1]}
OUTCOME_COLORS = {"Mantenida": CATEGORICAL[0], "Cancelada": CATEGORICAL[1]}
SEGMENT_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star"]

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif'


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
        "tickfont": {"color": TEXT_SECONDARY, "size": 12},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 13}, "standoff": 10},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT, "color": TEXT, "size": 13},
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "colorway": CATEGORICAL,
            "margin": {"l": 8, "r": 16, "t": 56, "b": 8},
            "title": {
                "x": 0,
                "xref": "container",
                "xanchor": "left",
                "y": 1,
                "yref": "container",
                "yanchor": "top",
                "pad": {"t": 14, "l": 6},
                "font": {"size": 15, "color": TEXT},
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
                "font": {"color": TEXT_SECONDARY, "size": 12},
                "title": {"text": ""},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "hoverlabel": {
                "bgcolor": SURFACE,
                "bordercolor": AXIS,
                "font": {"family": FONT, "color": TEXT, "size": 12},
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

CSS = """
<style>
.block-container {padding-top: 3rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetricValue"] {font-weight: 600;}
[data-testid="stMetricLabel"] p {color: #52514e; font-size: 0.86rem;}
[data-testid="stCaptionContainer"] {color: #6b6a65;}
.segment-chip {display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:middle;}
.eyebrow {color:#6b6a65;font-size:0.8rem;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.1rem;}
.risk-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.85rem;font-weight:600;}
</style>
"""
