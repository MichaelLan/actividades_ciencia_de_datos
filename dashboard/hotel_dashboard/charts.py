"""Plotly figure builders. Every chart uses the shared ``hotel`` template."""

from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl

from hotel_dashboard import config, theme
from hotel_dashboard.kpis import fmt_number, fmt_pct

MONTH_ABBR = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
LEAD_BINS = [7, 30, 90, 180, 365]
LEAD_LABELS = ["0–7", "8–30", "31–90", "91–180", "181–365", "> 365"]
MIN_GROUP = 30
SHORT_LABELS = {
    "lead_time": "Anticipación<br>(días)",
    "adults": "Adultos",
    "children": "Niños",
    "babies": "Bebés",
    "total_nights": "Noches",
    "previous_cancellations": "Cancelaciones<br>previas",
    "previous_bookings_not_canceled": "Reservas previas<br>mantenidas",
}


def _finish(fig: go.Figure, title: str, height: int = 360, legend: bool = True) -> go.Figure:
    """Apply title, height and legend visibility consistently."""
    fig.update_layout(
        title={"text": title}, height=height, showlegend=legend, margin={"t": 92 if legend else 60}
    )
    return fig


def _month_ticks(months: list[date], every: int = 3) -> dict:
    """Spanish tick labels for a monthly date axis."""
    picks = [m for m in months if (m.month - 1) % every == 0] or months
    return {
        "tickmode": "array",
        "tickvals": picks,
        "ticktext": [f"{MONTH_ABBR[m.month - 1]} {m.year}" for m in picks],
    }


def _hotel_label(code: str) -> str:
    return config.label("hotel", code)


def empty_figure(message: str = "Sin datos para la selección actual", height: int = 320) -> go.Figure:
    """Placeholder figure when filters leave no rows."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"color": theme.MUTED, "size": 14},
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height)
    return fig


def hotel_short(code: str) -> str:
    """Short Spanish hotel name for titles."""
    return {"City Hotel": "hotel urbano", "Resort Hotel": "hotel vacacional"}.get(code, code)


def monthly_series(frame: pl.DataFrame, metric: str, title: str | None = None) -> go.Figure:
    """Bookings per arrival month or cancellation rate per month, one soft area per hotel."""
    if frame.is_empty():
        return empty_figure()
    monthly = (
        frame.with_columns(month=pl.col("arrival_date").dt.truncate("1mo"))
        .group_by("hotel", "month")
        .agg(pl.len().alias("bookings"), pl.col("is_canceled").mean().alias("rate"))
        .sort("month")
    )
    fig = go.Figure()
    for hotel in ["City Hotel", "Resort Hotel"]:
        part = monthly.filter(pl.col("hotel").cast(pl.String) == hotel)
        if part.is_empty():
            continue
        is_rate = metric == "rate"
        values = part["rate"] * 100 if is_rate else part["bookings"]
        fig.add_trace(
            go.Scatter(
                x=part["month"].to_list(),
                y=values.to_list(),
                name=_hotel_label(hotel),
                mode="lines",
                line={"width": 2.5, "color": theme.HOTEL_COLORS[hotel], "shape": "spline", "smoothing": 0.5},
                fill="tozeroy",
                fillgradient=theme.fade(theme.HOTEL_COLORS[hotel]),
                customdata=[f"{MONTH_ABBR[m.month - 1]} {m.year}" for m in part["month"]],
                hovertemplate="%{customdata}<br>"
                + ("%{y:.1f} % canceladas" if is_rate else "%{y:,} reservas")
                + "<extra>%{fullData.name}</extra>",
            )
        )
    months = sorted(set(monthly["month"].to_list()))
    fig.update_xaxes(
        **_month_ticks(months, 6 if len(months) > 12 else 1), title=None, tickangle=0, showgrid=False
    )
    if metric == "rate":
        fig.update_yaxes(title=None, rangemode="tozero", ticksuffix=" %")
        default = "Cancelación mensual por hotel"
    else:
        fig.update_yaxes(title=None, rangemode="tozero")
        default = "Reservas mensuales por hotel"
    fig.update_layout(hovermode="x unified")
    return _finish(fig, title or default)


def rate_by_category(frame: pl.DataFrame, column: str, title: str, height: int = 340) -> go.Figure:
    """Sorted horizontal bars of cancellation rate; the riskiest category is highlighted.

    ``title`` may contain ``{top}``, replaced by the label of the highest-rate category.
    """
    grouped = (
        frame.group_by(pl.col(column).cast(pl.String))
        .agg(pl.len().alias("n"), pl.col("is_canceled").mean().alias("rate"))
        .filter(pl.col("n") >= MIN_GROUP)
        .sort("rate")
    )
    if grouped.is_empty():
        return empty_figure(height=height)
    labels = [config.label(column, v) for v in grouped[column]]
    colors = [theme.BLUE_RAMP[1]] * (len(labels) - 1) + [theme.TONES["risk"]["accent"]]
    fig = go.Figure(
        go.Bar(
            x=(grouped["rate"] * 100).to_list(),
            y=labels,
            orientation="h",
            marker={"color": colors, "cornerradius": 5},
            text=[fmt_pct(r) for r in grouped["rate"]],
            textposition="outside",
            textfont={"color": theme.TEXT, "size": 13},
            cliponaxis=False,
            customdata=[fmt_number(n) for n in grouped["n"]],
            hovertemplate="%{y}<br>%{x:.1f} % canceladas<br>%{customdata} reservas<extra></extra>",
            width=0.6,
        )
    )
    fig.update_xaxes(
        title=None,
        ticksuffix=" %",
        showticklabels=False,
        showgrid=False,
        showline=False,
        range=[0, max(grouped["rate"].max() * 120, 10)],
    )
    fig.update_yaxes(title=None, showgrid=False)
    fig.update_layout(bargap=0.4)
    return _finish(fig, title.format(top=labels[-1]), height=height, legend=False)


def rate_by_lead_time(frame: pl.DataFrame) -> go.Figure:
    """Cancellation rate across lead-time bands (ordered columns)."""
    if frame.is_empty():
        return empty_figure()
    grouped = (
        frame.with_columns(band=pl.col("lead_time").cut(LEAD_BINS, labels=LEAD_LABELS))
        .group_by("band")
        .agg(pl.len().alias("n"), pl.col("is_canceled").mean().alias("rate"))
        .with_columns(
            order=pl.col("band")
            .cast(pl.String)
            .replace_strict({b: i for i, b in enumerate(LEAD_LABELS)}, return_dtype=pl.Int8)
        )
        .sort("order")
    )
    order = grouped["order"].to_list()
    fig = go.Figure(
        go.Bar(
            x=grouped["band"].cast(pl.String).to_list(),
            y=(grouped["rate"] * 100).to_list(),
            marker={"color": [theme.RED_RAMP[i] for i in order], "cornerradius": 6},
            text=[fmt_pct(r, 0) for r in grouped["rate"]],
            textposition="outside",
            textfont={"color": theme.TEXT, "size": 14},
            cliponaxis=False,
            customdata=[fmt_number(n) for n in grouped["n"]],
            hovertemplate="%{x} días<br>%{y:.1f} % canceladas<br>%{customdata} reservas<extra></extra>",
            width=0.62,
        )
    )
    fig.update_xaxes(title="Días de anticipación", showgrid=False, type="category")
    fig.update_yaxes(
        title=None,
        ticksuffix=" %",
        showticklabels=False,
        showgrid=False,
        range=[0, max(grouped["rate"].max() * 118, 10)],
    )
    return _finish(fig, "Más anticipación, más cancelación", legend=False)


def adr_by_month_of_year(frame: pl.DataFrame) -> go.Figure:
    """Median ADR of kept bookings by month of year, one line per hotel."""
    kept = frame.filter(pl.col("is_canceled") == 0)
    if kept.is_empty():
        return empty_figure()
    grouped = (
        kept.with_columns(m=pl.col("arrival_date").dt.month())
        .group_by("hotel", "m")
        .agg(pl.col("adr").median().alias("adr"), pl.len().alias("n"))
        .sort("m")
    )
    fig = go.Figure()
    for hotel in ["City Hotel", "Resort Hotel"]:
        part = grouped.filter(pl.col("hotel").cast(pl.String) == hotel)
        if part.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=[MONTH_ABBR[m - 1] for m in part["m"]],
                y=part["adr"].to_list(),
                name=_hotel_label(hotel),
                mode="lines+markers",
                line={"width": 2.5, "color": theme.HOTEL_COLORS[hotel], "shape": "spline", "smoothing": 0.6},
                marker={"size": 8, "line": {"width": 2, "color": theme.SURFACE}},
                fill="tozeroy",
                fillgradient=theme.fade(theme.HOTEL_COLORS[hotel]),
                hovertemplate="%{x}<br>ADR mediano %{y:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(title=None, categoryorder="array", categoryarray=MONTH_ABBR, showgrid=False)
    fig.update_yaxes(title="ADR mediano (u.m.)", rangemode="tozero")
    return _finish(fig, "La tarifa sigue a la temporada")


def adr_by_category(frame: pl.DataFrame, column: str, title: str) -> go.Figure:
    """Box plot of ADR (kept bookings) per category, ordered by median; the priciest is highlighted.

    ``title`` may contain ``{top}``, replaced by the label of the highest-median category.
    """
    kept = frame.filter(pl.col("is_canceled") == 0)
    counts = (
        kept.group_by(pl.col(column).cast(pl.String))
        .agg(pl.len().alias("n"), pl.col("adr").median().alias("med"))
        .filter(pl.col("n") >= MIN_GROUP)
        .sort("med")
    )
    if counts.is_empty():
        return empty_figure()
    fig = go.Figure()
    last = counts[column][-1]
    for value in counts[column]:
        values = kept.filter(pl.col(column).cast(pl.String) == value)["adr"]
        color = theme.TONES["warm"]["accent"] if value == last else theme.BRAND
        fig.add_trace(
            go.Box(
                x=values.to_numpy(),
                name=config.label(column, value),
                marker={"color": color, "size": 3, "opacity": 0.35},
                line={"width": 1.5, "color": color},
                fillcolor=theme.rgba(color, 0.22),
                boxpoints=False,
                orientation="h",
                hoverinfo="x",
            )
        )
    upper = float(kept["adr"].quantile(0.995) or 300)
    fig.update_xaxes(title="ADR (u.m. por noche)", range=[0, upper])
    fig.update_yaxes(showgrid=False)
    return _finish(fig, title.format(top=config.label(column, last)), height=360, legend=False)


def segment_scatter(
    sample: pl.DataFrame, three_d: bool, booking: np.ndarray | None, explained: list[float]
) -> go.Figure:
    """PCA scatter of bookings coloured and shaped by K-Means segment."""
    fig = go.Figure()
    axis_titles = [f"CP{i + 1} ({fmt_pct(v)} var.)" for i, v in enumerate(explained)]
    for index, key in enumerate(config.SEGMENT_ORDER):
        part = sample.filter(pl.col("segment") == key)
        if part.is_empty():
            continue
        name = config.SEGMENTS[key]["name"]
        hover = (
            f"<b>{name}</b><br>Anticipación %{{customdata[0]}} días<br>"
            "Noches %{customdata[1]} · Adultos %{customdata[2]}<br>"
            "Niños %{customdata[3]} · Bebés %{customdata[4]}<extra></extra>"
        )
        custom = part.select(
            "lead_time", "total_nights", "adults", pl.col("children").fill_null(0).cast(pl.Int16), "babies"
        ).to_numpy()
        common = {
            "name": name,
            "mode": "markers",
            "customdata": custom,
            "hovertemplate": hover,
        }
        color = theme.CATEGORICAL[index]
        if three_d:
            fig.add_trace(
                go.Scatter3d(
                    x=part["pc1"],
                    y=part["pc2"],
                    z=part["pc3"],
                    marker={
                        "size": 2.6,
                        "color": color,
                        "opacity": 0.7,
                        "symbol": ["circle", "square", "diamond", "circle-open", "cross", "square-open"][
                            index
                        ],
                    },
                    **common,
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=part["pc1"],
                    y=part["pc2"],
                    marker={
                        "size": 7,
                        "color": color,
                        "opacity": 0.6,
                        "symbol": theme.SEGMENT_SYMBOLS[index],
                        "line": {"width": 0.6, "color": theme.SURFACE},
                    },
                    **common,
                )
            )
    if booking is not None:
        star = {
            "name": "Reserva simulada",
            "mode": "markers",
            "hovertemplate": "<b>Reserva simulada</b><extra></extra>",
        }
        if three_d:
            fig.add_trace(
                go.Scatter3d(
                    x=[booking[0]],
                    y=[booking[1]],
                    z=[booking[2]],
                    marker={
                        "size": 12,
                        "color": theme.TEXT,
                        "symbol": "diamond",
                        "line": {"width": 2, "color": theme.SURFACE},
                    },
                    **star,
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=[booking[0]],
                    y=[booking[1]],
                    marker={
                        "size": 20,
                        "color": theme.TEXT,
                        "symbol": "star",
                        "line": {"width": 2, "color": theme.SURFACE},
                    },
                    **star,
                )
            )
    legend = {
        "itemsizing": "constant",
        "orientation": "v",
        "x": 1.02,
        "xref": "paper",
        "y": 1,
        "yanchor": "top",
        "font": {"size": 12, "color": theme.TEXT_SECONDARY},
    }
    if three_d:
        scene_axis = {
            "backgroundcolor": theme.SURFACE,
            "gridcolor": theme.GRID,
            "zeroline": False,
            "showspikes": False,
            "tickfont": {"size": 10, "color": theme.TEXT_SECONDARY},
        }
        fig.update_layout(
            scene={
                "xaxis": {**scene_axis, "title": {"text": axis_titles[0]}},
                "yaxis": {**scene_axis, "title": {"text": axis_titles[1]}},
                "zaxis": {**scene_axis, "title": {"text": axis_titles[2]}},
                "camera": {"eye": {"x": 1.7, "y": 1.6, "z": 1.0}},
                "aspectmode": "cube",
                "domain": {"x": [0, 1], "y": [0, 1]},
            },
            legend={
                "itemsizing": "constant",
                "orientation": "h",
                "x": 0,
                "y": 1.0,
                "yanchor": "bottom",
                "font": {"size": 12, "color": theme.TEXT_SECONDARY},
            },
        )
    else:
        fig.update_xaxes(title=axis_titles[0], zeroline=True, zerolinecolor=theme.GRID)
        fig.update_yaxes(title=axis_titles[1], zeroline=True, zerolinecolor=theme.GRID)
        fig.update_layout(legend=legend)
    fig = _finish(fig, "Mapa de segmentos (PCA)", height=600 if three_d else 560)
    fig.update_layout(margin={"t": 110, "r": 0, "l": 0, "b": 0} if three_d else {"t": 56, "r": 8})
    return fig


def _wrap(text: str, width: int = 16) -> str:
    """Break a long label into two lines at the space nearest the middle."""
    if len(text) <= width or " " not in text:
        return text
    cut = min((i for i, c in enumerate(text) if c == " "), key=lambda i: abs(i - len(text) / 2))
    return f"{text[:cut]}<br>{text[cut + 1 :]}"


def segment_sizes(frame: pl.DataFrame) -> go.Figure:
    """Horizontal bars with the number of bookings per segment."""
    counts = frame.group_by("segment").agg(pl.len().alias("n"))
    lookup = dict(zip(counts["segment"].cast(pl.String), counts["n"], strict=True))
    total = max(sum(lookup.values()), 1)
    keys = [k for k in reversed(config.SEGMENT_ORDER)]
    values = [lookup.get(k, 0) for k in keys]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[_wrap(config.SEGMENTS[k]["name"]) for k in keys],
            orientation="h",
            marker={
                "color": [theme.CATEGORICAL[config.SEGMENT_ORDER.index(k)] for k in keys],
                "cornerradius": 5,
            },
            text=[fmt_pct(v / total) for v in values],
            textposition="outside",
            textfont={"color": theme.TEXT_SECONDARY, "size": 12},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} reservas<extra></extra>",
            width=0.6,
        )
    )
    fig.update_xaxes(
        type="log",
        title=None,
        tickangle=0,
        showticklabels=False,
        range=[0, np.log10(max(max(values), 10)) + 0.6],
    )
    fig.update_yaxes(showgrid=False)
    fig = _finish(fig, "Dos perfiles dominan", height=330, legend=False)
    fig.update_layout(margin={"r": 48})
    return fig


def profile_heatmap(profile: pl.DataFrame) -> go.Figure:
    """Segment × feature heatmap: colour = standardized mean, text = mean value."""
    features = config.CLUSTER_FEATURES
    means = profile.select([f"{f}_mean" for f in features]).to_numpy()
    overall = profile.select(
        [(pl.col(f"{f}_mean") * pl.col("size")).sum() / pl.col("size").sum() for f in features]
    ).to_numpy()[0]
    spread = means.std(axis=0)
    spread[spread == 0] = 1
    z = np.clip((means - overall) / spread, -2.5, 2.5)
    text = [[fmt_number(v, 1) for v in row] for row in means]
    names = [config.SEGMENTS[k]["name"] for k in profile["segment"]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[SHORT_LABELS[f] for f in features],
            y=names,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale=theme.DIVERGING,
            zmid=0,
            zmin=-2.5,
            zmax=2.5,
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": "Desv. típica<br>vs. media", "font": {"size": 12, "color": theme.MUTED}},
                "thickness": 10,
                "len": 0.9,
                "tickfont": {"size": 10},
            },
            hovertemplate="%{y}<br>%{x}: media %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False, showline=False, side="bottom", tickangle=0, tickfont={"size": 11})
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _finish(fig, "Qué distingue a cada segmento", height=400, legend=False)


def importance_bars(items: list[dict], unit: str, title: str, top: int = 10) -> go.Figure:
    """Permutation importance bars with standard deviation whiskers."""
    items = sorted(items[:top], key=lambda d: d["mean"])
    ramp = theme.BLUE_RAMP[::-1]
    colors = [ramp[min(len(items) - 1 - i, len(ramp) - 1)] for i in range(len(items))]
    fig = go.Figure(
        go.Bar(
            x=[d["mean"] for d in items],
            y=[config.FEATURE_LABELS.get(d["feature"], d["feature"]) for d in items],
            orientation="h",
            error_x={
                "type": "data",
                "array": [d["std"] for d in items],
                "color": theme.AXIS,
                "thickness": 1.2,
            },
            marker={"color": colors, "cornerradius": 5},
            hovertemplate="%{y}<br>%{x:.3f} " + unit + "<extra></extra>",
            width=0.6,
        )
    )
    fig.update_xaxes(title=f"Empeoramiento al permutar ({unit})", rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return _finish(fig, title, height=380, legend=False)


def actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Scatter of observed vs predicted ADR with the identity line."""
    upper = float(np.quantile(np.concatenate([actual, predicted]), 0.995))
    fig = go.Figure(
        go.Scattergl(
            x=predicted,
            y=actual,
            mode="markers",
            marker={"size": 5, "color": theme.BRAND, "opacity": 0.3},
            hovertemplate="Predicha %{x:.1f}<br>Observada %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, upper],
            y=[0, upper],
            mode="lines",
            name="Estimación exacta",
            line={"color": theme.TONES["loss"]["accent"], "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="ADR predicha (u.m.)", range=[0, upper])
    fig.update_yaxes(title="ADR observada (u.m.)", range=[0, upper])
    return _finish(fig, "Las tarifas altas se subestiman", height=400)


def residual_histogram(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Distribution of residuals (observed minus predicted)."""
    residual = actual - predicted
    low, high = np.quantile(residual, [0.005, 0.995])
    fig = go.Figure(
        go.Histogram(
            x=residual[(residual >= low) & (residual <= high)],
            nbinsx=60,
            marker={"color": theme.BLUE_RAMP[2], "line": {"width": 1, "color": theme.SURFACE}},
            hovertemplate="Residuo %{x}<br>%{y:,} reservas<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": theme.TONES["loss"]["accent"], "width": 2})
    fig.update_xaxes(title="Residuo: observada − predicha (u.m.)")
    fig.update_yaxes(title=None)
    return _finish(fig, "Errores centrados en cero", height=400, legend=False)


def confusion_heatmap(matrix: list[list[int]]) -> go.Figure:
    """2×2 confusion matrix with counts and row shares."""
    cm = np.array(matrix)
    shares = cm / cm.sum(axis=1, keepdims=True)
    text = [[f"{fmt_number(cm[i, j])}<br>{fmt_pct(shares[i, j])}" for j in range(2)] for i in range(2)]
    signed = np.where(np.eye(2, dtype=bool), shares, -shares)
    fig = go.Figure(
        go.Heatmap(
            z=signed,
            x=["Predicha: mantenida", "Predicha: cancelada"],
            y=["Real: mantenida", "Real: cancelada"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale=[[0, theme.RED_RAMP[4]], [0.5, "#f7f8fa"], [1, theme.BLUE_RAMP[4]]],
            showscale=False,
            xgap=4,
            ygap=4,
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False, showline=False, side="top")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(margin={"t": 80})
    return _finish(fig, "Matriz de confusión", height=360, legend=False)


def roc_curves(roc: dict, auc: dict) -> go.Figure:
    """ROC curves of the cancellation models against the chance diagonal."""
    names = {"random_forest": "Random Forest", "logistic": "Regresión logística"}
    fig = go.Figure()
    for index, (key, label) in enumerate(names.items()):
        fig.add_trace(
            go.Scatter(
                x=roc[key]["fpr"],
                y=roc[key]["tpr"],
                mode="lines",
                name=f"{label} (AUC {fmt_number(auc[key], 3)})",
                line={"width": 2.5, "color": theme.CATEGORICAL[index]},
                fill="tozeroy" if index == 0 else None,
                fillcolor=theme.rgba(theme.CATEGORICAL[index], 0.08),
                hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>" + label + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar (AUC 0,5)",
            line={"width": 1.2, "color": theme.MUTED},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title="Sensibilidad", range=[0, 1.02])
    return _finish(fig, "Curva ROC", height=400)


def k_selection(curve: list[dict], metric: str, chosen: int) -> go.Figure:
    """Elbow (inertia) or silhouette across K with the chosen K highlighted."""
    ks = [d["k"] for d in curve]
    values = [d[metric] for d in curve]
    fig = go.Figure(
        go.Scatter(
            x=ks,
            y=values,
            mode="lines+markers",
            line={"width": 2.5, "color": theme.BRAND},
            marker={"size": 8, "color": theme.BRAND, "line": {"width": 2, "color": theme.SURFACE}},
            fill="tozeroy",
            fillgradient=theme.fade(theme.BRAND, 0.16),
            hovertemplate="K=%{x}<br>%{y:,.3f}<extra></extra>",
            showlegend=False,
        )
    )
    chosen_value = values[ks.index(chosen)]
    fig.add_trace(
        go.Scatter(
            x=[chosen],
            y=[chosen_value],
            mode="markers+text",
            text=[f"K = {chosen}"],
            textposition="top center",
            textfont={"color": theme.TEXT},
            marker={
                "size": 15,
                "color": theme.CATEGORICAL[1],
                "line": {"width": 2.5, "color": theme.SURFACE},
            },
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        )
    )
    title = "Método del codo (inercia)" if metric == "inertia" else "Coeficiente de silueta"
    fig.update_xaxes(title="Número de grupos (K)", dtick=1)
    fig.update_yaxes(
        title=None,
        tickformat=",.0f" if metric == "inertia" else None,
        rangemode="tozero",
    )
    return _finish(fig, title, height=340, legend=False)


RISK_BANDS = (0.3, 0.5)


def risk_level(probability: float, threshold: float) -> tuple[str, str, str]:
    """Risk label, accent colour and icon name for a cancellation probability."""
    if probability >= threshold:
        return "Riesgo alto", theme.STATUS["critical"], "warning"
    if probability >= RISK_BANDS[0]:
        return "Riesgo moderado", theme.STATUS["warning"], "error"
    return "Riesgo bajo", theme.STATUS["good"], "check_circle"


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    """Gauge of cancellation probability with low / moderate / high bands and the threshold."""
    _, color, _ = risk_level(probability, threshold)
    low = RISK_BANDS[0] * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": " %",
                "valueformat": ".1f",
                "font": {"size": 48, "color": theme.TEXT, "weight": 700},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)",
                    "tickvals": [0, low, threshold * 100, 100],
                    "tickfont": {"size": 12, "color": theme.TEXT_SECONDARY},
                },
                "bar": {"color": color, "thickness": 0.42},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, low], "color": theme.rgba(theme.STATUS["good"], 0.18)},
                    {"range": [low, threshold * 100], "color": theme.rgba(theme.STATUS["warning"], 0.2)},
                    {"range": [threshold * 100, 100], "color": theme.rgba(theme.STATUS["critical"], 0.16)},
                ],
                "threshold": {
                    "line": {"color": theme.TEXT, "width": 3},
                    "thickness": 0.95,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=260, margin={"l": 36, "r": 36, "t": 28, "b": 8})
    return fig


def model_comparison(
    values: dict[str, float], best: str, decimals: int, title: str, height: int = 380
) -> go.Figure:
    """Horizontal bars comparing one metric across models, with the deployed model highlighted."""
    names = list(values)
    colors = [theme.BRAND if name == best else "#d5dbe5" for name in names]
    fig = go.Figure(
        go.Bar(
            x=list(values.values()),
            y=names,
            orientation="h",
            marker={"color": colors, "cornerradius": 6},
            text=[fmt_number(v, decimals) for v in values.values()],
            textposition="outside",
            textfont={"color": theme.TEXT, "size": 15},
            cliponaxis=False,
            hovertemplate="%{y}<br>%{text}<extra></extra>",
            width=0.4,
        )
    )
    fig.update_xaxes(
        showticklabels=False, showgrid=False, showline=False, range=[0, max(values.values()) * 1.22]
    )
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont={"size": 14, "color": theme.TEXT})
    return _finish(fig, title, height=height, legend=False)


def adr_context(observed: pl.Series, estimate: float, low: float, high: float, title: str) -> go.Figure:
    """Histogram of comparable observed ADR with the model estimate overlaid."""
    if observed.is_empty():
        return empty_figure(height=260)
    upper = float(max(observed.quantile(0.99) or estimate, high) * 1.05)
    fig = go.Figure(
        go.Histogram(
            x=observed.to_numpy(),
            xbins={"start": 0, "end": upper, "size": 5},
            marker={"color": theme.BLUE_RAMP[1], "line": {"width": 1, "color": theme.SURFACE}},
            hovertemplate="ADR %{x}<br>%{y:,} reservas<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vrect(x0=low, x1=high, fillcolor=theme.TONES["warm"]["accent"], opacity=0.14, line_width=0)
    fig.add_vline(x=estimate, line={"color": theme.TONES["loss"]["accent"], "width": 3})
    fig.add_annotation(
        x=estimate,
        y=1,
        yref="paper",
        text=f"<b>{fmt_number(estimate, 1)}</b>",
        showarrow=False,
        yanchor="bottom",
        font={"color": theme.TEXT, "size": 14},
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_xaxes(title="ADR (u.m. por noche)", range=[0, upper])
    fig.update_yaxes(title=None)
    return _finish(fig, title, height=300, legend=False)
