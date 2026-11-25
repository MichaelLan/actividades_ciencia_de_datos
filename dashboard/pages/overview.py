"""Executive overview: business KPIs on the filtered bookings and model KPIs."""

import polars as pl
import streamlit as st

from hotel_dashboard import charts, config
from hotel_dashboard.data import active_filter_count, load_bookings, load_metrics
from hotel_dashboard.kpis import business_kpis, fmt_compact, fmt_number, fmt_pct, monthly_kpis
from hotel_dashboard.ui import card, chip, header, kpi, plot, section, tile

bookings = load_bookings()
selection: pl.DataFrame = st.session_state.get("selection", bookings)
spec = st.session_state.get("filter_spec")
metrics = load_metrics()

n_filters = active_filter_count(spec, bookings) if spec else 0
share = selection.height / bookings.height
header(
    "Demanda, cancelaciones y tarifas",
    "Lisboa y Algarve · llegadas de julio de 2015 a agosto de 2017",
    "hotel",
    [
        chip(f"<b>{fmt_number(selection.height)}</b> reservas", "calendar_month"),
        chip(
            f"<b>{n_filters}</b> filtro{'s' if n_filters != 1 else ''} · {fmt_pct(share, 0)}"
            if n_filters
            else "Sin filtros",
            "filter_alt",
        ),
    ],
)

if selection.is_empty():
    st.warning(
        "Ningún registro cumple los filtros seleccionados. Amplíe la selección.",
        icon=":material/filter_alt_off:",
    )
    st.stop()

current, overall = business_kpis(selection), business_kpis(bookings)
monthly = monthly_kpis(selection)
filtered = selection.height < bookings.height


def delta(value: float | None, reference: float | None, kind: str) -> str | None:
    """Difference against the full dataset, only when filters are active."""
    if not filtered or value is None or reference is None:
        return None
    diff = value - reference
    if kind == "pp":
        return f"{fmt_number(diff * 100, 1)} p.p. vs. total"
    return f"{fmt_number(diff, 1)} vs. total"


cols = st.container(key="krow6").columns(6)
kpi(
    cols[0],
    "bookings",
    "Reservas",
    fmt_number(current.bookings),
    "brand",
    "event_available",
    "Reservas que cumplen los filtros, incluidas las canceladas. Línea: evolución mensual.",
    chart=monthly["bookings"],
)
kpi(
    cols[1],
    "cancel",
    "Cancelación",
    fmt_pct(current.cancel_rate),
    "risk",
    "event_busy",
    "Proporción de reservas canceladas antes de la llegada.",
    delta(current.cancel_rate, overall.cancel_rate, "pp"),
    monthly["cancel_rate"],
)
kpi(
    cols[2],
    "adr",
    "ADR medio",
    fmt_number(current.mean_adr, 1),
    "warm",
    "sell",
    "Tarifa diaria media de las reservas mantenidas (u.m. por noche).",
    delta(current.mean_adr, overall.mean_adr, "abs"),
    monthly["mean_adr"],
)
kpi(
    cols[3],
    "revenue",
    "Valor mantenido",
    fmt_compact(current.revenue),
    "positive",
    "payments",
    "Valor estimado de las reservas mantenidas: suma de ADR × noches (u.m.). No incluye extras ni descuentos.",
    chart=monthly["revenue"],
)
kpi(
    cols[4],
    "lost",
    "Valor cancelado",
    fmt_compact(current.lost_revenue),
    "loss",
    "money_off",
    "Valor estimado de las reservas canceladas: ADR × noches (u.m.). No es una pérdida realizada: "
    "no considera reventa de habitaciones ni penalizaciones.",
    chart=monthly["lost_revenue"],
)
kpi(
    cols[5],
    "lead",
    "Anticipación",
    f"{fmt_number(current.mean_lead_time, 0)} d",
    "violet",
    "schedule",
    "Media de días entre la creación de la reserva y la llegada prevista.",
    delta(current.mean_lead_time, overall.mean_lead_time, "abs"),
    monthly["mean_lead_time"],
)

segment_rates = (
    selection.group_by(pl.col("market_segment").cast(pl.String))
    .agg(pl.len().alias("n"), pl.col("is_canceled").mean().alias("rate"))
    .filter(pl.col("n") >= charts.MIN_GROUP)
    .sort("rate", descending=True)
)
lead = selection.select(
    pl.col("is_canceled").filter(pl.col("lead_time") > 180).mean().alias("long"),
    pl.col("is_canceled").filter(pl.col("lead_time") <= 30).mean().alias("short"),
).row(0, named=True)
tiles = []
if lead["long"] is not None and lead["short"] is not None:
    tiles.append(
        tile(
            fmt_pct(lead["long"], 0),
            f"cancela con <b>&gt; 180 días</b> de<br>anticipación ({fmt_pct(lead['short'], 0)} con ≤ 30)",
            "risk",
            "trending_up",
        )
    )
if not segment_rates.is_empty():
    top = segment_rates.row(0, named=True)
    tiles.append(
        tile(
            fmt_pct(top["rate"], 0),
            f"<b>{config.label('market_segment', top['market_segment'])}</b> es el segmento<br>que más cancela",
            "violet",
            "groups",
        )
    )
if current.revenue:
    tiles.append(
        tile(
            fmt_pct(current.lost_revenue / (current.revenue + current.lost_revenue), 0),
            "del valor reservado<br>corresponde a <b>reservas canceladas</b>",
            "loss",
            "money_off",
        )
    )
if tiles:
    for col, html in zip(st.columns(len(tiles)), tiles, strict=True):
        col.markdown(html, unsafe_allow_html=True)

hotel_rates = dict(
    selection.group_by(pl.col("hotel").cast(pl.String)).agg(pl.col("is_canceled").mean()).iter_rows()
)
rate_title = None
if len(hotel_rates) == 2:
    riskier = max(hotel_rates, key=hotel_rates.get)
    rate_title = f"El {charts.hotel_short(riskier)} cancela más"

demand, cancellations, rates, table = st.tabs(
    [
        ":material/show_chart: Demanda",
        ":material/event_busy: Cancelaciones",
        ":material/sell: Tarifas",
        ":material/table: Datos",
    ]
)
with demand:
    left, right = st.columns(2, gap="medium")
    with left, card("monthly-bookings"):
        plot(charts.monthly_series(selection, "bookings", "Primavera y verano concentran la demanda"))
    with right, card("monthly-rate"):
        plot(charts.monthly_series(selection, "rate", rate_title))

with cancellations:
    left, right = st.columns(2, gap="medium")
    with left, card("lead"):
        plot(charts.rate_by_lead_time(selection))
    with right, card("segment"):
        plot(charts.rate_by_category(selection, "market_segment", "{top}: el segmento que más cancela"))
    left, right = st.columns(2, gap="medium")
    with left, card("deposit"):
        plot(charts.rate_by_category(selection, "deposit_type", "Depósito: «{top}» cancela más", height=260))
    with right, card("customer"):
        plot(charts.rate_by_category(selection, "customer_type", "Cliente: «{top}» cancela más", height=260))

with rates:
    left, right = st.columns(2, gap="medium")
    with left, card("adr-month"):
        plot(charts.adr_by_month_of_year(selection))
    with right, card("adr-room"):
        plot(charts.adr_by_category(selection, "reserved_room_type", "{top}: la tarifa más alta"))

with table:
    columns = [
        "arrival_date",
        "hotel",
        "is_canceled",
        "lead_time",
        "total_nights",
        "adults",
        "children",
        "babies",
        "market_segment",
        "customer_type",
        "deposit_type",
        "reserved_room_type",
        "adr",
    ]
    preview = selection.select(columns).head(1000)
    st.dataframe(
        preview,
        hide_index=True,
        height=380,
        column_config={
            "arrival_date": st.column_config.DateColumn("Llegada", format="DD/MM/YYYY"),
            "hotel": "Hotel",
            "is_canceled": st.column_config.CheckboxColumn("Cancelada"),
            "lead_time": "Anticipación",
            "total_nights": "Noches",
            "adults": "Adultos",
            "children": st.column_config.NumberColumn("Niños", format="%d"),
            "babies": "Bebés",
            "market_segment": "Segmento",
            "customer_type": "Cliente",
            "deposit_type": "Depósito",
            "reserved_room_type": "Habitación",
            "adr": st.column_config.NumberColumn("ADR", format="%.2f"),
        },
    )
    st.caption(f"Primeras 1.000 de {fmt_number(selection.height)} filas")
    st.download_button(
        "Descargar selección (CSV)",
        selection.drop("pc1", "pc2", "pc3").write_csv().encode("utf-8"),
        file_name="reservas_filtradas.csv",
        mime="text/csv",
        icon=":material/download:",
    )

reg = metrics["regression"]["metrics"]
cls = metrics["classification"]["metrics"]
clu = metrics["clustering"]
rf_reg, lin_reg = reg["random_forest"]["validation"], reg["linear"]["validation"]
rf_cls, log_cls = cls["random_forest"]["validation"], cls["logistic"]["validation"]


def signed(value: float, decimals: int) -> str:
    """Number with an explicit sign."""
    return ("+" if value > 0 else "") + fmt_number(value, decimals)


section("Modelos", "model_training", "validación · Random Forest y K-Means")
cols = st.container(key="krow5").columns(5)
kpi(
    cols[0],
    "r2",
    "R² tarifa",
    fmt_number(rf_reg["r2"], 3),
    "warm",
    "sell",
    "Random Forest. Proporción de la varianza de la ADR explicada (9.299 reservas mantenidas).",
    f"{signed(rf_reg['r2'] - lin_reg['r2'], 3)} vs. lineal",
)
kpi(
    cols[1],
    "rmse",
    "RMSE tarifa",
    fmt_number(rf_reg["rmse"], 2),
    "warm",
    "straighten",
    "Raíz del error cuadrático medio, en u.m. por noche.",
    f"{signed(rf_reg['rmse'] - lin_reg['rmse'], 2)} vs. lineal",
)
kpi(
    cols[2],
    "auc",
    "AUC cancelación",
    fmt_number(rf_cls["roc_auc"], 3),
    "brand",
    "target",
    "Random Forest. Capacidad de ordenar reservas por riesgo (0,5 = azar; 14.821 reservas).",
    f"{signed(rf_cls['roc_auc'] - log_cls['roc_auc'], 3)} vs. logística",
)
kpi(
    cols[3],
    "f1",
    "F1 cancelación",
    fmt_number(rf_cls["f1"], 3),
    "brand",
    "balance",
    f"Umbral 0,50. Precisión {fmt_pct(rf_cls['precision'])}, sensibilidad {fmt_pct(rf_cls['recall'])}.",
    f"{signed(rf_cls['f1'] - log_cls['f1'], 3)} vs. logística",
)
kpi(
    cols[4],
    "sil",
    "Silueta K-Means",
    fmt_number(clu["silhouette_validation"], 3),
    "violet",
    "bubble_chart",
    "K-Means sobre siete variables estandarizadas. Separación moderada (−1 a 1).",
    f"K = {clu['k']} grupos",
    arrow=False,
)
