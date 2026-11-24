"""Executive overview: business KPIs on the filtered bookings and model KPIs."""

import polars as pl
import streamlit as st

from hotel_dashboard import charts, config
from hotel_dashboard.data import active_filter_count, load_bookings, load_metrics
from hotel_dashboard.kpis import business_kpis, fmt_compact, fmt_number, fmt_pct
from hotel_dashboard.ui import header, plot

bookings = load_bookings()
selection: pl.DataFrame = st.session_state.get("selection", bookings)
spec = st.session_state.get("filter_spec")
metrics = load_metrics()

header(
    "Hotel Booking Demand · Lisboa y Algarve, 2015–2017",
    "Demanda, cancelaciones y tarifas",
    "Panorama ejecutivo de las reservas de dos hoteles portugueses. Los filtros de la barra "
    "lateral recalculan los indicadores y los gráficos; las métricas de los modelos se "
    "calcularon en la partición de validación.",
)

n_filters = active_filter_count(spec, bookings) if spec else 0
st.caption(
    f"Mostrando **{fmt_number(selection.height)}** de {fmt_number(bookings.height)} reservas "
    f"({fmt_pct(selection.height / bookings.height, 0)}) · "
    + (f"{n_filters} filtro(s) activo(s)" if n_filters else "sin filtros activos")
)

if selection.is_empty():
    st.warning(
        "Ningún registro cumple los filtros seleccionados. Amplíe la selección.",
        icon=":material/filter_alt_off:",
    )
    st.stop()

current, overall = business_kpis(selection), business_kpis(bookings)
filtered = selection.height < bookings.height


def delta(value: float | None, reference: float | None, kind: str) -> str | None:
    """Difference against the full dataset, only when filters are active."""
    if not filtered or value is None or reference is None:
        return None
    diff = value - reference
    if kind == "pp":
        return f"{fmt_number(diff * 100, 1)} p.p. vs. total"
    return f"{fmt_number(diff, 1)} vs. total"


st.subheader("Indicadores de negocio", anchor=False)
cols = st.columns(6)
cols[0].metric(
    "Reservas",
    fmt_number(current.bookings),
    border=True,
    help="Reservas que cumplen los filtros, incluidas las canceladas.",
)
cols[1].metric(
    "Cancelación",
    fmt_pct(current.cancel_rate),
    delta(current.cancel_rate, overall.cancel_rate, "pp"),
    delta_color="inverse",
    border=True,
    help="Proporción de reservas canceladas antes de la llegada.",
)
cols[2].metric(
    "ADR medio",
    fmt_number(current.mean_adr, 1),
    delta(current.mean_adr, overall.mean_adr, "abs"),
    border=True,
    help="Tarifa diaria media de las reservas mantenidas (u.m. por noche).",
)
cols[3].metric(
    "Ingreso",
    fmt_compact(current.revenue),
    border=True,
    help="Ingreso estimado: suma de ADR × noches de las reservas mantenidas (u.m.).",
)
cols[4].metric(
    "Ingreso perdido",
    fmt_compact(current.lost_revenue),
    border=True,
    help="ADR × noches de las reservas canceladas: valor que no llegó a materializarse.",
)
cols[5].metric(
    "Anticipación",
    f"{fmt_number(current.mean_lead_time, 0)} d",
    delta(current.mean_lead_time, overall.mean_lead_time, "abs"),
    delta_color="off",
    border=True,
    help="Media de días entre la creación de la reserva y la llegada prevista.",
)

reg = metrics["regression"]["metrics"]
cls = metrics["classification"]["metrics"]
clu = metrics["clustering"]
st.subheader("Desempeño de los modelos", anchor=False)
st.caption(
    "Validación: 9.299 reservas mantenidas (ADR) y 14.821 reservas (cancelación y segmentación). Las diferencias se miden frente al modelo de referencia."
)
cols = st.columns(5)
rf_reg, lin_reg = reg["random_forest"]["validation"], reg["linear"]["validation"]
rf_cls, log_cls = cls["random_forest"]["validation"], cls["logistic"]["validation"]
cols[0].metric(
    "R² · tarifa (ADR)",
    fmt_number(rf_reg["r2"], 3),
    f"{fmt_number(rf_reg['r2'] - lin_reg['r2'], 3)} vs. lineal",
    border=True,
    help="Random Forest. Proporción de la varianza de la tarifa explicada.",
)
cols[1].metric(
    "RMSE · tarifa (ADR)",
    fmt_number(rf_reg["rmse"], 2),
    f"{fmt_number(rf_reg['rmse'] - lin_reg['rmse'], 2)} vs. lineal",
    delta_color="inverse",
    delta_arrow="off",
    border=True,
    help="Raíz del error cuadrático medio, en u.m. por noche.",
)
cols[2].metric(
    "AUC · cancelación",
    fmt_number(rf_cls["roc_auc"], 3),
    f"{fmt_number(rf_cls['roc_auc'] - log_cls['roc_auc'], 3)} vs. logística",
    border=True,
    help="Random Forest. Capacidad de ordenar reservas por riesgo (0,5 = azar).",
)
cols[3].metric(
    "F1 · cancelación",
    fmt_number(rf_cls["f1"], 3),
    f"{fmt_number(rf_cls['f1'] - log_cls['f1'], 3)} vs. logística",
    border=True,
    help=f"Umbral 0,50. Precisión {fmt_pct(rf_cls['precision'])}, sensibilidad {fmt_pct(rf_cls['recall'])}.",
)
cols[4].metric(
    "Silueta · K-Means",
    fmt_number(clu["silhouette_validation"], 3),
    f"K = {clu['k']} grupos",
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="K-Means sobre siete variables estandarizadas. Separación moderada (−1 a 1).",
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
with st.container(border=True):
    st.markdown("**Lectura rápida**")
    lines = [
        f"- De cada 100 reservas seleccionadas se cancelan **{fmt_number((current.cancel_rate or 0) * 100, 0)}**."
    ]
    if lead["long"] is not None and lead["short"] is not None:
        lines.append(
            f"- Con más de 180 días de anticipación la cancelación llega a **{fmt_pct(lead['long'])}**, "
            f"frente a **{fmt_pct(lead['short'])}** en reservas hechas con un mes o menos."
        )
    if not segment_rates.is_empty():
        top = segment_rates.row(0, named=True)
        lines.append(
            f"- El segmento con mayor tasa es **{config.label('market_segment', top['market_segment'])}** "
            f"({fmt_pct(top['rate'])} de {fmt_number(top['n'])} reservas)."
        )
    if current.revenue:
        lines.append(
            f"- Las cancelaciones equivalen a **{fmt_pct(current.lost_revenue / (current.revenue + current.lost_revenue))}** "
            "del valor total reservado (ADR × noches)."
        )
    st.markdown("\n".join(lines))

demand, cancellations, rates, table = st.tabs(["Demanda", "Cancelaciones", "Tarifas", "Datos"])
with demand:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(charts.monthly_series(selection, "bookings"))
        st.caption(
            "El hotel urbano concentra dos tercios del volumen; ambos hoteles crecen en primavera y verano."
        )
    with right, st.container(border=True):
        plot(charts.monthly_series(selection, "rate"))
        st.caption(
            "En el conjunto completo, el hotel urbano cancela más que el vacacional en 24 de los 26 meses (42 % frente a 28 %)."
        )

with cancellations:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(charts.rate_by_lead_time(selection))
        st.caption(
            "Cuanto antes se reserva, más probable es cancelar: la anticipación es el principal factor del modelo."
        )
    with right, st.container(border=True):
        plot(charts.rate_by_category(selection, "market_segment", "Cancelación por segmento de mercado"))
        st.caption("Se omiten categorías con menos de 30 reservas en la selección.")
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(
            charts.rate_by_category(selection, "deposit_type", "Cancelación por tipo de depósito", height=260)
        )
        st.caption(
            "Las reservas no reembolsables se cancelan casi siempre, una anomalía ya señalada en la Fase 1; por eso el depósito no se usa como predictor."
        )
    with right, st.container(border=True):
        plot(
            charts.rate_by_category(selection, "customer_type", "Cancelación por tipo de cliente", height=260)
        )
        st.caption(
            "El tipo de cliente es, junto con la anticipación, la variable que más pesa en el modelo de cancelación."
        )

with rates:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(charts.adr_by_month_of_year(selection))
        st.caption(
            "En el conjunto completo, la mediana del hotel vacacional pasa de unos 50 en invierno a 180 en agosto; la del urbano oscila entre 80 y 120."
        )
    with right, st.container(border=True):
        plot(
            charts.adr_by_category(selection, "reserved_room_type", "Tarifa por tipo de habitación reservada")
        )
        st.caption("Cajas: mediana y rango intercuartílico de la ADR de reservas mantenidas.")

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
    st.caption(
        f"Primeras 1.000 filas de {fmt_number(selection.height)}. Valores en el formato original de la fuente."
    )
    st.download_button(
        "Descargar selección (CSV)",
        selection.drop("pc1", "pc2", "pc3").write_csv().encode("utf-8"),
        file_name="reservas_filtradas.csv",
        mime="text/csv",
        icon=":material/download:",
    )
