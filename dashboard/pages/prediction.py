"""Live prediction: cancellation probability, expected ADR and segment for a new booking."""

from functools import partial

import polars as pl
import streamlit as st

from hotel_dashboard import charts, config, modeling
from hotel_dashboard.data import load_bookings, load_metrics
from hotel_dashboard.kpis import fmt_number, fmt_pct
from hotel_dashboard.ui import card, chip, get_models, header, icon, plot, segment_chip, segment_color

DEFAULTS = {
    "hotel": "City Hotel",
    "arrival_date_month": "August",
    "lead_time": 70,
    "total_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "customer_type": "Transient",
    "reserved_room_type": "A",
    "is_repeated_guest": False,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
}
OPTIONS = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": config.MONTHS,
    "meal": ["BB", "HB", "FB", "SC"],
    "market_segment": [
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Direct",
        "Corporate",
        "Complementary",
        "Aviation",
    ],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H"],
}
NUMERIC_LIMITS = {
    "lead_time": (0, 700),
    "total_nights": (1, 30),
    "adults": (1, 4),
    "children": (0, 3),
    "babies": (0, 2),
    "previous_cancellations": (0, 30),
    "previous_bookings_not_canceled": (0, 80),
}

for name, value in DEFAULTS.items():
    st.session_state.setdefault(f"p_{name}", value)


def select(name: str) -> str:
    """Selectbox with Spanish labels for a categorical predictor."""
    return st.selectbox(
        config.FEATURE_LABELS[name],
        OPTIONS[name],
        format_func=partial(config.label, name),
        key=f"p_{name}",
        persist_state="session",
    )


def number(name: str, help_text: str | None = None) -> int:
    """Integer input bounded to plausible values."""
    low, high = NUMERIC_LIMITS[name]
    return int(
        st.number_input(
            config.FEATURE_LABELS[name],
            min_value=low,
            max_value=high,
            step=1,
            key=f"p_{name}",
            persist_state="session",
            help=help_text,
        )
    )


def reset() -> None:
    """Restore the default booking."""
    for name, value in DEFAULTS.items():
        st.session_state[f"p_{name}"] = value


header(
    "Predicción en vivo",
    "Ajuste la reserva: los modelos responden al instante",
    "online_prediction",
    [chip("Random Forest", "forest"), chip("Solo datos disponibles al reservar", "lock_clock")],
)

form_col, result_col = st.columns([5, 7], gap="large")

with form_col, card("form"):
    st.markdown("**:material/hotel: Estancia**")
    a, b = st.columns(2)
    with a:
        select("hotel")
        number("lead_time", "Días entre la creación de la reserva y la llegada.")
    with b:
        select("arrival_date_month")
        number("total_nights")
    st.markdown("**:material/group: Huéspedes**")
    a, b, c = st.columns(3)
    with a:
        number("adults")
    with b:
        number("children")
    with c:
        number("babies")
    st.markdown("**:material/storefront: Canal y producto**")
    a, b = st.columns(2)
    with a:
        select("market_segment")
        select("customer_type")
        select("reserved_room_type")
    with b:
        select("distribution_channel")
        select("meal")
    st.markdown("**:material/history: Historial del cliente**")
    a, b = st.columns(2)
    with a:
        number("previous_cancellations")
    with b:
        number("previous_bookings_not_canceled")
    st.toggle("Cliente repetido", key="p_is_repeated_guest", persist_state="session")
    st.button(
        "Restablecer reserva de ejemplo", on_click=reset, icon=":material/restart_alt:", width="stretch"
    )

booking = {name: st.session_state[f"p_{name}"] for name in DEFAULTS}
booking["is_repeated_guest"] = int(booking["is_repeated_guest"])
st.session_state["booking"] = booking

models = get_models()
metrics = load_metrics()
probability = modeling.predict_cancellation(models.cancel, booking)
adr, adr_low, adr_high = modeling.predict_adr(models.adr, booking)
segment, _ = modeling.assign_segment(models.segmentation, booking)

rf = metrics["classification"]["metrics"]["random_forest"]["validation"]
prevalence = metrics["classification"]["validation_prevalence"]
threshold = config.THRESHOLD

level, color, glyph = charts.risk_level(probability, threshold)
ratio = probability / prevalence
alert = probability >= threshold
mae = metrics["regression"]["metrics"]["random_forest"]["validation"]["mae"]

with result_col:
    with card("risk"):
        st.markdown("**:material/event_busy: Probabilidad de cancelación**")
        gauge_col, text_col = st.columns([7, 5], vertical_alignment="center")
        with gauge_col:
            plot(charts.probability_gauge(probability, threshold), key="gauge")
        with text_col:
            st.markdown(
                f'<span class="risk-badge" style="--risk:{color}">{icon(glyph)}{level}</span>',
                unsafe_allow_html=True,
            )
            st.metric(
                "Frente a la media",
                f"{fmt_number(ratio, 1)}×",
                help=f"Probabilidad estimada dividida por la tasa de cancelación en validación ({fmt_pct(prevalence)}).",
            )
            st.metric(
                "Decisión con umbral 0,50",
                "Alerta" if alert else "Sin alerta",
                help=(
                    f"La reserva se marca como alerta si la probabilidad supera {fmt_number(threshold, 2)} "
                    f"(línea negra). En validación, el {fmt_pct(rf['precision'], 0)} de las alertas fueron "
                    f"cancelaciones reales y se detectó el {fmt_pct(rf['recall'], 0)} de ellas. "
                    "Bandas: bajo < 30 %, moderado 30–50 %, alto ≥ 50 %."
                ),
            )

    with card("adr"):
        st.markdown("**:material/sell: Tarifa diaria esperada (ADR)**")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "ADR estimada",
            f"{fmt_number(adr, 1)} u.m.",
            help=f"Media de los árboles del Random Forest (reservas mantenidas). Error típico en validación: MAE {fmt_number(mae, 1)} u.m.",
        )
        m2.metric(
            "Rango entre árboles",
            f"{fmt_number(adr_low, 0)} – {fmt_number(adr_high, 0)}",
            help="Percentiles 10 y 90 de las predicciones de los árboles; no es un intervalo de confianza formal.",
        )
        m3.metric(
            "Valor de la estancia",
            f"{fmt_number(adr * booking['total_nights'], 0)} u.m.",
            help="ADR estimada × noches, si la reserva no se cancela.",
        )
        bookings = load_bookings()
        kept = bookings.filter(
            (pl.col("is_canceled") == 0)
            & (pl.col("hotel").cast(pl.String) == booking["hotel"])
            & (pl.col("arrival_date_month").cast(pl.String) == booking["arrival_date_month"])
        )
        same_room = kept.filter(pl.col("reserved_room_type").cast(pl.String) == booking["reserved_room_type"])
        comparable = same_room if same_room.height >= 30 else kept
        scope = "mismo hotel, mes y habitación" if comparable is same_room else "mismo hotel y mes"
        plot(
            charts.adr_context(
                comparable["adr"], adr, adr_low, adr_high, f"La estimación frente a reservas reales · {scope}"
            ),
            key="adr_context",
        )

    with card("segment"):
        info = config.SEGMENTS[segment]
        seg_col, link_col = st.columns([3, 2], vertical_alignment="center")
        seg_col.markdown(
            f'<div class="seg-inline">{segment_chip(segment_color(segment))}'
            f'<div><div class="name">{info["name"]}</div><div class="tag">{info["tagline"]}</div></div></div>',
            unsafe_allow_html=True,
        )
        link_col.page_link(
            "pages/segmentation.py",
            label="Ver en el mapa de segmentos",
            icon=":material/bubble_chart:",
        )
