"""Segmentation module: K-Means segments on a PCA map, sizes and numeric profiles."""

import polars as pl
import streamlit as st

from hotel_dashboard import charts, config, modeling
from hotel_dashboard.data import load_bookings, load_metrics
from hotel_dashboard.kpis import fmt_number, fmt_pct
from hotel_dashboard.ui import (
    card,
    chip,
    get_models,
    header,
    kpi,
    plot,
    segment_card,
    segment_chip,
    segment_color,
)

MAX_POINTS = 6000
MIN_POINTS_PER_SEGMENT = 150

bookings = load_bookings()
selection: pl.DataFrame = st.session_state.get("selection", bookings)
metrics = load_metrics()["clustering"]

header(
    "Perfiles de reserva",
    "Seis segmentos según anticipación, huéspedes, noches e historial",
    "bubble_chart",
    [
        chip(f"K-Means · <b>K = {metrics['k']}</b>", "hub"),
        chip(f"<b>{fmt_number(selection.height)}</b> reservas", "calendar_month"),
    ],
)

if selection.is_empty():
    st.warning(
        "Ningún registro cumple los filtros seleccionados. Amplíe la selección.",
        icon=":material/filter_alt_off:",
    )
    st.stop()


def segment_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Size, outcomes and numeric means/medians per segment for the selection."""
    return (
        frame.group_by("segment")
        .agg(
            pl.len().alias("size"),
            pl.col("is_canceled").mean().alias("cancel_rate"),
            pl.col("adr").filter(pl.col("is_canceled") == 0).mean().alias("adr"),
            *[pl.col(f).mean().alias(f"{f}_mean") for f in config.CLUSTER_FEATURES],
            *[pl.col(f).median().alias(f"{f}_median") for f in config.CLUSTER_FEATURES],
        )
        .sort("segment")
    )


def balanced_sample(frame: pl.DataFrame) -> pl.DataFrame:
    """Proportional sample that keeps small segments visible."""
    if frame.height <= MAX_POINTS:
        return frame
    parts = []
    for key in config.SEGMENT_ORDER:
        part = frame.filter(pl.col("segment") == key)
        quota = max(MIN_POINTS_PER_SEGMENT, round(MAX_POINTS * part.height / frame.height))
        parts.append(part.sample(n=min(part.height, quota), seed=config.SEED))
    return pl.concat(parts)


summary = segment_summary(selection)
total = summary["size"].sum()
lookup = {row["segment"]: row for row in summary.iter_rows(named=True)}

for start in (0, 3):
    cols = st.columns(3)
    for col, key in zip(cols, config.SEGMENT_ORDER[start : start + 3], strict=True):
        row = lookup.get(key)
        if row is None:
            stats = [("Reservas", "—")]
        else:
            adr = fmt_number(row["adr"], 0) if row["adr"] is not None else "—"
            stats = [
                ("Volumen", fmt_pct(row["size"] / total, 1)),
                ("Cancelación", fmt_pct(row["cancel_rate"], 0)),
                ("ADR", adr),
            ]
        col.markdown(segment_card(key, stats), unsafe_allow_html=True)

booking = st.session_state.get("booking")
coords = None
if booking is not None:
    booking_segment, coords = modeling.assign_segment(get_models().segmentation, booking)

map_col, side_col = st.columns([7, 4], gap="medium")
with map_col, card("map"):
    controls = st.columns([2, 5, 1], vertical_alignment="center")
    with controls[0]:
        view = st.segmented_control(
            "Vista",
            ["2D", "3D"],
            default="2D",
            key="seg_view",
            label_visibility="collapsed",
            persist_state="session",
        )
    with controls[1]:
        show_booking = st.toggle(
            "Reserva del simulador",
            value=booking is not None,
            disabled=booking is None,
            key="seg_show_booking",
            help="Marca negra: la reserva configurada en la página Predicción.",
        )
    sample = balanced_sample(selection)
    explained = sum(metrics["pca_explained_variance"][: 3 if view == "3D" else 2])
    with controls[2], st.popover("", icon=":material/info:"):
        st.markdown(
            f"{fmt_number(sample.height)} reservas muestreadas de {fmt_number(selection.height)}; los "
            f"segmentos pequeños se sobrerrepresentan (mínimo {MIN_POINTS_PER_SEGMENT}). Los ejes explican "
            f"el {fmt_pct(explained, 0)} de la varianza: los grupos se separan mejor en las siete "
            "dimensiones originales que en esta proyección."
        )
    plot(
        charts.segment_scatter(
            sample, view == "3D", coords if show_booking else None, metrics["pca_explained_variance"]
        ),
        key="pca_map",
    )
    if booking is not None and show_booking:
        st.markdown(
            f"{segment_chip(segment_color(booking_segment))} Reserva simulada → "
            f"**{config.SEGMENTS[booking_segment]['name']}**",
            unsafe_allow_html=True,
        )

with side_col:
    with card("sizes"):
        plot(charts.segment_sizes(selection), key="sizes")
    kpi(
        st.container(),
        "sil",
        "Silueta (validación)",
        fmt_number(metrics["silhouette_validation"], 3),
        "violet",
        "hub",
        "Coeficiente de silueta (−1 a 1) en muestras de 2.500 reservas: separación moderada y estable.",
        f"entrenamiento {fmt_number(metrics['silhouette_train'], 3)}",
        arrow=False,
    )

with card("profile"):
    plot(charts.profile_heatmap(summary), key="profile")

with st.expander("Tabla de perfiles y notas", icon=":material/table:"):
    table = summary.select(
        pl.col("segment")
        .cast(pl.String)
        .replace_strict({k: v["name"] for k, v in config.SEGMENTS.items()})
        .alias("Segmento"),
        pl.col("size").alias("Reservas"),
        (pl.col("cancel_rate") * 100).round(1).alias("Cancelación (%)"),
        pl.col("adr").round(1).alias("ADR media"),
        *[
            pl.format("{} / {}", pl.col(f"{f}_mean").round(1), pl.col(f"{f}_median")).alias(
                config.FEATURE_LABELS[f]
            )
            for f in config.CLUSTER_FEATURES
        ],
    )
    st.dataframe(table, hide_index=True)
    st.caption(
        "Columnas de perfil: media / mediana sobre la selección filtrada. En el mapa de calor, el número es "
        "la media del segmento y el color, cuántas desviaciones típicas se aleja de la media ponderada "
        "(rojo: por encima; azul: por debajo). Tamaños en escala logarítmica."
    )
    for key in config.SEGMENT_ORDER:
        info = config.SEGMENTS[key]
        st.markdown(
            f"{segment_chip(segment_color(key))} **{info['name']}** · {info['description']}",
            unsafe_allow_html=True,
        )
