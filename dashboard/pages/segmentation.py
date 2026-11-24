"""Segmentation module: K-Means segments on a PCA map, sizes and numeric profiles."""

import polars as pl
import streamlit as st

from hotel_dashboard import charts, config, modeling, theme
from hotel_dashboard.data import load_bookings, load_metrics
from hotel_dashboard.kpis import fmt_number, fmt_pct
from hotel_dashboard.ui import get_models, header, plot, segment_chip

MAX_POINTS = 6000
MIN_POINTS_PER_SEGMENT = 150

bookings = load_bookings()
selection: pl.DataFrame = st.session_state.get("selection", bookings)
metrics = load_metrics()["clustering"]

header(
    "Aprendizaje no supervisado",
    "Perfiles de reserva",
    f"K-Means con K = {metrics['k']} agrupa las reservas según anticipación, composición de "
    "huéspedes, noches e historial (siete variables estandarizadas). Dos perfiles concentran "
    "más del 90 % de la demanda; los otros cuatro son nichos con comportamientos extremos.",
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
        info = config.SEGMENTS[key]
        row = lookup.get(key)
        color = theme.CATEGORICAL[config.SEGMENT_ORDER.index(key)]
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"{segment_chip(color)}**{info['name']}**", unsafe_allow_html=True)
            if row is None:
                st.caption("Sin reservas en la selección actual.")
                continue
            adr = fmt_number(row["adr"], 0) if row["adr"] is not None else "—"
            st.markdown(
                f"{fmt_number(row['size'])} reservas · **{fmt_pct(row['size'] / total)}**  \n"
                f"Cancelación **{fmt_pct(row['cancel_rate'])}** · ADR media **{adr}**"
            )
            st.caption(info["description"])

booking = st.session_state.get("booking")
coords = None
if booking is not None:
    booking_segment, coords = modeling.assign_segment(get_models().segmentation, booking)

map_col, side_col = st.columns([8, 4], gap="medium")
with map_col, st.container(border=True):
    controls = st.columns([3, 4], vertical_alignment="center")
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
            "Mostrar la reserva del simulador",
            value=booking is not None,
            disabled=booking is None,
            key="seg_show_booking",
        )
    sample = balanced_sample(selection)
    plot(
        charts.segment_scatter(
            sample, view == "3D", coords if show_booking else None, metrics["pca_explained_variance"]
        ),
        key="pca_map",
    )
    explained = sum(metrics["pca_explained_variance"][: 3 if view == "3D" else 2])
    st.caption(
        f"{fmt_number(sample.height)} reservas muestreadas de {fmt_number(selection.height)}; los segmentos "
        f"pequeños se sobrerrepresentan (mínimo {MIN_POINTS_PER_SEGMENT}) para que sean visibles. "
        f"Los ejes explican el {fmt_pct(explained, 0)} de la varianza: los grupos se separan mejor en "
        "las siete dimensiones originales que en esta proyección."
    )
    if booking is not None and show_booking:
        st.caption(
            f"La reserva simulada (marcador negro) pertenece a **{config.SEGMENTS[booking_segment]['name']}**."
        )

with side_col:
    with st.container(border=True):
        plot(charts.segment_sizes(selection), key="sizes")
        st.caption("Escala logarítmica: los nichos tienen cientos de reservas frente a decenas de miles.")
    with st.container(border=True):
        st.markdown("**Calidad del agrupamiento**")
        a, b = st.columns(2)
        a.metric("Silueta (entren.)", fmt_number(metrics["silhouette_train"], 3))
        b.metric("Silueta (valid.)", fmt_number(metrics["silhouette_validation"], 3))
        st.caption("Separación moderada y estable entre particiones.")

with st.container(border=True):
    plot(charts.profile_heatmap(summary), key="profile")
    st.caption(
        "El número de cada celda es la media del segmento; el color indica cuántas desviaciones "
        "estándar se aleja de la media ponderada de todos los segmentos (rojo: por encima; azul: por debajo)."
    )

with st.expander("Tabla de perfiles (media y mediana)", icon=":material/table:"):
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
    st.caption("Columnas de perfil: media / mediana sobre la selección filtrada.")
