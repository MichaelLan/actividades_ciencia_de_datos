"""Sidebar filter widgets shared by the exploratory pages."""

from functools import partial

import polars as pl
import streamlit as st

from hotel_dashboard import config
from hotel_dashboard.data import (
    FilterSpec,
    category_options,
    date_bounds,
    range_bounds,
)

CATEGORY_WIDGETS = {
    "market_segment": "Segmento de mercado",
    "customer_type": "Tipo de cliente",
    "deposit_type": "Tipo de depósito",
    "distribution_channel": "Canal de distribución",
}
RANGE_WIDGETS = {
    "lead_time": ("Anticipación (días)", 5.0, "%d"),
    "adr": ("ADR (u.m. por noche)", 5.0, "%d"),
    "total_nights": ("Noches de estancia", 1.0, "%d"),
}


def _defaults(frame: pl.DataFrame) -> dict:
    """Default widget values: everything selected."""
    values = {"f_dates": date_bounds(frame), "f_hotel": []}
    values |= {f"f_{column}": [] for column in CATEGORY_WIDGETS}
    values |= {f"f_{column}": bounds for column, bounds in range_bounds(frame).items()}
    return values


def _reset(frame: pl.DataFrame) -> None:
    """Restore every filter widget to its default value."""
    for key, value in _defaults(frame).items():
        st.session_state[key] = value


def render_filters(frame: pl.DataFrame) -> FilterSpec:
    """Draw the sidebar filters and return the resulting selection."""
    options = category_options(frame)
    bounds = range_bounds(frame)
    first, last = date_bounds(frame)
    for key, value in _defaults(frame).items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.markdown("#### :material/tune: Filtros")
        dates = st.date_input(
            "Fecha de llegada",
            min_value=first,
            max_value=last,
            format="DD/MM/YYYY",
            key="f_dates",
            persist_state="session",
        )
        hotels = st.pills(
            "Hotel",
            options["hotel"],
            selection_mode="multi",
            format_func=partial(config.label, "hotel"),
            key="f_hotel",
            persist_state="session",
        )
        categories = {"hotel": tuple(hotels)}
        for column, title in CATEGORY_WIDGETS.items():
            categories[column] = tuple(
                st.multiselect(
                    title,
                    options[column],
                    format_func=partial(config.label, column),
                    placeholder="Todos",
                    key=f"f_{column}",
                    persist_state="session",
                )
            )
        ranges = {}
        for column, (title, step, fmt) in RANGE_WIDGETS.items():
            low, high = bounds[column]
            ranges[column] = st.slider(
                title,
                min_value=low,
                max_value=high,
                step=step,
                format=fmt,
                help=f"El extremo superior ({fmt_cap(high)}) incluye también los valores mayores.",
                key=f"f_{column}",
                persist_state="session",
            )
        st.button(
            "Restablecer filtros",
            on_click=_reset,
            args=(frame,),
            icon=":material/restart_alt:",
            width="stretch",
        )

    date_range = tuple(dates) if isinstance(dates, tuple | list) and len(dates) == 2 else None
    return FilterSpec(
        date_range=date_range,
        categories=categories,
        ranges=ranges,
        range_caps={column: high for column, (_, high) in bounds.items()},
    )


def fmt_cap(value: float) -> str:
    """Integer label for a slider cap."""
    return f"{int(value)}"
