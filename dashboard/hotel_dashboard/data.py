"""Loading of the compact booking table and pure filtering logic."""

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl
import streamlit as st

from hotel_dashboard import config

FILTER_CATEGORIES = [
    "hotel",
    "market_segment",
    "customer_type",
    "deposit_type",
    "distribution_channel",
]
RANGE_COLUMNS = ["lead_time", "adr", "total_nights"]
RANGE_QUANTILE = 0.995


@st.cache_data(show_spinner=False)
def load_bookings(path: Path = config.DATA_PATH) -> pl.DataFrame:
    """Read the compact booking table shipped with the dashboard."""
    return pl.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: Path = config.METRICS_PATH) -> dict:
    """Read the metrics written by the training script."""
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_validation_predictions(path: Path = config.PREDICTIONS_PATH) -> pl.DataFrame:
    """Read validation-set predictions for diagnostic charts."""
    return pl.read_parquet(path)


@dataclass(frozen=True)
class FilterSpec:
    """User selection; ``None`` or empty tuple means no restriction."""

    date_range: tuple[date, date] | None = None
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    range_caps: dict[str, float] = field(default_factory=dict)


def category_options(frame: pl.DataFrame) -> dict[str, list[str]]:
    """Filterable category values ordered by frequency."""
    return {
        column: frame[column].cast(pl.String).value_counts(sort=True)[column].to_list()
        for column in FILTER_CATEGORIES
    }


def range_bounds(frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Slider bounds; the upper end is a high quantile so outliers do not flatten it."""
    bounds = {}
    for column in RANGE_COLUMNS:
        upper = float(frame[column].quantile(RANGE_QUANTILE))
        step = 10 if upper > 100 else 1
        bounds[column] = (0.0, float(math.ceil(upper / step) * step))
    return bounds


def date_bounds(frame: pl.DataFrame) -> tuple[date, date]:
    """First and last arrival dates in the data."""
    return frame["arrival_date"].min(), frame["arrival_date"].max()


def apply_filters(frame: pl.DataFrame, spec: FilterSpec) -> pl.DataFrame:
    """Return the bookings matching every active filter in ``spec``."""
    conditions = []
    if spec.date_range is not None:
        start, end = spec.date_range
        conditions.append(pl.col("arrival_date").is_between(start, end))
    for column, values in spec.categories.items():
        if values:
            conditions.append(pl.col(column).cast(pl.String).is_in(list(values)))
    for column, (low, high) in spec.ranges.items():
        condition = pl.col(column) >= low
        if high < spec.range_caps.get(column, math.inf):
            condition &= pl.col(column) <= high
        conditions.append(condition)
    return frame.filter(*conditions) if conditions else frame


def active_filter_count(spec: FilterSpec, frame: pl.DataFrame) -> int:
    """Number of filters that actually restrict the data."""
    count = sum(bool(values) for values in spec.categories.values())
    bounds = range_bounds(frame)
    count += sum(tuple(spec.ranges[column]) != bounds[column] for column in spec.ranges)
    if spec.date_range is not None and spec.date_range != date_bounds(frame):
        count += 1
    return count
