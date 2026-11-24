"""Tests for data loading, filtering, KPIs and inference on the shipped artifacts."""

from datetime import date

import polars as pl
import pytest

from hotel_dashboard import config, modeling
from hotel_dashboard.data import (
    FilterSpec,
    apply_filters,
    category_options,
    load_bookings,
    load_metrics,
    range_bounds,
)
from hotel_dashboard.kpis import business_kpis, fmt_number, fmt_pct

SAMPLE_BOOKING = {
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
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
}


@pytest.fixture(scope="module")
def bookings() -> pl.DataFrame:
    return load_bookings()


@pytest.fixture(scope="module")
def models() -> modeling.Models:
    return modeling.load_models()


def test_dataset_schema_and_size(bookings):
    assert bookings.height == 118_564
    required = set(config.FEATURES) | {"arrival_date", "is_canceled", "adr", "segment", "pc1", "pc2", "pc3"}
    assert required <= set(bookings.columns)
    assert bookings["segment"].null_count() == 0
    assert bookings["split"].cast(pl.String).value_counts()["count"].sum() == bookings.height


def test_no_filters_returns_everything(bookings):
    assert apply_filters(bookings, FilterSpec()).height == bookings.height


def test_combined_filters(bookings):
    spec = FilterSpec(
        date_range=(date(2016, 1, 1), date(2016, 12, 31)),
        categories={"hotel": ("Resort Hotel",), "market_segment": ()},
        ranges={"lead_time": (10.0, 100.0)},
        range_caps={"lead_time": 480.0},
    )
    result = apply_filters(bookings, spec)
    assert 0 < result.height < bookings.height
    assert result["hotel"].cast(pl.String).unique().to_list() == ["Resort Hotel"]
    assert result["arrival_date"].dt.year().unique().to_list() == [2016]
    assert result["lead_time"].min() >= 10 and result["lead_time"].max() <= 100


def test_range_cap_keeps_outliers(bookings):
    bounds = range_bounds(bookings)
    low, high = bounds["adr"]
    spec = FilterSpec(ranges={"adr": (low, high)}, range_caps={"adr": high})
    assert apply_filters(bookings, spec).height == bookings.height


def test_category_options_cover_values(bookings):
    options = category_options(bookings)
    assert set(options["hotel"]) == {"City Hotel", "Resort Hotel"}
    assert all(config.label(column, value) for column, values in options.items() for value in values)


def test_business_kpis(bookings):
    kpis = business_kpis(bookings)
    assert kpis.bookings == bookings.height
    assert 0.36 < kpis.cancel_rate < 0.38
    assert kpis.revenue > 0 and kpis.room_nights > 0
    assert business_kpis(bookings.head(0)).bookings == 0


def test_spanish_formatting():
    assert fmt_number(118564) == "118.564"
    assert fmt_number(0.8117, 3) == "0,812"
    assert fmt_pct(0.373) == "37,3 %"


def test_cancellation_inference(models):
    probability = modeling.predict_cancellation(models.cancel, SAMPLE_BOOKING)
    assert 0.0 <= probability <= 1.0
    risky = SAMPLE_BOOKING | {"lead_time": 400, "previous_cancellations": 5}
    assert modeling.predict_cancellation(models.cancel, risky) > probability


def test_adr_inference(models):
    estimate, low, high = modeling.predict_adr(models.adr, SAMPLE_BOOKING)
    assert low <= estimate <= high
    assert 50 < estimate < 250


def test_segment_assignment(models):
    segment, coords = modeling.assign_segment(models.segmentation, SAMPLE_BOOKING)
    assert segment in config.SEGMENTS
    assert coords.shape == (3,)
    family = modeling.assign_segment(models.segmentation, SAMPLE_BOOKING | {"babies": 1})[0]
    assert family == "baby"


def test_metrics_reproduce_phase3():
    """Phase 3 winning configurations reproduce the notebook validation metrics."""
    metrics = load_metrics()
    adr = metrics["regression"]["metrics"]["random_forest"]["validation"]
    cancel = metrics["classification"]["metrics"]["random_forest"]["validation"]
    assert metrics["regression"]["params"]["n_estimators"] == 58
    assert metrics["classification"]["params"]["max_features"] == "sqrt"
    # Exact with scikit-learn 1.7.2 (the notebook version); small tolerance for other builds.
    assert abs(adr["mae"] - 15.089) < 0.01 and abs(adr["rmse"] - 23.541) < 0.01
    assert abs(cancel["roc_auc"] - 0.8117) < 0.001
    assert cancel["confusion_matrix"] == [[8954, 345], [3281, 2241]]


def test_segmentation_matches_phase3():
    clustering = load_metrics()["clustering"]
    sizes = sorted(row["size"] for row in clustering["profiles_train"])
    assert sizes == sorted([54492, 6274, 695, 222, 27181, 59])
    assert clustering["dbscan"]["clusters"] == 5 and clustering["dbscan"]["eps"] == 1.438558
