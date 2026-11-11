"""Small offline behavioral tests."""

import numpy as np
import polars as pl
import pytest

from src.cleaning import clean_bookings
from src.data import COLUMNS, read_bookings
from src.modeling import (
    feature_columns,
    make_preprocessor,
    score_predictions,
    split_bookings,
)


def fixture_frame(size=200):
    values = {name: [0] * size for name in COLUMNS}
    for name in (
        "hotel",
        "arrival_date_month",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type",
        "reservation_status",
        "reservation_status_date",
    ):
        values[name] = ["A"] * size
    values.update(
        adults=[2] * size,
        children=[None] * size,
        babies=[0] * size,
        stays_in_week_nights=[2] * size,
        adr=[80.0] * size,
        is_canceled=[i % 2 for i in range(size)],
        lead_time=list(range(size)),
    )
    return pl.DataFrame(values)


def test_read_schema_and_nulls(tmp_path):
    path = tmp_path / "bookings.csv"
    fixture_frame().write_csv(path)
    frame = read_bookings(path)
    assert frame.shape == (200, 32)
    assert frame["children"].null_count() == 200
    assert frame.schema["agent"] == pl.String
    pl.DataFrame({"bad": [1]}).write_csv(path)
    with pytest.raises(ValueError, match="Expected 32 columns"):
        read_bookings(path)


def test_cleaning_trace_features_and_preserved_duplicates():
    frame = fixture_frame(6).with_columns(
        pl.Series("adr", [80.0, -1.0, 0.0, 5400.0, 20.0, 30.0]),
        pl.Series("adults", [2, 2, 0, 2, 2, 2]),
        pl.Series("stays_in_week_nights", [2, 2, 2, 2, 0, 2]),
    )
    frame = pl.concat([frame, frame.head(1)])
    cleaned, audit = clean_bookings(frame)
    assert cleaned.height == 4
    assert cleaned["total_nights"].to_list() == [2, 2, 2, 2]
    assert cleaned["total_guests"].null_count() == 4
    assert audit["removed_rows"] == 3
    assert cleaned["original_fingerprint"].n_unique() == 3
    assert audit["rules"]["negative_adr"] == 1


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_known_adults_survive_unknown_children(task):
    numeric, categorical = feature_columns(task)
    assert "adults" in numeric
    assert "total_guests" not in numeric + categorical
    train, _ = clean_bookings(
        fixture_frame(4).with_columns(pl.Series("children", [0.0, 1.0, 1.0, 2.0]))
    )
    processor = make_preprocessor(numeric, categorical)
    processor.fit(train.select(numeric + categorical))
    numeric_pipeline = processor.named_transformers_["numeric"]
    children_index = numeric.index("children")
    adults_index = numeric.index("adults")
    assert numeric_pipeline.named_steps["imputer"].statistics_[children_index] == 1
    before = numeric_pipeline.named_steps["imputer"].statistics_.copy()
    for adults in [3, 5]:
        holdout, _ = clean_bookings(
            fixture_frame(1).with_columns(
                pl.lit(adults).alias("adults"),
                pl.lit(None, dtype=pl.Float64).alias("children"),
                pl.lit("UNSEEN").alias("hotel"),
            )
        )
        assert holdout["total_guests"].null_count() == 1
        values = holdout.select(numeric)
        imputed = numeric_pipeline.named_steps["imputer"].transform(values)
        assert imputed[0, adults_index] == adults
        assert imputed[0, children_index] == 1
        assert (
            imputed[0, adults_index]
            + imputed[0, children_index]
            + imputed[0, numeric.index("babies")]
            == adults + 1
        )
        assert np.isfinite(
            processor.transform(holdout.select(numeric + categorical))
        ).all()
        np.testing.assert_array_equal(
            before, numeric_pipeline.named_steps["imputer"].statistics_
        )


def test_weighted_600_rows_keep_groups_and_stratify():
    raw = fixture_frame(400).with_columns(
        pl.lit(0.0, dtype=pl.Float64).alias("children")
    )
    frame, _ = clean_bookings(raw)
    weighted = pl.concat([frame, *[frame.head(1)] * 200])
    parts = split_bookings(weighted, classification=True)
    assert [part.height for part in parts.values()] == [450, 75, 75]
    assert parts["train"]["is_canceled"].to_numpy().mean() == pytest.approx(1 / 3)
    again = split_bookings(weighted, classification=True)
    for name, part in parts.items():
        assert part.equals(again[name])
        assert abs(part["is_canceled"].to_numpy().mean() - 1 / 3) < 0.015
    sets = [set(part["original_fingerprint"]) for part in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(weighted["original_fingerprint"])


@pytest.mark.parametrize("classification", [False, True])
def test_repeated_groups_are_represented_in_every_split(classification):
    frame, _ = clean_bookings(fixture_frame(480))
    weighted = pl.concat([frame, *[frame.head(48)] * 3])
    parts = split_bookings(weighted, classification=classification)
    again = split_bookings(weighted, classification=classification)
    assert [part.height for part in parts.values()] == [468, 78, 78]
    for name, part in parts.items():
        sizes = part.group_by("original_fingerprint").len()
        repeated_rows = int(sizes.filter(pl.col("len") > 1)["len"].sum())
        assert repeated_rows > 0, name
        assert abs(repeated_rows / part.height - 192 / 624) < 0.04
        if classification:
            assert abs(part["is_canceled"].to_numpy().mean() - 0.5) < 0.04
        assert part.equals(again[name])
    sets = [set(part["original_fingerprint"]) for part in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(weighted["original_fingerprint"])


def test_impossible_4400_rows_rejected_before_export(tmp_path):
    from src.modeling import run_task

    frame, _ = clean_bookings(fixture_frame(400))
    weighted = pl.concat([frame, *[frame.head(1)] * 4000])
    with pytest.raises(ValueError, match="Cannot satisfy.*70.*80.*4001.*4400"):
        split_bookings(weighted, classification=True)
    with pytest.raises(ValueError, match="Cannot satisfy"):
        run_task(weighted, "classification", tmp_path / "processed")
    assert not (tmp_path / "processed").exists()


def test_unequal_groups_train_fraction_meets_guide():
    frame, _ = clean_bookings(fixture_frame())
    # Los grupos de distinto peso no deben reducir el entrenamiento bajo el mínimo.
    unequal = pl.concat([frame, frame.head(30), frame.head(10), frame.head(5)])
    parts = split_bookings(unequal, classification=True)
    assert 0.70 <= parts["train"].height / unequal.height <= 0.80
    assert sum(part.height for part in parts.values()) == unequal.height
    sets = [set(part["original_fingerprint"]) for part in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_grouped_stratified_reproducible_splits():
    frame, _ = clean_bookings(pl.concat([fixture_frame()] * 2))
    parts = split_bookings(frame, classification=True)
    again = split_bookings(frame, classification=True)
    assert sum(part.height for part in parts.values()) == frame.height
    for name, part in parts.items():
        assert part.equals(again[name])
        assert abs(part["is_canceled"].to_numpy().mean() - 0.5) < 0.04
        assert (
            abs(
                part.height / frame.height
                - {"train": 0.75, "validation": 0.125, "test": 0.125}[name]
            )
            < 0.06
        )
    sets = [set(part["original_fingerprint"]) for part in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_features_exclude_targets_and_uncertain_information(task):
    numeric, categorical = feature_columns(task)
    assert not set(numeric + categorical) & {
        "is_canceled",
        "adr",
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "booking_changes",
        "days_in_waiting_list",
        "deposit_type",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "original_fingerprint",
        "row_id",
        "country",
        "agent",
        "company",
    }
    assert "total_nights" in numeric
    assert "hotel" in categorical


def test_train_only_imputation_scaling_and_unknown_categories():
    train = pl.DataFrame({"value": [1.0, None, 3.0], "category": ["A", "A", "B"]})
    test = pl.DataFrame({"value": [100.0, None], "category": ["NEW", "A"]})
    processor = make_preprocessor(["value"], ["category"])
    processor.fit(train)
    before = (
        processor.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_.copy()
    )
    transformed = processor.transform(test)
    np.testing.assert_array_equal(before, [2.0])
    np.testing.assert_array_equal(
        processor.named_transformers_["numeric"].named_steps["scale"].mean_, [2.0]
    )
    np.testing.assert_array_equal(
        processor.named_transformers_["categorical"]
        .named_steps["encode"]
        .categories_[0],
        ["A", "B"],
    )
    np.testing.assert_array_equal(
        processor.named_transformers_["numeric"].named_steps["imputer"].statistics_,
        before,
    )
    assert transformed.shape == (2, 3)
    assert np.isfinite(transformed).all()


def test_metrics_exact_values():
    regression = score_predictions(np.array([1.0, 3.0]), np.array([2.0, 2.0]), None)
    assert regression == {"mae": 1.0, "rmse": 1.0, "r2": 0.0}
    classification = score_predictions(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.4, 0.2])
    )
    assert classification["accuracy"] == 0.75
    assert classification["confusion_matrix"] == [[2, 0], [1, 1]]
    assert classification["roc_auc"] == 1.0


def test_summary_handles_entirely_missing_column():
    from src.eda import quantitative_summary

    result = quantitative_summary(pl.Series("children", [None, None], dtype=pl.Float64))
    assert result["valid"] == 0
    assert result["mean"] is None
    assert result["iqr_outliers"] == 0
    assert result["mode"] == []


def test_summary_exact_statistics():
    from src.eda import quantitative_summary

    result = quantitative_summary(pl.Series("value", [1.0, 2.0, 3.0, 4.0, None]))
    assert result["valid"] == 4
    assert result["mean"] == result["median"] == 2.5
    assert result["variance"] == pytest.approx(5 / 3)
    assert result["q1"] == 1.75
    assert result["q3"] == 3.25
    assert result["iqr"] == 1.5
    assert result["mode"] == [1.0, 2.0, 3.0, 4.0]


def test_chi_square_expected_and_effect():
    from src.eda import chi_square

    result = chi_square(
        pl.DataFrame(
            {
                "hotel": ["A"] * 20 + ["B"] * 20,
                "is_canceled": [0] * 15 + [1] * 5 + [0] * 5 + [1] * 15,
            }
        ),
        "hotel",
    )
    assert result["chi2"] == pytest.approx(10.0)
    assert result["cramers_v"] == pytest.approx(0.5)
    assert result["expected_min"] == 10
    assert result["assumptions_met"]


def test_invalid_binary_target_is_rejected(tmp_path):
    path = tmp_path / "input.csv"
    fixture_frame(2).with_columns(pl.Series("is_canceled", [0, 2])).write_csv(path)
    with pytest.raises(ValueError, match="only 0 and 1"):
        read_bookings(path)


def test_download_failure_is_explicit_and_preserves_raw(tmp_path, monkeypatch):
    import kagglehub

    from src.data import prepare

    def fail_download(_):
        raise OSError("offline fixture")

    monkeypatch.setattr(kagglehub, "dataset_download", fail_download)
    with pytest.raises(RuntimeError, match="Kaggle download failed.*offline fixture"):
        prepare(tmp_path / "raw", tmp_path / "outputs")
    assert not (tmp_path / "raw" / "hotel_bookings.csv").exists()
    assert not (tmp_path / "outputs").exists()


def test_offline_task_exports_are_target_safe(tmp_path):
    from src.modeling import run_task

    frame, _ = clean_bookings(fixture_frame())
    for task in ["classification", "regression"]:
        result = run_task(frame, task, tmp_path)
        assert set(result["metrics"]["dummy"]) == {"validation", "test"}
        assert set(result["fingerprint_overlap"].values()) == {0}
        for split in ["train", "validation", "test"]:
            stored = pl.read_parquet(tmp_path / f"{task}_{split}.parquet")
            assert stored.height == result["splits"][split]["rows"]
            assert "reservation_status" not in stored.columns
            assert result["target"] in stored.columns
            assert "total_guests" in stored.columns
            assert "total_guests" not in result["numeric_features"]
            assert "adults" in result["numeric_features"]
            sizes = result["splits"][split]["group_sizes"]
            assert sum(item["rows"] for item in sizes["histogram"]) == stored.height
            assert (
                sum(item["groups"] for item in sizes["histogram"])
                == stored["original_fingerprint"].n_unique()
            )
