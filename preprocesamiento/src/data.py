"""Raw acquisition, strict schema and JSON serialization."""

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]
CATEGORICAL = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "reservation_status",
    "is_canceled",
]
STRINGS = set(CATEGORICAL) - {"is_canceled", "is_repeated_guest"} | {
    "reservation_status_date"
}
SCHEMA = {
    name: pl.String
    if name in STRINGS
    else pl.Float64
    if name in {"children", "adr"}
    else pl.Int64
    for name in COLUMNS
}
DATASET = "jessemostipak/hotel-booking-demand"


def read_bookings(path: Path) -> pl.DataFrame:
    names = pl.read_csv(path, n_rows=0).schema
    if set(names) != set(COLUMNS):
        raise ValueError("Expected 32 columns matching Hotel Booking Demand schema")
    frame = pl.read_csv(
        path,
        schema_overrides=SCHEMA,
        null_values=["NULL", "NA", ""],
        infer_schema_length=10000,
    )
    if frame.is_empty():
        raise ValueError("Dataset must contain bookings")
    if (
        not set(frame["is_canceled"].drop_nulls().unique()) <= {0, 1}
        or frame["is_canceled"].null_count()
    ):
        raise ValueError("is_canceled must contain only 0 and 1")
    return frame.select(COLUMNS)


def prepare(raw_dir: Path, output_dir: Path) -> pl.DataFrame:
    path = raw_dir / "hotel_bookings.csv"
    source_metadata = output_dir / "source.json"
    prior = (
        json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_metadata.exists()
        else {}
    )
    acquisition = prior.get(
        "acquisition",
        {
            "obtained_at_utc": datetime.now(UTC).isoformat(),
            "origin": "local_cached" if path.exists() else "pending",
            "downloaded_at_utc": None,
            "local_file_modified_at_utc": datetime.fromtimestamp(
                path.stat().st_mtime, UTC
            ).isoformat()
            if path.exists()
            else None,
            "meaning": "Local acquisition/inspection recorded now; original download timestamp was not captured and remains unknown. Filesystem modification is not proof of network download.",
        },
    )
    if not path.exists():
        import kagglehub

        try:
            source = Path(kagglehub.dataset_download(DATASET)) / path.name
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            acquisition = {
                "obtained_at_utc": datetime.now(UTC).isoformat(),
                "origin": "kagglehub_cache_or_download",
                "downloaded_at_utc": None,
                "method": "kagglehub.dataset_download (may reuse Kaggle cache), then local original copy",
                "kaggle_cache_version": source.parent.name,
            }
        except Exception as exc:
            raise RuntimeError(f"Kaggle download failed for {DATASET}: {exc}") from exc
    frame = read_bookings(path)
    if frame.shape != (119390, 32):
        raise ValueError(f"Unexpected public dataset dimensions: {frame.shape}")
    write_json(
        output_dir / "source.json",
        {
            "dataset": DATASET,
            "acquisition": acquisition,
            "license": "CC BY 4.0, independently verified Kaggle and Antonio et al.; see specification.md",
            "attribution": "Antonio, de Almeida and Nunes (2019), DOI 10.1016/j.dib.2018.11.126; Kaggle redistributor Jesse Mostipak",
            "kaggle_version": 1,
            "url": f"https://www.kaggle.com/datasets/{DATASET}",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": frame.height,
            "columns": frame.width,
            "schema": {key: str(value) for key, value in frame.schema.items()},
        },
    )
    return frame


def write_json(path: Path, data: object) -> None:
    def sanitize(value):
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "item"):
            return sanitize(value.item())
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
