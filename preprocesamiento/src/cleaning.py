"""Domain exclusions without deduplication or learned transformations."""

import polars as pl

from src.data import COLUMNS


def clean_bookings(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    # La huella precede toda transformación: las reservas repetidas se conservan juntas.
    frame = frame.with_columns(
        pl.struct(COLUMNS).hash(seed=42).alias("original_fingerprint")
    ).with_row_index("row_id")
    frame = frame.with_columns(
        (pl.col("stays_in_weekend_nights") + pl.col("stays_in_week_nights")).alias(
            "total_nights"
        ),
        (pl.col("adults") + pl.col("children") + pl.col("babies")).alias(
            "total_guests"
        ),
    )
    rules = {
        "negative_adr": pl.col("adr") < 0,
        "negative_counts": pl.any_horizontal(
            pl.col(name) < 0
            for name in [
                "adults",
                "children",
                "babies",
                "stays_in_weekend_nights",
                "stays_in_week_nights",
                "lead_time",
            ]
        ),
        "zero_guests": (
            pl.col("adults") + pl.col("children").fill_null(0) + pl.col("babies")
        )
        == 0,
        "zero_nights": pl.col("total_nights") == 0,
        "missing_target": pl.col("adr").is_null() | ~pl.col("adr").is_finite(),
    }
    frame = frame.with_columns(
        [expr.fill_null(False).alias(f"invalid_{name}") for name, expr in rules.items()]
    )
    invalid = pl.any_horizontal(pl.col(f"invalid_{name}") for name in rules)
    rejected = frame.filter(invalid)
    cleaned = frame.filter(~invalid)
    return cleaned, {
        "input_rows": frame.height,
        "retained_rows": cleaned.height,
        "removed_rows": rejected.height,
        "rules": {name: int(frame[f"invalid_{name}"].sum()) for name in rules},
        "rejected": rejected.select(
            "row_id", "original_fingerprint", *[f"invalid_{name}" for name in rules]
        ).to_dicts(),
        "policy": "Keep duplicates, zero ADR and finite positive extremes; exclude negative ADR, negative counts, zero known guests and zero nights. Missing children remain unknown; train-only median imputation. IQR tails may represent real demand: retain rather than data-dependent trimming; log/winsorization are alternatives not tuned on test.",
    }
