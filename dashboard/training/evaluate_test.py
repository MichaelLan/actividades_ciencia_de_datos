"""Single held-out evaluation of the selected Phase 4 models on the Phase 2 test partition.

Run from the ``dashboard`` folder after ``training.train``::

    uv run python -m training.evaluate_test --processed-dir ../preprocesamiento/data/processed

The script only *loads* the already-fitted artifacts (trained on the train partition,
hyperparameters fixed in Phase 3 / ``training.train``) and scores them once on the test
partition. Nothing is refitted, tuned or thresholded on test: the decision threshold stays
at 0.50 and the K-Means model, scaler and segment mapping are reused as they are. The
segmentation is scored on the test part of the Phase 2 clustering partition, rebuilt from
``cleaned.parquet`` exactly as in training.
Output: ``artifacts/test_metrics.json``.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, silhouette_score

from hotel_dashboard import config
from hotel_dashboard.modeling import prepare_features, scale_for_clustering
from training.train import (
    DEFAULT_PROCESSED,
    classification_scores,
    clustering_partitions,
    regression_scores,
    sampled_silhouette,
)

BOOTSTRAP_ROWS = 1_000


def bootstrap_interval(statistic, *arrays, repeats: int = BOOTSTRAP_ROWS) -> list[float]:
    """Percentile 95 % interval of ``statistic`` over row-level bootstrap resamples.

    Rows are resampled independently, so the interval ignores the dependence between
    bookings that share an original fingerprint group; it is an approximate summary of
    sampling variability, not an exact confidence interval.
    """
    rng = np.random.default_rng(config.SEED)
    size = len(arrays[0])
    values = []
    for _ in range(repeats):
        index = rng.integers(0, size, size)
        values.append(statistic(*(array[index] for array in arrays)))
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_regression(test: pl.DataFrame) -> dict:
    """Score the saved ADR Random Forest on non-cancelled test bookings."""
    model = joblib.load(config.ADR_MODEL_PATH)
    y = test["adr"].to_numpy()
    predicted = model.predict(prepare_features(test))
    return {
        "rows": test.height,
        "random_forest": regression_scores(y, predicted),
        "rmse_ci95": bootstrap_interval(lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), y, predicted),
    }


def evaluate_classification(test: pl.DataFrame) -> dict:
    """Score the saved cancellation Random Forest at the fixed 0.50 threshold."""
    model = joblib.load(config.CANCEL_MODEL_PATH)
    y = test["is_canceled"].to_numpy()
    probability = model.predict_proba(prepare_features(test))[:, 1]
    return {
        "rows": test.height,
        "prevalence": float(y.mean()),
        "threshold": config.THRESHOLD,
        "random_forest": classification_scores(y, probability),
        "roc_auc_ci95": bootstrap_interval(roc_auc_score, y, probability),
    }


def evaluate_segmentation(test: pl.DataFrame) -> dict:
    """Silhouette and segment shares of test bookings under the saved K-Means model.

    ``silhouette`` follows the Phase 3 protocol (2,500 rows, seed 42), as for train and
    validation; ``silhouette_full`` uses every test booking.
    """
    segmentation = joblib.load(config.SEGMENTATION_PATH)
    scaled = scale_for_clustering(segmentation, test)
    labels = segmentation["kmeans"].predict(scaled)
    segments = [segmentation["cluster_to_segment"][int(label)] for label in labels]
    counts = pl.Series("segment", segments).value_counts().sort("count", descending=True)
    return {
        "rows": test.height,
        "silhouette": sampled_silhouette(scaled, labels),
        "silhouette_full": float(silhouette_score(scaled, labels)),
        "sizes": dict(zip(counts["segment"].to_list(), counts["count"].to_list(), strict=True)),
    }


def main() -> None:
    """Load the test partitions once, score every selected model and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()

    regression_test = pl.read_parquet(args.processed_dir / "regression_test.parquet")
    classification_test = pl.read_parquet(args.processed_dir / "classification_test.parquet")
    clustering_test = clustering_partitions(pl.read_parquet(args.processed_dir / "cleaned.parquet"))["test"]
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Single evaluation of fitted artifacts on the untouched test partition; no refit or tuning.",
        "regression": evaluate_regression(regression_test),
        "classification": evaluate_classification(classification_test),
        "clustering": evaluate_segmentation(clustering_test),
    }
    path = config.ARTIFACTS_DIR / "test_metrics.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
