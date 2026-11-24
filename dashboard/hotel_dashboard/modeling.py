"""Feature preparation, pipeline construction and inference helpers."""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_dashboard import config

MISSING = "__MISSING__"


def prepare_features(frame: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    """Select model predictors with categoricals as strings, as in Phase 2."""
    if isinstance(frame, pd.DataFrame):
        frame = pl.from_pandas(frame)
    return (
        frame.select(config.FEATURES)
        .with_columns(
            [pl.col(name).cast(pl.Float64) for name in config.NUMERIC_FEATURES]
            + [pl.col(name).cast(pl.String).fill_null(MISSING) for name in config.CATEGORICAL_FEATURES]
        )
        .to_pandas()
    )


def make_preprocessor(sparse_output: bool = False) -> ColumnTransformer:
    """Median/scale numeric and one-hot encode categoricals, fitted on train only.

    ``sparse_output=True`` reproduces the Phase 3 notebook preprocessor exactly (sparse one-hot
    matrix), which the Random Forests were tuned and fitted on.
    """
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
                config.NUMERIC_FEATURES,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0 if sparse_output else 0.0,
    )


def make_pipeline_for(model, sparse_output: bool = False) -> Pipeline:
    """Wrap an estimator with the shared leakage-safe preprocessor."""
    return Pipeline([("preprocess", make_preprocessor(sparse_output)), ("model", model)])


@dataclass(frozen=True)
class Models:
    """Trained artifacts used for live inference."""

    adr: Pipeline
    cancel: Pipeline
    segmentation: dict


def load_models() -> Models:
    """Load the serialized pipelines from the artifacts folder."""
    return Models(
        adr=joblib.load(config.ADR_MODEL_PATH),
        cancel=joblib.load(config.CANCEL_MODEL_PATH),
        segmentation=joblib.load(config.SEGMENTATION_PATH),
    )


def predict_cancellation(model: Pipeline, booking: dict) -> float:
    """Return the cancellation probability of a single booking."""
    frame = prepare_features(pl.DataFrame([booking]))
    return float(model.predict_proba(frame)[0, 1])


def predict_adr(model: Pipeline, booking: dict) -> tuple[float, float, float]:
    """Return the ADR estimate and the 10th/90th percentile across trees."""
    frame = prepare_features(pl.DataFrame([booking]))
    transformed = model.named_steps["preprocess"].transform(frame)
    forest = model.named_steps["model"]
    per_tree = np.array([tree.predict(transformed)[0] for tree in forest.estimators_])
    low, high = np.percentile(per_tree, [10, 90])
    return float(per_tree.mean()), float(low), float(high)


def cluster_matrix(frame: pl.DataFrame) -> pd.DataFrame:
    """Numeric matrix used by the segmentation pipeline."""
    return frame.select(config.CLUSTER_FEATURES).with_columns(pl.all().cast(pl.Float64)).to_pandas()


def scale_for_clustering(segmentation: dict, frame: pl.DataFrame) -> np.ndarray:
    """Standardize the clustering features as ``float32``, the dtype K-Means was fitted on."""
    return segmentation["scaler"].transform(cluster_matrix(frame)).astype(np.float32)


def assign_segment(segmentation: dict, booking: dict) -> tuple[str, np.ndarray]:
    """Return the segment key and PCA coordinates of a single booking."""
    scaled = scale_for_clustering(segmentation, pl.DataFrame([booking]))
    cluster = int(segmentation["kmeans"].predict(scaled)[0])
    coords = segmentation["pca"].transform(scaled)[0]
    return segmentation["cluster_to_segment"][cluster], coords
