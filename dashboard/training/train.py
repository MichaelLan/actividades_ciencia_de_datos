"""Reproducible Phase 4 training: refits the Phase 3 selected models and writes app artifacts.

Run from the ``dashboard`` folder::

    uv run python -m training.train --processed-dir ../preprocesamiento/data/processed

Inputs are the Phase 2 Parquet partitions (train/validation) and ``cleaned.parquet``, from
which the unstratified clustering partition of Phase 2 is rebuilt. The test partitions are
kept untouched, as in Phase 3. Outputs: compressed joblib pipelines, ``metrics.json``,
validation predictions and the compact ``data/bookings.parquet`` used by the app.

Every setting mirrors the executed Phase 3 notebook (``fase_3/Fase_3_Hotel_Booking.ipynb``):
the winning ``RandomizedSearchCV`` configurations of both Random Forests, the sparse one-hot
preprocessor, permutation importance on 800 validation rows x 3 repeats, K-Means with three
seeds (``n_init=10``, ``max_iter=500``, ``float32`` input, silhouette on 2,500 rows) and the
DBSCAN screening on a 5,000-row training sample. With scikit-learn 1.7.2 (the notebook
version) the validation metrics of the notebook are reproduced exactly.
"""

import argparse
import itertools
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hotel_dashboard import config
from hotel_dashboard.modeling import cluster_matrix, make_pipeline_for, prepare_features, scale_for_clustering

DEFAULT_PROCESSED = config.ROOT.parent / "preprocesamiento" / "data" / "processed"

# Winning candidates of the Phase 3 RandomizedSearchCV (3 candidates x 3 grouped folds,
# refit on RMSE / ROC AUC). ``best_params_`` was not printed in the notebook; each winner was
# identified by refitting the three logged candidates on train: only these reproduce the
# notebook validation metrics exactly (and they also rank first in cross-validation).
ADR_FOREST = {
    "n_estimators": 58,
    "max_depth": 18,
    "max_features": 0.5,
    "min_samples_leaf": 6,
    "min_samples_split": 3,
}
CANCEL_FOREST = {
    "n_estimators": 53,
    "max_depth": 18,
    "max_features": "sqrt",
    "min_samples_leaf": 8,
    "min_samples_split": 8,
    "class_weight": None,
}
PHASE3_SEARCH = {
    "notebook": "fase_3/Fase_3_Hotel_Booking.ipynb",
    "method": "RandomizedSearchCV",
    "n_iter": 3,
    "cv_folds": 3,
    "random_state": config.SEED,
    "regression": {
        "cv": "GroupKFold(3) por original_fingerprint",
        "refit": "rmse",
        "space": {
            "n_estimators": "randint(35, 91)",
            "max_depth": [6, 10, 14, 18],
            "min_samples_split": "randint(2, 21)",
            "min_samples_leaf": "randint(1, 11)",
            "max_features": [1.0, "sqrt", 0.5, 0.75],
        },
        "candidates": [
            {
                "n_estimators": 53,
                "max_depth": 14,
                "max_features": 0.75,
                "min_samples_leaf": 8,
                "min_samples_split": 8,
            },
            {
                "n_estimators": 74,
                "max_depth": 14,
                "max_features": 0.5,
                "min_samples_leaf": 8,
                "min_samples_split": 5,
            },
            ADR_FOREST,
        ],
        "selected_index": 2,
        "cv_rmse_mean": 23.433,
        "cv_rmse_std": 0.186,
    },
    "classification": {
        "cv": "StratifiedGroupKFold(3, shuffle=True, random_state=42) por original_fingerprint",
        "refit": "roc_auc",
        "space": {
            "n_estimators": "randint(35, 91)",
            "max_depth": [6, 10, 14, 18],
            "min_samples_split": "randint(2, 21)",
            "min_samples_leaf": "randint(1, 11)",
            "max_features": ["sqrt", "log2", 0.5, 0.75],
            "class_weight": [None, "balanced"],
        },
        "candidates": [
            CANCEL_FOREST,
            {
                "n_estimators": 74,
                "max_depth": 14,
                "max_features": 0.5,
                "min_samples_leaf": 8,
                "min_samples_split": 5,
                "class_weight": None,
            },
            {
                "n_estimators": 58,
                "max_depth": 14,
                "max_features": "log2",
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "class_weight": "balanced",
            },
        ],
        "selected_index": 0,
        "cv_roc_auc_mean": 0.823,
        "cv_roc_auc_std": 0.006,
    },
}
PERMUTATION = {"rows": 800, "repeats": 3}
K_CLUSTERS = 6
K_RANGE = range(2, 11)
KMEANS_SEEDS = (config.SEED, config.SEED + 1000, config.SEED + 2000)
KMEANS_SETTINGS = {"init": "k-means++", "n_init": 10, "max_iter": 500}
SILHOUETTE_SAMPLE = 2_500
DBSCAN_SAMPLE = 5_000
DBSCAN_NEIGHBORS = 10
DBSCAN_QUANTILES = (0.85, 0.90, 0.95, 0.975)
DBSCAN_MIN_SAMPLES = (5, 10, 20)
SPLIT_FRACTIONS = (0.75, 0.125, 0.125)

PHASE3_REFERENCE = {
    "regression": {
        "Constante": {"mae": 37.39, "rmse": 49.16, "r2": 0.0},
        "Regresión lineal": {"mae": 22.35, "rmse": 31.39, "r2": 0.592},
        "ElasticNet": {"mae": 22.35, "rmse": 31.41, "r2": 0.592},
        "Random Forest": {"mae": 15.09, "rmse": 23.54, "r2": 0.771},
    },
    "classification": {
        "Logística": {"accuracy": 0.731, "precision": 0.764, "recall": 0.401, "f1": 0.526, "roc_auc": 0.754},
        "Random Forest": {
            "accuracy": 0.755,
            "precision": 0.867,
            "recall": 0.406,
            "f1": 0.553,
            "roc_auc": 0.812,
        },
        "XGBoost": {"accuracy": 0.755, "precision": 0.806, "recall": 0.452, "f1": 0.579, "roc_auc": 0.808},
        "MLP pequeña": {
            "accuracy": 0.757,
            "precision": 0.776,
            "recall": 0.488,
            "f1": 0.599,
            "roc_auc": 0.804,
        },
        "MLP balanceada": {
            "accuracy": 0.759,
            "precision": 0.816,
            "recall": 0.456,
            "f1": 0.585,
            "roc_auc": 0.810,
        },
        "MLP regularizada": {
            "accuracy": 0.756,
            "precision": 0.837,
            "recall": 0.429,
            "f1": 0.567,
            "roc_auc": 0.804,
        },
    },
    "clustering": {
        "k": 6,
        "silhouette_train": 0.334,
        "silhouette_validation": 0.341,
        "sizes_train": [54492, 6274, 695, 222, 27181, 59],
        "median_lead_time": [34, 64, 51, 4, 211, 244],
        "median_nights": [2, 3, 4, 2, 4, 2],
        "dbscan": {"clusters": 5, "noise_share": 0.0298, "silhouette": 0.349},
    },
}


def regression_scores(y, predicted) -> dict:
    """MAE, RMSE and R² for a set of predictions."""
    return {
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(root_mean_squared_error(y, predicted)),
        "r2": float(r2_score(y, predicted)),
    }


def classification_scores(y, probability, threshold: float = config.THRESHOLD) -> dict:
    """Threshold metrics, confusion matrix and ROC AUC for probabilities."""
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def importance(pipeline, validation: pl.DataFrame, target: str, scoring: str) -> list[dict]:
    """Permutation importance on 800 validation rows and three repeats, sampled as in Phase 3."""
    sample = validation.sample(n=min(PERMUTATION["rows"], validation.height), shuffle=True, seed=config.SEED)
    features = prepare_features(sample)
    result = permutation_importance(
        pipeline,
        features,
        sample[target].to_numpy(),
        scoring=scoring,
        n_repeats=PERMUTATION["repeats"],
        random_state=config.SEED,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return [
        {
            "feature": features.columns[index],
            "mean": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index in order
    ]


def train_regression(train: pl.DataFrame, validation: pl.DataFrame) -> tuple:
    """Fit ADR baselines and the Random Forest on non-cancelled bookings."""
    x_train, x_val = prepare_features(train), prepare_features(validation)
    y_train, y_val = train["adr"].to_numpy(), validation["adr"].to_numpy()
    candidates = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            random_state=config.SEED, n_jobs=-1, bootstrap=True, **ADR_FOREST
        ),
    }
    metrics, fitted, predictions = {}, {}, {}
    for name, estimator in candidates.items():
        # The forest uses the sparse Phase 3 preprocessor; baselines keep the Phase 2 dense one.
        pipeline = make_pipeline_for(estimator, sparse_output=name == "random_forest").fit(x_train, y_train)
        predictions[name] = pipeline.predict(x_val)
        metrics[name] = {
            "train": regression_scores(y_train, pipeline.predict(x_train)),
            "validation": regression_scores(y_val, predictions[name]),
        }
        fitted[name] = pipeline
        print(f"ADR {name}: {metrics[name]['validation']}")
    forest = fitted["random_forest"]
    return (
        {
            "rows": {"train": train.height, "validation": validation.height},
            "params": ADR_FOREST,
            "search": PHASE3_SEARCH["regression"],
            "metrics": metrics,
            "permutation_importance": importance(forest, validation, "adr", "neg_root_mean_squared_error"),
        },
        forest,
        pl.DataFrame(
            {
                "adr": y_val,
                "adr_random_forest": predictions["random_forest"],
                "adr_linear": predictions["linear"],
            }
        ),
    )


def train_classification(train: pl.DataFrame, validation: pl.DataFrame) -> tuple:
    """Fit cancellation baselines and the Random Forest without leakage columns."""
    x_train, x_val = prepare_features(train), prepare_features(validation)
    y_train, y_val = train["is_canceled"].to_numpy(), validation["is_canceled"].to_numpy()
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=2000, random_state=config.SEED),
        "random_forest": RandomForestClassifier(random_state=config.SEED, n_jobs=-1, **CANCEL_FOREST),
    }
    metrics, fitted, probabilities, roc = {}, {}, {}, {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline_for(estimator, sparse_output=name == "random_forest").fit(x_train, y_train)
        probabilities[name] = pipeline.predict_proba(x_val)[:, 1]
        metrics[name] = {
            "train": classification_scores(y_train, pipeline.predict_proba(x_train)[:, 1]),
            "validation": classification_scores(y_val, probabilities[name]),
        }
        fpr, tpr, _ = roc_curve(y_val, probabilities[name])
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        roc[name] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}
        fitted[name] = pipeline
        print(f"Cancelación {name}: {metrics[name]['validation']}")
    forest = fitted["random_forest"]
    return (
        {
            "rows": {"train": train.height, "validation": validation.height},
            "validation_prevalence": float(y_val.mean()),
            "params": CANCEL_FOREST,
            "search": PHASE3_SEARCH["classification"],
            "threshold": config.THRESHOLD,
            "metrics": metrics,
            "roc": roc,
            "permutation_importance": importance(forest, validation, "is_canceled", "roc_auc"),
        },
        forest,
        pl.DataFrame(
            {
                "is_canceled": y_val,
                "p_random_forest": probabilities["random_forest"],
                "p_logistic": probabilities["logistic"],
            }
        ),
    )


def name_clusters(train: pl.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Map arbitrary K-Means ids to semantic segment keys from their profiles."""
    profile = (
        train.with_columns(cluster=pl.Series(labels))
        .group_by("cluster")
        .agg(pl.col(config.CLUSTER_FEATURES).mean())
    )
    remaining = set(profile["cluster"].to_list())
    mapping: dict[int, str] = {}
    for key, column in [
        ("serial_cancellers", "previous_cancellations"),
        ("loyal", "previous_bookings_not_canceled"),
        ("baby", "babies"),
        ("children", "children"),
    ]:
        best = profile.filter(pl.col("cluster").is_in(list(remaining))).sort(column, descending=True)[
            "cluster"
        ][0]
        mapping[int(best)] = key
        remaining.discard(best)
    by_lead = profile.filter(pl.col("cluster").is_in(list(remaining))).sort("lead_time")["cluster"].to_list()
    mapping[int(by_lead[0])] = "short_lead"
    mapping[int(by_lead[1])] = "long_lead"
    return mapping


def segment_profiles(frame: pl.DataFrame, column: str) -> list[dict]:
    """Size, mean and median profile per segment, plus business outcomes."""
    return (
        frame.group_by(column)
        .agg(
            pl.len().alias("size"),
            *[pl.col(name).mean().alias(f"{name}_mean") for name in config.CLUSTER_FEATURES],
            *[pl.col(name).median().alias(f"{name}_median") for name in config.CLUSTER_FEATURES],
            pl.col("is_canceled").mean().alias("cancel_rate"),
        )
        .sort(column)
        .to_dicts()
    )


def clustering_partitions(cleaned: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Rebuild the Phase 2 clustering partition (``split_bookings(classification=False)``).

    Phase 2 split the clustering data separately from the classification data, without class
    stratification: whole fingerprint groups are assigned greedily (heaviest first, ties broken
    by a seeded random draw) to reach 75/12.5/12.5 % of the rows. The Phase 3 notebook read these
    ``clustering_*.parquet`` files; this port reproduces them from ``cleaned.parquet``.
    """
    groups = (
        cleaned.group_by("original_fingerprint", maintain_order=True)
        .agg(pl.len().alias("weight"))
        .with_columns(
            tie_break=pl.Series(
                np.random.default_rng(config.SEED).random(cleaned.n_unique("original_fingerprint"))
            )
        )
        .sort(["weight", "tie_break"], descending=[True, False])
    )
    target = np.array(SPLIT_FRACTIONS) * cleaned.height
    assigned = np.zeros(3, dtype=np.int64)
    fingerprints: list[list[int]] = [[], [], []]
    for fingerprint, weight in groups.select("original_fingerprint", "weight").iter_rows():
        remaining = target - assigned
        fitting = [index for index in range(3) if weight <= remaining[index]]
        if fitting:
            destination = max(fitting, key=lambda i: (remaining[i] / target[i], target[i], -i))
        else:
            relative = (assigned - target) / target
            increase = ((assigned + weight - target) / target) ** 2 - relative**2
            destination = min(range(3), key=lambda i: (increase[i], -target[i], i))
        fingerprints[destination].append(fingerprint)
        assigned[destination] += weight
    dtype = cleaned.schema["original_fingerprint"]
    return {
        name: cleaned.filter(
            pl.col("original_fingerprint").is_in(pl.Series(fingerprints[index], dtype=dtype).implode())
        )
        for index, name in enumerate(["train", "validation", "test"])
    }


def sampled_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette on at most 2,500 rows drawn with the Phase 3 seed (as in the notebook)."""
    return float(
        silhouette_score(
            matrix, labels, sample_size=min(SILHOUETTE_SAMPLE, len(matrix)), random_state=config.SEED
        )
    )


def kmeans_curve(z_train: np.ndarray) -> tuple[list[dict], dict[int, KMeans]]:
    """Inertia, silhouette and seed stability (ARI) for K = 2..10, as in Phase 3."""
    rows, models = [], {}
    for k in K_RANGE:
        runs = [KMeans(k, random_state=seed, **KMEANS_SETTINGS).fit(z_train) for seed in KMEANS_SEEDS]
        ari = [adjusted_rand_score(a.labels_, b.labels_) for a, b in itertools.combinations(runs, 2)]
        inertia = [run.inertia_ for run in runs]
        models[k] = runs[0]
        rows.append(
            {
                "k": k,
                "inertia": float(np.mean(inertia)),
                "inertia_std": float(np.std(inertia)),
                "silhouette": sampled_silhouette(z_train, runs[0].labels_),
                "stability_ari_mean": float(np.mean(ari)),
                "stability_ari_min": float(np.min(ari)),
            }
        )
        print(f"K={k}: {rows[-1]}")
    return rows, models


def select_k(curve: list[dict], tolerance: float = 0.02, minimum_stability: float = 0.85) -> dict:
    """Phase 3 automatic rule: elbow neighbourhood, silhouette tolerance and ARI >= 0.85."""
    k = np.array([row["k"] for row in curve], dtype=float)
    inertia = np.array([row["inertia"] for row in curve])
    x, y = (k - k.min()) / np.ptp(k), (inertia - inertia.min()) / np.ptp(inertia)
    elbow_k = int(k[np.argmax(np.abs(x + y - 1.0) / np.sqrt(2.0))])
    best = max(row["silhouette"] for row in curve)
    eligible = sorted(
        (
            row
            for row in curve
            if abs(row["k"] - elbow_k) <= 1
            and row["silhouette"] >= best - tolerance
            and row["stability_ari_mean"] >= minimum_stability
        ),
        key=lambda row: (-row["silhouette"], -row["stability_ari_mean"], row["k"]),
    )
    return {
        "elbow_k": elbow_k,
        "max_silhouette": best,
        "max_silhouette_k": max(curve, key=lambda row: row["silhouette"])["k"],
        "automatic_k": eligible[0]["k"] if eligible else None,
        "adopted_k": K_CLUSTERS,
        "decision": "manual: elbow and seed stability (no K met every automatic criterion)"
        if not eligible
        else "automatic",
    }


def dbscan_screening(z_train: np.ndarray) -> dict:
    """DBSCAN pilot of Phase 3: k-distance eps candidates and a 4 x 3 grid on 5,000 rows."""
    rng = np.random.default_rng(config.SEED + DBSCAN_SAMPLE)
    pilot = z_train[np.sort(rng.choice(len(z_train), size=DBSCAN_SAMPLE, replace=False))]
    distances = (
        NearestNeighbors(n_neighbors=DBSCAN_NEIGHBORS, algorithm="ball_tree").fit(pilot).kneighbors(pilot)[0]
    )
    eps_values = sorted({round(float(np.quantile(distances[:, -1], q)), 6) for q in DBSCAN_QUANTILES})
    grid = []
    for min_samples in DBSCAN_MIN_SAMPLES:
        for eps in eps_values:
            labels = DBSCAN(
                eps=eps, min_samples=min_samples, algorithm="ball_tree", leaf_size=40
            ).fit_predict(pilot)
            assigned = labels != -1
            clusters = len(set(labels[assigned]))
            silhouette = (
                sampled_silhouette(pilot[assigned], labels[assigned])
                if clusters >= 2 and assigned.sum() > clusters
                else None
            )
            grid.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "clusters": clusters,
                    "coverage": float(assigned.mean()),
                    "noise_share": float(1 - assigned.mean()),
                    "silhouette": silhouette,
                    "sizes": np.unique(labels, return_counts=True)[1].tolist(),
                }
            )
    eligible = [
        row
        for row in grid
        if 2 <= row["clusters"] <= 20
        and row["coverage"] >= 0.5
        and row["noise_share"] <= 0.5
        and row["silhouette"] is not None
    ]
    selected = min(
        eligible, key=lambda r: (-r["silhouette"], -r["coverage"], r["clusters"], r["eps"], r["min_samples"])
    )
    print(f"DBSCAN: {selected}")
    return {"rows": DBSCAN_SAMPLE, "eps_candidates": eps_values, "grid": grid, **selected}


def train_segmentation(train: pl.DataFrame, validation: pl.DataFrame) -> tuple:
    """K-Means (K=6) on the seven standardized numeric features, PCA map and DBSCAN pilot."""
    scaler = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    scaler.fit(cluster_matrix(train))
    artifact = {"scaler": scaler}
    z_train = scale_for_clustering(artifact, train)
    z_val = scale_for_clustering(artifact, validation)

    k_selection, models = kmeans_curve(z_train)
    kmeans = models[K_CLUSTERS]
    labels_train, labels_val = kmeans.labels_, kmeans.predict(z_val)
    mapping = name_clusters(train, labels_train)
    pca = PCA(n_components=3, random_state=config.SEED).fit(z_train)

    def with_segment(frame, labels):
        return frame.with_columns(segment=pl.Series([mapping[int(label)] for label in labels]))

    metrics = {
        "k": K_CLUSTERS,
        "features": config.CLUSTER_FEATURES,
        "partition": "Fase 2, partición de clustering sin estratificar (split_bookings(classification=False))",
        "rows": {"train": train.height, "validation": validation.height},
        "kmeans_settings": {**KMEANS_SETTINGS, "seeds": list(KMEANS_SEEDS), "dtype": "float32"},
        "silhouette_train": next(row["silhouette"] for row in k_selection if row["k"] == K_CLUSTERS),
        "silhouette_sample_size": SILHOUETTE_SAMPLE,
        "silhouette_validation": sampled_silhouette(z_val, labels_val),
        "k_selection": k_selection,
        "k_decision": select_k(k_selection),
        "pca_explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
        "pca_loadings": {
            feature: pca.components_[:, index].round(4).tolist()
            for index, feature in enumerate(config.CLUSTER_FEATURES)
        },
        "profiles_train": segment_profiles(with_segment(train, labels_train), "segment"),
        "profiles_validation": segment_profiles(with_segment(validation, labels_val), "segment"),
        "dbscan": dbscan_screening(z_train),
    }
    artifact |= {"kmeans": kmeans, "pca": pca, "cluster_to_segment": mapping}
    return metrics, artifact


def build_app_dataset(cleaned: pl.DataFrame, splits: dict, segmentation: dict) -> pl.DataFrame:
    """Compact, typed booking table with split, segment and PCA coordinates."""
    split_labels = pl.concat(
        [part.select("row_id").with_columns(split=pl.lit(name)) for name, part in splits.items()]
    )
    month_number = {name: index + 1 for index, name in enumerate(config.MONTHS)}
    frame = cleaned.join(split_labels, on="row_id", how="left").with_columns(
        arrival_date=pl.date(
            pl.col("arrival_date_year"),
            pl.col("arrival_date_month").replace_strict(month_number),
            pl.col("arrival_date_day_of_month"),
        )
    )
    scaled = scale_for_clustering(segmentation, frame)
    clusters = segmentation["kmeans"].predict(scaled)
    coords = segmentation["pca"].transform(scaled).astype(np.float32)
    frame = frame.with_columns(
        segment=pl.Series([segmentation["cluster_to_segment"][int(c)] for c in clusters]).cast(
            pl.Enum(config.SEGMENT_ORDER)
        ),
        pc1=pl.Series(coords[:, 0]),
        pc2=pl.Series(coords[:, 1]),
        pc3=pl.Series(coords[:, 2]),
    )
    categorical = [
        "hotel",
        "arrival_date_month",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "deposit_type",
        "customer_type",
        "split",
    ]
    small_ints = [
        "lead_time",
        "adults",
        "babies",
        "total_nights",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "total_of_special_requests",
        "required_car_parking_spaces",
        "booking_changes",
    ]
    return frame.select(
        "arrival_date",
        pl.col(categorical).cast(pl.Categorical),
        pl.col("is_canceled").cast(pl.Int8),
        pl.col("is_repeated_guest").cast(pl.Int8),
        pl.col(small_ints).cast(pl.Int16),
        pl.col("children").cast(pl.Float32),
        pl.col("adr").cast(pl.Float32),
        "segment",
        "pc1",
        "pc2",
        "pc3",
    ).sort("arrival_date")


def main() -> None:
    """Train every model, then write artifacts, metrics and the app dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    processed = args.processed_dir

    def read(name: str) -> pl.DataFrame:
        return pl.read_parquet(processed / f"{name}.parquet")

    cleaned = read("cleaned")
    cls = {s: read(f"classification_{s}") for s in ["train", "validation", "test"]}
    reg = {s: read(f"regression_{s}") for s in ["train", "validation"]}

    regression, adr_model, adr_predictions = train_regression(reg["train"], reg["validation"])
    classification, cancel_model, cancel_predictions = train_classification(cls["train"], cls["validation"])
    clu = clustering_partitions(cleaned)
    assert [part.height for part in clu.values()] == [cls[name].height for name in clu], (
        "Unexpected clustering sizes"
    )
    clustering, segmentation = train_segmentation(clu["train"], clu["validation"])

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(adr_model, config.ADR_MODEL_PATH, compress=("xz", 3))
    joblib.dump(cancel_model, config.CANCEL_MODEL_PATH, compress=("xz", 3))
    joblib.dump(segmentation, config.SEGMENTATION_PATH, compress=3)

    pl.concat(
        [
            cancel_predictions.with_columns(task=pl.lit("classification")),
            adr_predictions.with_columns(task=pl.lit("regression")),
        ],
        how="diagonal",
    ).with_columns(pl.selectors.float().cast(pl.Float32)).write_parquet(
        config.PREDICTIONS_PATH, compression="zstd"
    )

    dataset = build_app_dataset(cleaned, cls, segmentation)
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    dataset.write_parquet(config.DATA_PATH, compression="zstd", compression_level=19)

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "polars": pl.__version__,
        },
        "source": str(processed.name),
        "regression": regression,
        "classification": classification,
        "clustering": clustering,
        "phase3_reference": PHASE3_REFERENCE,
        "phase3_search": PHASE3_SEARCH,
        "dataset": {"rows": dataset.height, "citation": config.DATASET_CITATION},
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artefactos escritos en {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
