"""Grouped holdouts and basic, train-only sklearn pipelines."""

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def feature_columns(task: str) -> tuple[list[str], list[str]]:
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown task")
    return [
        "lead_time",
        "adults",
        "children",
        "babies",
        "total_nights",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ], [
        "hotel",
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "is_repeated_guest",
        "reserved_room_type",
        "customer_type",
    ]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def split_bookings(
    frame: pl.DataFrame, classification: bool = False
) -> dict[str, pl.DataFrame]:
    # Repartir por filas de cada clase, sin romper las huellas ni eliminar reservas.
    groups = frame.group_by("original_fingerprint", maintain_order=True).agg(
        pl.col("is_canceled").first(), pl.len().alias("weight")
    )
    largest = int(groups["weight"].to_numpy().max()) if groups.height else 0
    if frame.is_empty() or largest > 0.80 * frame.height:
        raise ValueError(
            f"Cannot satisfy train row fraction 70%-80% with intact original groups: largest group {largest}, total rows {frame.height}"
        )
    groups = groups.with_columns(
        pl.Series("tie_break", np.random.default_rng(SEED).random(groups.height))
    ).sort(["weight", "tie_break"], descending=[True, False])
    weights_by_class = {}
    for row in groups.iter_rows(named=True):
        label = row["is_canceled"] if classification else 0
        weights_by_class[label] = weights_by_class.get(label, 0) + row["weight"]
    fractions = np.array([0.75, 0.125, 0.125])
    targets = {label: fractions * weight for label, weight in weights_by_class.items()}
    assigned = {label: np.zeros(3, dtype=np.int64) for label in targets}
    fingerprints = [[], [], []]
    for row in groups.iter_rows(named=True):
        label = row["is_canceled"] if classification else 0
        target = targets[label]
        remaining = target - assigned[label]
        fitting = [index for index in range(3) if row["weight"] <= remaining[index]]
        if fitting:
            # Déficit relativo: no agotar los grupos repetidos solo en entrenamiento.
            destination = max(
                fitting,
                key=lambda index: (
                    remaining[index] / target[index],
                    target[index],
                    -index,
                ),
            )
        else:
            # Sin encaje exacto, minimizar el aumento del error relativo cuadrático.
            relative_error = (assigned[label] - target) / target
            increase = (
                (assigned[label] + row["weight"] - target) / target
            ) ** 2 - relative_error**2
            destination = min(
                range(3), key=lambda index: (increase[index], -target[index], index)
            )
        fingerprints[destination].append(row["original_fingerprint"])
        assigned[label][destination] += row["weight"]
    parts = {
        name: frame.filter(
            pl.col("original_fingerprint").is_in(
                pl.Series(
                    fingerprints[index], dtype=frame.schema["original_fingerprint"]
                ).implode()
            )
        )
        for index, name in enumerate(["train", "validation", "test"])
    }
    train_fraction = parts["train"].height / frame.height
    if not 0.70 <= train_fraction <= 0.80 or any(
        part.is_empty() for part in parts.values()
    ):
        raise ValueError(
            f"Cannot satisfy train row fraction 70%-80% and nonempty holdouts using weighted greedy assignment: train fraction {train_fraction:.6f}, largest group {largest}, total rows {frame.height}"
        )
    return parts


def group_size_summary(frame: pl.DataFrame) -> dict:
    sizes = frame.group_by("original_fingerprint").len(name="size")
    weights = sizes["size"].to_numpy()
    repeated = weights[weights > 1]
    return {
        "groups": len(weights),
        "singleton_groups": int((weights == 1).sum()),
        "repeated_groups": len(repeated),
        "rows_in_repeated_groups": int(repeated.sum()),
        "fraction_rows_in_repeated_groups": float(repeated.sum() / frame.height),
        "min": int(weights.min()),
        "median": float(np.median(weights)),
        "max": int(weights.max()),
        "histogram": sizes.group_by("size")
        .len(name="groups")
        .with_columns((pl.col("size") * pl.col("groups")).alias("rows"))
        .sort("size")
        .to_dicts(),
    }


def score_predictions(y, predicted, probability) -> dict:
    if probability is None:
        return {
            "mae": float(mean_absolute_error(y, predicted)),
            "rmse": float(root_mean_squared_error(y, predicted)),
            "r2": float(r2_score(y, predicted)),
        }
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else None,
    }


def run_task(frame: pl.DataFrame, task: str, processed_dir) -> dict:
    classification = task == "classification"
    if not classification:
        frame = frame.filter(pl.col("is_canceled") == 0)
    parts = split_bookings(frame, classification)
    numeric, categorical = feature_columns(task)
    target = "is_canceled" if classification else "adr"

    # NumPy solamente al límite sklearn; los nominales binarios tampoco son magnitudes.
    def x(part):
        return part.select(numeric + categorical).with_columns(
            [
                pl.col(name).cast(pl.String).fill_null("__MISSING__")
                for name in categorical
            ]
        )

    prevalence = float(parts["train"]["is_canceled"].to_numpy().mean())
    severe = min(prevalence, 1 - prevalence) < 0.10
    models = {
        "dummy": DummyClassifier(strategy="prior")
        if classification
        else DummyRegressor(strategy="mean"),
        "logistic" if classification else "linear": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced" if severe else None,
        )
        if classification
        else LinearRegression(),
    }
    results = {}
    diagnostics = {}
    for name, model in models.items():
        pipeline = Pipeline(
            [("preprocess", make_preprocessor(numeric, categorical)), ("model", model)]
        )
        pipeline.fit(x(parts["train"]), parts["train"][target].to_numpy())
        results[name] = {}
        for split in ["validation", "test"]:
            predicted = pipeline.predict(x(parts[split]))
            probability = (
                pipeline.predict_proba(x(parts[split]))[:, 1]
                if classification
                else None
            )
            results[name][split] = score_predictions(
                parts[split][target].to_numpy(), predicted, probability
            )
        preprocessor = pipeline.named_steps["preprocess"]
        diagnostics[name] = {
            "transformed_features": preprocessor.get_feature_names_out().tolist(),
            "train_medians": preprocessor.named_transformers_["numeric"]
            .named_steps["imputer"]
            .statistics_.tolist(),
            "train_scaling_mean": preprocessor.named_transformers_["numeric"]
            .named_steps["scale"]
            .mean_.tolist(),
        }
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, part in parts.items():
        part.select(
            "row_id",
            "original_fingerprint",
            *numeric,
            "total_guests",
            *categorical,
            target,
        ).write_parquet(processed_dir / f"{task}_{split}.parquet")
    fingerprints = {
        name: set(part["original_fingerprint"]) for name, part in parts.items()
    }
    overlaps = {
        f"{a}_{b}": len(fingerprints[a] & fingerprints[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    return {
        "population_rows": frame.height,
        "target": target,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "splits": {
            name: {
                "rows": part.height,
                "fraction": part.height / frame.height,
                "cancellation_prevalence": float(part["is_canceled"].to_numpy().mean()),
                "unique_original_fingerprints": part["original_fingerprint"].n_unique(),
                "group_sizes": group_size_summary(part),
            }
            for name, part in parts.items()
        },
        "fingerprint_overlap": overlaps,
        "x_pattern_overlap": {
            f"{a}_{b}": len(
                set(x(parts[a]).hash_rows(seed=SEED))
                & set(x(parts[b]).hash_rows(seed=SEED))
            )
            for a, b in [
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ]
        },
        "x_overlap_interpretation": "Different original reservations can share conservative X values. These are diagnosed, not removed or grouped by target-erased predictors. No exact original row crosses partitions.",
        "redundancy_policy": "Keep adults, children and babies in X so known adults survive missing children; median-impute children on train only. Exclude total_guests from X: its exact sum is redundant when complete and its unknown total must not be independently imputed. Preserve total_guests (null when any component is unknown) in cleaned and task/split Parquet for traceability, not as a model predictor. Keep total_nights instead of its night components. Full OHE is redundant with intercept; LinearRegression uses least squares/SVD, not coefficient inference. Logistic uses default L2; no advanced regularized regression or tuning.",
        "split_policy": "Target row weights 75/12.5/12.5 within each class (classification) or overall (regression); original-fingerprint groups sorted largest first with seed42 random tie-breaks, then assigned to greatest relative row deficit among bins where the group fits. Ties prefer larger target then fixed split order; group-size ties use seed42. If no bin fits, minimize the increase in squared relative target error. Validate train row fraction 70%-80% and nonempty holdouts before fitting/exporting. Intact groups can prevent exact target sizes; greedy assignment is not an exhaustive feasibility solver.",
        "imbalance": {
            "train_prevalence": prevalence,
            "severe_threshold_minority": 0.10,
            "severe": severe if classification else False,
            "treatment": "class_weight=balanced"
            if classification and severe
            else "No resampling: minority is not severe (threshold 10%); regression conditioned on non-cancellation.",
        },
        "metrics": results,
        "transforms": diagnostics,
    }
