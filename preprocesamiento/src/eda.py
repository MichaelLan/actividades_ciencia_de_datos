"""Full original-data exploration; no model fitting or cleaning."""

import calendar
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import chi2_contingency

from src.data import CATEGORICAL, COLUMNS, write_json

QUANTITATIVE = [
    name for name in COLUMNS if name not in CATEGORICAL + ["reservation_status_date"]
]


def quantitative_summary(series: pl.Series) -> dict[str, Any]:
    series = series.drop_nulls()
    if series.is_empty():
        return {
            "valid": 0,
            "mode": [],
            "iqr_outliers": 0,
            **dict.fromkeys(
                [
                    "mean",
                    "median",
                    "std",
                    "variance",
                    "min",
                    "max",
                    "q1",
                    "q3",
                    "iqr",
                    "skewness",
                    "kurtosis",
                ]
            ),
        }
    q1 = series.quantile(0.25, interpolation="linear")
    q3 = series.quantile(0.75, interpolation="linear")
    assert q1 is not None and q3 is not None
    iqr = q3 - q1
    return {
        "valid": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().sort().to_list(),
        "std": series.std(ddof=1),
        "variance": series.var(ddof=1),
        "min": series.min(),
        "max": series.max(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": series.skew(bias=False)
        if len(series) > 2 and series.n_unique() > 1
        else None,
        "kurtosis": series.kurtosis(fisher=True, bias=False)
        if len(series) > 3 and series.n_unique() > 1
        else None,
        "iqr_outliers": int(
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        ),
    }


def chi_square(frame: pl.DataFrame, column: str) -> dict:
    table = (
        frame.with_columns(pl.col(column).cast(pl.String).fill_null("__MISSING__"))
        .group_by(column)
        .agg(
            (pl.col("is_canceled") == 0).sum().alias("no"),
            (pl.col("is_canceled") == 1).sum().alias("yes"),
        )
        .sort(column)
    )
    observed = table.select("no", "yes").to_numpy()
    if min(observed.shape) < 2 or (observed.sum(axis=0) == 0).any():
        return {"not_testable": True}
    statistic, p_value, dof, expected = chi2_contingency(observed, correction=False)
    return {
        "chi2": float(statistic),
        "p_value": float(p_value),
        "dof": dof,
        "cramers_v": float(
            np.sqrt(
                statistic
                / (observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1))
            )
        ),
        "expected_min": float(expected.min()),
        "expected_below_5_percent": float((expected < 5).mean() * 100),
        "assumptions_met": bool(expected.min() >= 1 and (expected < 5).mean() <= 0.2),
        "independence_caveat": "Repeated/group bookings may violate independent observations. Exploratory association, not causality; no multiplicity correction.",
        "table": table.to_dicts(),
    }


def arrival_dates(frame: pl.DataFrame) -> pl.DataFrame:
    months = {month: index for index, month in enumerate(calendar.month_name) if month}
    return frame.with_columns(
        pl.date(
            pl.col("arrival_date_year"),
            pl.col("arrival_date_month").replace_strict(
                months, default=None, return_dtype=pl.Int64
            ),
            pl.col("arrival_date_day_of_month"),
        ).alias("arrival_date")
    )


def explore(frame: pl.DataFrame, output_dir: Path, figures_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    missing = {
        name: {
            "count": frame[name].null_count(),
            "percent": frame[name].null_count() / frame.height * 100,
        }
        for name in COLUMNS
    }
    numeric = {name: quantitative_summary(frame[name]) for name in QUANTITATIVE}
    categorical = {}
    for name in CATEGORICAL:
        categorical[name] = (
            frame.select(
                pl.col(name).cast(pl.String).fill_null("__MISSING__").alias("value")
            )
            .group_by("value")
            .len(name="count")
            .with_columns((pl.col("count") / frame.height).alias("relative"))
            .sort(["count", "value"], descending=[True, False])
            .to_dicts()
        )
    dated = arrival_dates(frame).with_columns(
        pl.col("reservation_status_date")
        .str.to_date(strict=False)
        .alias("status_date"),
        (pl.col("stays_in_weekend_nights") + pl.col("stays_in_week_nights")).alias(
            "total_nights"
        ),
    )
    extreme = numeric["adr"]["q3"] + 1.5 * numeric["adr"]["iqr"]
    conditions = {
        "negative_adr": pl.col("adr") < 0,
        "zero_adr": pl.col("adr") == 0,
        "extreme_adr_upper_iqr": pl.col("adr") > extreme,
        "zero_nights": pl.col("total_nights") == 0,
        "zero_guests": pl.sum_horizontal("adults", "children", "babies") == 0,
        "status_after_arrival": pl.col("status_date") > pl.col("arrival_date"),
        "checkout_before_arrival": (pl.col("reservation_status") == "Check-Out")
        & (pl.col("status_date") < pl.col("arrival_date")),
        "unparseable_status_date": pl.col("status_date").is_null(),
    }
    anomalies = {
        name: dated.filter(condition).height for name, condition in conditions.items()
    }
    absence = {}
    for name in COLUMNS:
        if missing[name]["count"]:
            absence[name] = {
                group: frame.group_by(group)
                .agg(
                    pl.len().alias("rows"),
                    pl.col(name).is_null().sum().alias("missing"),
                    (pl.col(name).is_null().mean() * 100).alias("missing_percent"),
                )
                .sort(group)
                .to_dicts()
                for group in ["hotel", "is_canceled", "market_segment"]
            }
    corr_columns = [
        name for name in QUANTITATIVE if not name.startswith("arrival_date")
    ] + ["is_canceled"]
    correlations = {
        "columns": corr_columns,
        "rationale": "Pearson describes linear relationships; Spearman is robust to skew/tails and monotonic nonlinearity. Pairwise complete observations; binary cancellation is descriptive. Identifiers and calendar components excluded.",
        "pearson": [],
        "spearman": [],
    }
    for method in ["pearson", "spearman"]:
        correlations[method] = [
            [frame.select(pl.corr(a, b, method=method)).item() for b in corr_columns]
            for a in corr_columns
        ]
    dictionary = {}
    for name in COLUMNS:
        categorical_variable = name in CATEGORICAL
        dictionary[name] = {
            "dtype": str(frame.schema[name]),
            "statistical_type": "nominal"
            if categorical_variable
            else "calendar/date"
            if name.startswith("arrival_date") or name == "reservation_status_date"
            else "continuous"
            if name == "adr"
            else "discrete",
            "observed_domain": frame[name].drop_nulls().unique().sort().to_list()
            if categorical_variable or name == "reservation_status_date"
            else {"min": frame[name].min(), "max": frame[name].max()},
            "null_count": missing[name]["count"],
        }
    result = {
        "quality": {
            "rows": frame.height,
            "columns": frame.width,
            "missing": missing,
            "duplicates_extra": frame.height - frame.unique().height,
            "duplicate_member_rows": int(frame.is_duplicated().sum()),
            "anomalies": anomalies,
            "anomaly_definitions": {
                "extreme_adr_upper_iqr": f"ADR > {extreme}; exploratory flag, not an exclusion",
                "zero_guests": "adults + known children + babies == 0; unknown children counted as zero only for flagging",
                "status_after_arrival": "Descriptive, normally valid for checkout; never automatically excluded",
                "checkout_before_arrival": "Check-Out status recorded before scheduled arrival",
            },
        },
        "quantitative": numeric,
        "categorical": categorical,
        "dictionary": dictionary,
        "coverage": {
            "arrival_min": str(dated["arrival_date"].min()),
            "arrival_max": str(dated["arrival_date"].max()),
            "hotel_counts": frame.group_by("hotel").len().sort("hotel").to_dicts(),
            "cancellation_prevalence": frame["is_canceled"].mean(),
        },
        "missingness_by_group": absence,
        "missingness_interpretation": "Observed group differences can suggest MAR conditional on recorded features; MCAR is not established or formally rejected. Per Antonio et al. table 1, NULL agent/company means not applicable, not unknown identifiers: never median-impute them. Unknown children are median-imputed on train; country is excluded from models (may be recorded at check-in). MNAR for unknown values cannot be excluded. Sensitivity to missingness remains a limitation.",
        "relationships": {
            name: dated.group_by(name)
            .agg(
                pl.len().alias("rows"),
                pl.col("adr").mean().alias("adr_mean"),
                pl.col("adr").median().alias("adr_median"),
                pl.col("is_canceled").mean().alias("cancellation_rate"),
            )
            .sort(name)
            .to_dicts()
            for name in [
                "hotel",
                "market_segment",
                "reserved_room_type",
                "arrival_date_month",
                "total_nights",
            ]
        },
        "lead_time_bins": dated.with_columns(
            pl.when(pl.col("lead_time") <= 7)
            .then(pl.lit("00: <=7"))
            .when(pl.col("lead_time") <= 30)
            .then(pl.lit("01: 8-30"))
            .when(pl.col("lead_time") <= 90)
            .then(pl.lit("02: 31-90"))
            .when(pl.col("lead_time") <= 180)
            .then(pl.lit("03: 91-180"))
            .when(pl.col("lead_time") <= 365)
            .then(pl.lit("04: 181-365"))
            .otherwise(pl.lit("05: >365"))
            .alias("lead_bin")
        )
        .group_by("lead_bin")
        .agg(
            pl.len().alias("rows"),
            pl.col("adr").mean().alias("adr_mean"),
            pl.col("total_nights").mean().alias("mean_total_nights"),
            pl.col("is_canceled").mean().alias("cancellation_rate"),
        )
        .sort("lead_bin")
        .to_dicts(),
        "correlations": correlations,
        "chi_square": {
            name: chi_square(frame, name)
            for name in [
                "hotel",
                "meal",
                "market_segment",
                "distribution_channel",
                "deposit_type",
                "customer_type",
                "reserved_room_type",
            ]
        },
        "statistics_conventions": "Sample SD/variance ddof=1, unbiased skewness, excess Fisher kurtosis; linear quartiles, all tied modes; missing excluded from quantitative summaries; categorical relative frequencies include missing. Calendar integer summaries are descriptive coding ranges, not interval-scale effects; excluded from correlation. Raw observations, not causal estimates.",
    }
    write_json(output_dir / "eda.json", result)
    plot_figures(frame, dated, result, figures_dir)
    return result


def plot_figures(frame, dated, result, directory):
    # A ancho completo (~7 pulgadas), incluso los ticks deben conservar >=7 pt.
    plt.rcParams.update(
        {
            "font.size": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), layout="constrained")
    for ax, name in zip(
        axes.flat, ["lead_time", "adr", "stays_in_week_nights", "adults"], strict=True
    ):
        ax.hist(frame[name].drop_nulls().to_numpy(), bins=60, color="#277e8e")
        ax.set(
            xlabel=name,
            ylabel="Reservas",
            yscale="log",
            title="Distribución original (conteo log)",
        )
        # Evitar exponentes matemáticos reducidos en los ticks de escala logarítmica.
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    save(fig, "distributions")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    image = ax.imshow(
        result["correlations"]["spearman"], vmin=-1, vmax=1, cmap="RdBu_r"
    )
    names = result["correlations"]["columns"]
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Correlación de Spearman · original")
    fig.colorbar(image, ax=ax, shrink=0.7)
    save(fig, "correlations")
    sample = frame.sample(n=min(4000, frame.height), seed=42).with_columns(
        (pl.col("stays_in_weekend_nights") + pl.col("stays_in_week_nights")).alias(
            "total_nights"
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout="constrained")
    for ax, name in zip(axes, ["lead_time", "total_nights", "adults"], strict=True):
        ax.scatter(
            sample[name],
            sample["adr"],
            c=sample["is_canceled"],
            cmap="coolwarm",
            s=3,
            alpha=0.3,
        )
        ax.set(xlabel=name, ylabel="ADR", title="Muestra n=4000\nRojo: cancelada")
    save(fig, "relationships")
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    hotels = sorted(frame["hotel"].unique().to_list())
    ax.boxplot(
        [frame.filter(pl.col("hotel") == name)["adr"].to_numpy() for name in hotels],
        tick_labels=hotels,
        showfliers=False,
    )
    ax.set(ylabel="ADR", title="ADR por hotel · bigotes 1,5 IQR; atípicos ocultos")
    save(fig, "adr_by_hotel")
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), layout="constrained")
    for ax, name in zip(axes, ["hotel", "market_segment", "deposit_type"], strict=True):
        table = (
            frame.group_by(name)
            .agg(pl.col("is_canceled").mean().alias("rate"), pl.len().alias("n"))
            .sort(name)
        )
        ax.barh(
            [f"{row[name]} (n={row['n']})" for row in table.to_dicts()],
            table["rate"] * 100,
        )
        ax.set(xlabel="Cancelación (%)", title=name, xlim=(0, 100))
    save(fig, "cancellation_categories")
    table = (
        dated.with_columns(pl.col("arrival_date").dt.truncate("1mo").alias("month"))
        .group_by("month", "hotel")
        .len()
        .sort("month")
    )
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    for hotel in hotels:
        group = table.filter(pl.col("hotel") == hotel)
        ax.plot(group["month"].to_list(), group["len"], label=hotel)
    ax.legend()
    ax.set(
        title="Llegadas por mes · extremos temporales incompletos",
        ylabel="Reservas",
        xlabel="Mes de llegada",
    )
    save(fig, "arrivals")
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    missing = {
        name: value["percent"]
        for name, value in result["quality"]["missing"].items()
        if value["count"]
    }
    ax.barh(list(missing), list(missing.values()))
    ax.set(xlabel="Ausencia (%)", title="Nulos en el original", xlim=(0, 100))
    save(fig, "missingness")
