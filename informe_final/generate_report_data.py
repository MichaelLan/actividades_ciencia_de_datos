"""Build the data file and the figures of the final (Phase 4) report.

Run from the ``dashboard`` folder so the shipped artifacts and its uv environment are used::

    cd dashboard
    uv run --with matplotlib python ../informe_final/generate_report_data.py

Inputs (read only): Phase 1 EDA JSON, Phase 2 cleaning/partition JSON, the saved outputs of
the executed Phase 3 notebook (``fase_3/Fase_3_Hotel_Booking.ipynb``), dashboard
``metrics.json``, ``test_metrics.json``, ``validation_predictions.parquet`` and
``data/bookings.parquet``; Fig. 1 is rebuilt from the raw
``preprocesamiento/data/raw/hotel_bookings.csv`` with the Phase 1 bins and limits. Outputs: ``informe_final/datos.json`` (numbers injected in the
Typst sources) and SVG/PNG figures in ``informe_final/imagenes``. No model is fitted here.
"""

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
DASHBOARD = ROOT / "dashboard"
FIGURES = REPORT / "imagenes"
PHASE3_NOTEBOOK = ROOT / "fase_3" / "Fase_3_Hotel_Booking.ipynb"

# Palette shared with dashboard/hotel_dashboard/theme.py (fixed categorical order).
BLUE, ORANGE, GREEN, YELLOW, PINK, DARK_GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
TEXT, TEXT_2, MUTED, GRID, AXIS = "#1f1e1c", "#52514e", "#898781", "#ebeae4", "#c3c2b7"
SEGMENT_ORDER = ["short_lead", "long_lead", "children", "baby", "loyal", "serial_cancellers"]
SEGMENT_NAMES = {
    "short_lead": "Corto plazo",
    "long_lead": "Anticipada",
    "children": "Familias con niños",
    "baby": "Familias con bebé",
    "loyal": "Clientes fieles",
    "serial_cancellers": "Historial de cancel.",
}
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, [BLUE, ORANGE, GREEN, YELLOW, PINK, DARK_GREEN], strict=True))
FEATURE_NAMES = {
    "lead_time": "Anticipación",
    "adults": "Adultos",
    "children": "Niños",
    "babies": "Bebés",
    "total_nights": "Noches",
    "previous_cancellations": "Cancel. previas",
    "previous_bookings_not_canceled": "Reservas prev. mant.",
    "hotel": "Hotel",
    "arrival_date_month": "Mes de llegada",
    "meal": "Régimen",
    "market_segment": "Segmento mercado",
    "distribution_channel": "Canal",
    "is_repeated_guest": "Cliente repetido",
    "reserved_room_type": "Habitación reserv.",
    "customer_type": "Tipo de cliente",
}
# Phase 3 values that the notebook prints only inside rendered tables or figures (MLP table,
# confusion matrices, permutation tables); copied from the saved outputs of
# fase_3/Fase_3_Hotel_Booking.ipynb so the Typst sources never hand-type them. The printed
# dictionaries (CV/validation summaries, K-Means and DBSCAN selections) are parsed below.
PHASE3_REPORT = {
    "cv_folds": 3,
    "configurations_per_model": 3,
    "fold_cancel_rate": [0.359, 0.382],
    "mlp": {
        "best_auc": 0.810,
        "hidden": [128, 64],
        "best_val_loss": 0.481,
        "best_epoch": 13,
        "candidates": [[64, 32], [128, 64], [128, 64, 32]],
        "dropout": [0.20, 0.25, 0.35],
        "max_epochs": 60,
        "patience": 8,
        "batch": 512,
    },
    "rf_confusion": {"tp": 2241, "fp": 345, "fn": 3281},
    "xgb_confusion": {"tp": 2494, "fp": 599, "fn": 3028},
    "permutation": {"rows": 800, "repeats": 3, "adr_month": 20.18, "cancel_customer_type": 0.093},
    "kmeans": {"best_silhouette_k": 4, "best_silhouette": 0.552, "ari_k4": 0.512, "ari_k6": 0.992, "seeds": 3},
    "dbscan": {"rows": 5000, "eps": 1.438558, "min_samples": 20, "clusters": 5, "assigned": 4851, "noise": 149},
}
COLUMN_WIDTH = 3.45  # inches, IEEE single column


def style() -> None:
    """Light, recessive matplotlib style sized for an IEEE column."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.dpi": 150,
        }
    )


def decimal_comma(fig) -> None:
    """Replace the decimal point of numeric tick labels with a comma (Spanish convention)."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))


def save(fig, name: str) -> None:
    """Write an SVG figure (decimal commas) and close it."""
    decimal_comma(fig)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_outputs(path: Path = PHASE3_NOTEBOOK) -> dict:
    """Parse the dictionaries printed by the executed Phase 3 notebook (no code is run).

    Returns the cross-validation and validation summaries of the four searched models and the
    K-Means/DBSCAN selection records, read from the saved text outputs.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(output.get("text", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    )
    literal = lambda raw: ast.literal_eval(raw)  # noqa: E731
    models = {
        name: {"cv": literal(cv), "validation": literal(validation)}
        for name, cv, validation in re.findall(r"^(\w+) CV: (\{.*\})\nValidation: (\{.*\})$", text, re.M)
    }
    for summary in models.values():
        summary["validation"].pop("confusion_matrix", None)
    return {
        "source": str(path.relative_to(ROOT)),
        "models": models,
        "kmeans_selection": literal(re.search(r"^Selección K-Means: (\{.*\})$", text, re.M).group(1)),
        "dbscan_eps_candidates": literal(re.search(r"^Valores eps candidatos: (\[.*\])$", text, re.M).group(1)),
        "dbscan_selection": literal(re.search(r"^Selección DBSCAN piloto: (\{.*\})$", text, re.M).group(1)),
    }


def build_data() -> dict:
    """Collect the numbers cited in the report into one JSON document."""
    eda = read_json(ROOT / "preprocesamiento/outputs/eda.json")
    cleaning = read_json(ROOT / "preprocesamiento/outputs/cleaning.json")
    phase2 = read_json(ROOT / "preprocesamiento/outputs/phase2.json")
    metrics = read_json(DASHBOARD / "artifacts/metrics.json")
    test = read_json(DASHBOARD / "artifacts/test_metrics.json")
    for task in ("regression", "classification"):
        for split in phase2["tasks"][task]["splits"].values():
            split.pop("group_sizes", None)
        phase2["tasks"][task].pop("transforms", None)
    metrics["classification"].pop("roc", None)
    bookings = pl.read_parquet(DASHBOARD / "data/bookings.parquet")
    value = pl.col("adr") * pl.col("total_nights")
    kept = pl.col("is_canceled") == 0
    business = bookings.select(
        pl.len().alias("bookings"),
        value.filter(kept).sum().alias("revenue"),
        value.filter(~kept).sum().alias("lost_revenue"),
        pl.col("adr").filter(kept).mean().alias("mean_adr_kept"),
    ).row(0, named=True)
    business["lost_share"] = business["lost_revenue"] / (business["revenue"] + business["lost_revenue"])
    predictions = pl.read_parquet(DASHBOARD / "artifacts/validation_predictions.parquet")
    business["mean_adr_validation"] = float(predictions.filter(pl.col("task") == "regression")["adr"].mean())
    return {
        "phase3_report": PHASE3_REPORT,
        "phase3_notebook": notebook_outputs(),
        "business": business,
        "eda": {
            "quality": {k: eda["quality"][k] for k in ("rows", "columns", "duplicates_extra", "missing")},
            "anomalies": eda["quality"]["anomalies"],
            "coverage": eda["coverage"],
            "adr": eda["quantitative"]["adr"],
            "lead_time": eda["quantitative"]["lead_time"],
            "hotel": eda["relationships"]["hotel"],
            "lead_time_bins": eda["lead_time_bins"],
            "spearman_lead_cancel": eda["correlations"]["spearman"][0][-1],
            "chi_deposit": eda["chi_square"]["deposit_type"],
        },
        "cleaning": {k: v for k, v in cleaning.items() if k != "rejected"},
        "phase2": phase2,
        "metrics": metrics,
        "test": test,
    }


def figure_distributions() -> None:
    """Phase 1 distribution figure (same raw data, bins and limits) with the report labels and palette."""
    frame = pl.read_csv(
        ROOT / "preprocesamiento/data/raw/hotel_bookings.csv",
        null_values=["NULL", "NA", ""],
        infer_schema_length=10000,
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), layout="constrained")
    for ax, name, limit, label in zip(
        axes[:2], ("adr", "lead_time"), (300, 500), ("ADR (u.m.)", "Anticipación (días)"), strict=True
    ):
        values = frame[name].drop_nulls().to_numpy()
        ax.hist(values, bins=np.linspace(0, limit, 41), color=BLUE)
        ax.set(xlabel=label, ylabel="Reservas", xlim=(0, limit))
    hotels = ("City Hotel", "Resort Hotel")
    boxes = axes[2].boxplot(
        [frame.filter(pl.col("hotel") == h)["adr"].to_numpy() for h in hotels],
        tick_labels=("Urbano", "Vacacional"),
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": TEXT, "lw": 1.0},
        whiskerprops={"color": TEXT_2, "lw": 0.8},
        capprops={"color": TEXT_2, "lw": 0.8},
    )
    for patch, color in zip(boxes["boxes"], (BLUE, ORANGE), strict=True):
        patch.set(facecolor=color, edgecolor=TEXT_2, alpha=0.85, lw=0.8)
    axes[2].set(xlabel="Hotel", ylabel="ADR (u.m.)")
    axes[2].grid(axis="x", visible=False)
    save(fig, "distributions")


def figure_pipeline() -> None:
    """Full-width flow diagram of the project lifecycle."""
    steps = [
        ("Fase 1", "Datos y EDA", "119 390 × 32\ncalidad, asociaciones"),
        ("Fase 2", "Preparación", "reglas de dominio\nparticiones por huella"),
        ("Fase 3", "Modelado", "RF, ElasticNet, XGBoost\nMLP, K-Means, DBSCAN"),
        ("Fase 4", "Producto", "reentrenamiento,\nprueba única, tablero"),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 0.95))
    ax.set_axis_off()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    for index, (phase, title, detail) in enumerate(steps):
        x = index + 0.06
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.06),
                0.84,
                0.88,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor="#f6f6f3",
                edgecolor=AXIS,
                linewidth=0.8,
            )
        )
        ax.text(x + 0.42, 0.78, phase, ha="center", va="center", color=BLUE, fontsize=7, weight="bold")
        ax.text(x + 0.42, 0.57, title, ha="center", va="center", color=TEXT, fontsize=8, weight="bold")
        ax.text(x + 0.42, 0.27, detail, ha="center", va="center", color=TEXT_2, fontsize=6.8, linespacing=1.3)
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(index + 1.05, 0.5),
                xytext=(x + 0.86, 0.5),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.9},
            )
    save(fig, "pipeline")


def figure_models(data: dict) -> None:
    """Validation comparison: ADR RMSE (left) and cancellation ROC AUC (right)."""
    reg = data["metrics"]["regression"]["metrics"]
    ref = data["metrics"]["phase3_reference"]
    test = data["test"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.9), layout="constrained")

    names = ["Constante", "Lineal", "ElasticNet*", "R. Forest"]
    values = [
        reg["dummy"]["validation"]["rmse"],
        reg["linear"]["validation"]["rmse"],
        ref["regression"]["ElasticNet"]["rmse"],
        reg["random_forest"]["validation"]["rmse"],
    ]
    colors = [AXIS, AXIS, AXIS, BLUE]
    ax = axes[0]
    ax.barh(names, values, color=colors, height=0.62)
    for y, v in enumerate(values):
        ax.text(v + 0.8, y, f"{v:.1f}".replace(".", ","), va="center", fontsize=6.8, color=TEXT)
    ax.axvline(test["regression"]["random_forest"]["rmse"], color=ORANGE, lw=1, ls="--")
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("RMSE de ADR (validación)")
    ax.grid(axis="y", visible=False)
    ax.set_title("a) Tarifa", loc="left")

    cls = ref["classification"]
    names = ["Logística", "R. Forest", "XGBoost*", "MLP peq.*", "MLP bal.*", "MLP reg.*"]
    keys = ["Logística", "Random Forest", "XGBoost", "MLP pequeña", "MLP balanceada", "MLP regularizada"]
    recomputed = data["metrics"]["classification"]["metrics"]
    values = [cls[k]["roc_auc"] for k in keys]
    values[0] = recomputed["logistic"]["validation"]["roc_auc"]
    values[1] = recomputed["random_forest"]["validation"]["roc_auc"]
    colors = [AXIS, BLUE, AXIS, AXIS, AXIS, AXIS]
    ax = axes[1]
    ax.barh(names, values, color=colors, height=0.62)
    for y, v in enumerate(values):
        ax.text(v + 0.003, y, f"{v:.3f}".replace(".", ","), va="center", fontsize=6.8, color=TEXT)
    ax.axvline(test["classification"]["random_forest"]["roc_auc"], color=ORANGE, lw=1, ls="--")
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.86)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("AUC de cancelación (validación)")
    ax.grid(axis="y", visible=False)
    ax.set_title("b) Cancelación", loc="left")
    save(fig, "model_comparison")


def figure_overfitting(data: dict) -> None:
    """Train, validation and test scores of the selected Random Forests."""
    reg = data["metrics"]["regression"]["metrics"]["random_forest"]
    cls = data["metrics"]["classification"]["metrics"]["random_forest"]
    test = data["test"]
    groups = ["R² ADR", "AUC cancel.", "F1 cancel."]
    series = {
        "Entrenamiento": [reg["train"]["r2"], cls["train"]["roc_auc"], cls["train"]["f1"]],
        "Validación": [reg["validation"]["r2"], cls["validation"]["roc_auc"], cls["validation"]["f1"]],
        "Prueba": [
            test["regression"]["random_forest"]["r2"],
            test["classification"]["random_forest"]["roc_auc"],
            test["classification"]["random_forest"]["f1"],
        ],
    }
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.65), layout="constrained")
    width = 0.26
    x = np.arange(len(groups))
    for offset, (name, color) in zip(
        (-width, 0, width), (("Entrenamiento", MUTED), ("Validación", BLUE), ("Prueba", ORANGE)), strict=True
    ):
        values = series[name]
        ax.bar(x + offset, values, width * 0.92, color=color, label=name)
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(xi, v + 0.015, f"{v:.2f}".replace(".", ","), ha="center", fontsize=6.2, color=TEXT)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "overfitting")


def figure_adr_predictions() -> None:
    """Observed vs predicted ADR (validation) and residual distribution."""
    pred = pl.read_parquet(DASHBOARD / "artifacts/validation_predictions.parquet").filter(
        pl.col("task") == "regression"
    )
    observed = pred["adr"].to_numpy()
    predicted = pred["adr_random_forest"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.75), layout="constrained")
    ax = axes[0]
    hb = ax.hexbin(predicted, observed, gridsize=38, extent=(0, 300, 0, 300), bins="log", cmap="Blues", mincnt=1)
    hb.set_edgecolor("face")
    ax.plot([0, 300], [0, 300], color=ORANGE, lw=0.9)
    ax.set(xlim=(0, 300), ylim=(0, 300), xlabel="ADR predicha", ylabel="ADR observada")
    ax.set_title("a) Validación", loc="left")
    ax = axes[1]
    residual = observed - predicted
    ax.hist(residual, bins=np.linspace(-120, 120, 49), color=BLUE)
    ax.axvline(0, color=TEXT_2, lw=0.7)
    ax.set(xlabel="Residuo (obs. − pred.)", ylabel="Reservas")
    ax.set_title("b) Residuos", loc="left")
    save(fig, "adr_predictions")


def figure_classification(data: dict) -> None:
    """ROC curves (validation) and confusion matrix of the Random Forest at 0.50."""
    metrics = read_json(DASHBOARD / "artifacts/metrics.json")["classification"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.75), layout="constrained")
    ax = axes[0]
    for key, label, color in (("random_forest", "Random Forest", BLUE), ("logistic", "Logística", MUTED)):
        roc = metrics["roc"][key]
        auc = metrics["metrics"][key]["validation"]["roc_auc"]
        ax.plot(roc["fpr"], roc["tpr"], color=color, lw=1.4, label=f"{label} ({auc:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, ls="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.2)
    ax.set_title("a) ROC", loc="left")
    ax = axes[1]
    matrix = np.array(metrics["metrics"]["random_forest"]["validation"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.3)
    for (i, j), v in np.ndenumerate(matrix):
        color = "white" if v > matrix.max() * 0.6 else TEXT
        ax.text(j, i, f"{v:,}".replace(",", " "), ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks([0, 1], ["Mantenida", "Cancelada"])
    ax.set_yticks([0, 1], ["Mantenida", "Cancelada"], rotation=90, va="center")
    ax.set(xlabel="Predicha", ylabel="Real")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("b) Umbral 0,50", loc="left")
    save(fig, "classification")


def figure_importance(data: dict) -> None:
    """Permutation importance (top features) for both supervised tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.95), layout="constrained")
    specs = (
        ("regression", "a) ADR: aumento de RMSE", 1.0),
        ("classification", "b) Cancel.: caída de AUC", 1.0),
    )
    for ax, (task, title, _) in zip(axes, specs, strict=True):
        rows = data["metrics"][task]["permutation_importance"][:7]
        names = [FEATURE_NAMES.get(r["feature"], r["feature"]) for r in rows]
        means = [r["mean"] for r in rows]
        stds = [r["std"] for r in rows]
        ax.barh(names, means, xerr=stds, color=BLUE, height=0.6, error_kw={"ecolor": TEXT_2, "lw": 0.7})
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", labelsize=6.5)
    save(fig, "importance")


def figure_k_selection(data: dict) -> None:
    """Elbow (inertia), silhouette and seed stability for K = 2..10 on the training partition."""
    rows = data["metrics"]["clustering"]["k_selection"]
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.55), layout="constrained")
    panels = (
        ([r["inertia"] / 1e3 for r in rows], "Inercia (miles)", "a) Codo"),
        ([r["silhouette"] for r in rows], "Silueta", "b) Silueta"),
        ([r["stability_ari_mean"] for r in rows], "ARI medio", "c) Estabilidad"),
    )
    for ax, (values, label, title) in zip(axes, panels, strict=True):
        ax.plot(ks, values, color=BLUE, marker="o", ms=2.5, lw=1.2)
        ax.set(xlabel="K", ylabel=label)
        ax.set_title(title, loc="left")
        ax.axvline(6, color=ORANGE, lw=0.9, ls="--")
        ax.set_xticks(ks[::2])
    axes[2].axhline(0.85, color=MUTED, lw=0.7, ls=":")
    axes[2].set_ylim(0, 1.05)
    save(fig, "k_selection")


def figure_pca() -> None:
    """PCA map of a stratified sample of the cleaned bookings coloured by segment."""
    frame = pl.read_parquet(DASHBOARD / "data/bookings.parquet")
    parts = []
    for segment in SEGMENT_ORDER:
        part = frame.filter(pl.col("segment") == segment)
        parts.append(part.sample(n=min(part.height, 2500), seed=42))
    sample = pl.concat(parts)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.35), layout="constrained")
    for segment in SEGMENT_ORDER:
        part = sample.filter(pl.col("segment") == segment)
        ax.scatter(
            part["pc1"],
            part["pc2"],
            s=4,
            color=SEGMENT_COLORS[segment],
            alpha=0.55,
            linewidths=0,
            label=SEGMENT_NAMES[segment],
            rasterized=True,
        )
    ax.set(xlabel="CP1", ylabel="CP2")
    ax.legend(markerscale=2.5, loc="upper left", fontsize=6.3, handletextpad=0.2)
    save(fig, "pca_clusters")


def main() -> None:
    """Write datos.json, copy reused Phase 1 figures and render every new figure."""
    FIGURES.mkdir(exist_ok=True)
    style()
    data = build_data()
    (REPORT / "datos.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    figure_distributions()
    figure_pipeline()
    figure_models(data)
    figure_overfitting(data)
    figure_adr_predictions()
    figure_classification(data)
    figure_importance(data)
    figure_k_selection(data)
    figure_pca()
    print(f"Datos y figuras escritos en {REPORT}")


if __name__ == "__main__":
    main()
