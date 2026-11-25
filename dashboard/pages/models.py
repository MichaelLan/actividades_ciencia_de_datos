"""Models and methodology: metrics vs baselines and Phase 3, importance and diagnostics."""

import numpy as np
import polars as pl
import streamlit as st

from hotel_dashboard import charts, config
from hotel_dashboard.data import load_metrics, load_validation_predictions
from hotel_dashboard.kpis import fmt_number, fmt_pct
from hotel_dashboard.ui import card, chip, header, kpi, plot, section

metrics = load_metrics()
predictions = load_validation_predictions()
reg, cls, clu = metrics["regression"], metrics["classification"], metrics["clustering"]
phase3 = metrics["phase3_reference"]

MODEL_NAMES = {
    "dummy": "Referencia constante",
    "linear": "Regresión lineal",
    "logistic": "Regresión logística",
    "random_forest": "Random Forest",
}
DECIMALS = {"mae": 2, "rmse": 2}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "accuracy": "Exactitud",
    "precision": "Precisión",
    "recall": "Sensibilidad",
    "f1": "F1",
    "roc_auc": "AUC",
}


def metric_table(task_metrics: dict, keys: list[str], splits: dict[str, list[str]]) -> pl.DataFrame:
    """Metrics per model; ``splits`` maps split name to the metric keys shown for it."""
    tags = {"validation": "valid.", "train": "entren."}
    rows = []
    for model, values in task_metrics.items():
        row = {"Modelo": MODEL_NAMES[model]}
        for split, split_keys in splits.items():
            row |= {
                f"{METRIC_LABELS[key]} ({tags[split]})": fmt_number(values[split][key], DECIMALS.get(key, 3))
                for key in keys
                if key in split_keys
            }
        rows.append(row)
    return pl.DataFrame(rows)


def reference_table(reference: dict) -> pl.DataFrame:
    """Phase 3 reported validation metrics as a table."""
    return pl.DataFrame(
        [
            {
                "Modelo (Fase 3)": name,
                **{METRIC_LABELS[k]: fmt_number(v, DECIMALS.get(k, 3)) for k, v in values.items()},
            }
            for name, values in reference.items()
        ]
    )


header(
    "Modelos y desempeño",
    "Métricas en validación · configuraciones ganadoras de la Fase 3",
    "insights",
    [chip("Prueba usada una sola vez", "lock"), chip("Reproduce la Fase 3", "verified")],
)


def signed(value: float, decimals: int) -> str:
    """Number with an explicit sign."""
    return ("+" if value > 0 else "") + fmt_number(value, decimals)


def comparison(task_metrics: dict, key: str, decimals: int, title: str) -> None:
    """Bar chart of one validation metric across the trained models."""
    values = {MODEL_NAMES[m]: v["validation"][key] for m, v in task_metrics.items()}
    plot(charts.model_comparison(values, MODEL_NAMES["random_forest"], decimals, title), key=f"cmp_{key}")


adr_tab, cancel_tab, cluster_tab, method_tab = st.tabs(
    [
        ":material/sell: Tarifa (ADR)",
        ":material/event_busy: Cancelación",
        ":material/bubble_chart: Segmentación",
        ":material/menu_book: Metodología",
    ]
)

with adr_tab:
    rf = reg["metrics"]["random_forest"]["validation"]
    lin = reg["metrics"]["linear"]["validation"]
    with st.container(key="krow-adr"):
        cols = st.columns(3)
    kpi(
        cols[0],
        "adr-r2",
        "R²",
        fmt_number(rf["r2"], 3),
        "warm",
        "donut_large",
        "Proporción de la varianza de la ADR explicada por el Random Forest (9.299 reservas mantenidas).",
        f"{signed(rf['r2'] - lin['r2'], 3)} vs. lineal",
    )
    kpi(
        cols[1],
        "adr-rmse",
        "RMSE (u.m.)",
        fmt_number(rf["rmse"], 2),
        "warm",
        "straighten",
        "Raíz del error cuadrático medio por noche.",
        f"{signed(rf['rmse'] - lin['rmse'], 2)} vs. lineal",
    )
    kpi(
        cols[2],
        "adr-mae",
        "MAE (u.m.)",
        fmt_number(rf["mae"], 2),
        "warm",
        "straighten",
        "Error absoluto medio por noche.",
        f"{signed(rf['mae'] - lin['mae'], 2)} vs. lineal",
    )
    left, right = st.columns(2, gap="medium")
    with left, card("adr-cmp", height="stretch"):
        comparison(
            reg["metrics"], "rmse", 2, f"RMSE: −{fmt_pct(1 - rf['rmse'] / lin['rmse'], 0)} frente al lineal"
        )
    with right, card("adr-imp"):
        plot(
            charts.importance_bars(reg["permutation_importance"], "RMSE", "Mes y hotel explican la tarifa"),
            key="imp_adr",
        )
    adr_pred = predictions.filter(pl.col("task") == "regression")
    actual = adr_pred["adr"].to_numpy()
    predicted = adr_pred["adr_random_forest"].to_numpy()
    sample = np.random.default_rng(config.SEED).choice(len(actual), min(4000, len(actual)), replace=False)
    left, right = st.columns(2, gap="medium")
    with left, card("avp"):
        plot(charts.actual_vs_predicted(actual[sample], predicted[sample]), key="avp")
    with right, card("resid"):
        plot(charts.residual_histogram(actual, predicted), key="resid")
    with st.expander("Tablas de métricas y referencia de la Fase 3", icon=":material/table:"):
        keys = ["mae", "rmse", "r2"]
        st.markdown("**Recalculadas** · 55.791 entren. / 9.299 valid.")
        st.dataframe(
            metric_table(reg["metrics"], keys, {"validation": keys, "train": ["rmse", "r2"]}), hide_index=True
        )
        st.markdown("**Reportadas en la Fase 3** · validación")
        st.dataframe(reference_table(phase3["regression"]), hide_index=True)
        gap = reg["metrics"]["random_forest"]["train"]["r2"] - rf["r2"]
        st.caption(
            f"Brecha de R² entrenamiento–validación: {fmt_number(gap, 3)}. Bosque de "
            f"{reg['params']['n_estimators']} árboles, profundidad {reg['params']['max_depth']}, 50 % de variables "
            f"por división, {reg['params']['min_samples_leaf']} reservas por hoja. Importancia: aumento del RMSE "
            "al mezclar cada variable (800 reservas, 3 repeticiones); mide dependencia del modelo, no causalidad."
        )

with cancel_tab:
    rf = cls["metrics"]["random_forest"]["validation"]
    log = cls["metrics"]["logistic"]["validation"]
    with st.container(key="krow-cls"):
        cols = st.columns(4)
    for col, (key, name, glyph, decimals) in zip(
        cols,
        [
            ("roc_auc", "AUC", "target", 3),
            ("f1", "F1", "balance", 3),
            ("precision", "Precisión", "verified", 3),
            ("recall", "Sensibilidad", "radar", 3),
        ],
        strict=True,
    ):
        kpi(
            col,
            f"cls-{key}",
            name,
            fmt_number(rf[key], decimals),
            "brand",
            glyph,
            "Random Forest en validación (14.821 reservas), umbral 0,50.",
            f"{signed(rf[key] - log[key], 3)} vs. logística",
        )
    left, right = st.columns(2, gap="medium")
    with left, card("cls-cmp", height="stretch"):
        comparison(cls["metrics"], "roc_auc", 3, "El bosque ordena mejor el riesgo")
    with right, card("cls-imp"):
        plot(
            charts.importance_bars(cls["permutation_importance"], "AUC", "Cliente y anticipación, al frente"),
            key="imp_cls",
        )

    probs = predictions.filter(pl.col("task") == "classification")
    y = probs["is_canceled"].to_numpy().astype(int)
    p = probs["p_random_forest"].to_numpy()
    section("Umbral de decisión", "tune", "muévalo para ver el equilibrio")
    with card("threshold"):
        threshold = st.slider(
            "Umbral de decisión",
            0.05,
            0.95,
            config.THRESHOLD,
            0.05,
            format="%.2f",
            label_visibility="collapsed",
            help="Probabilidad a partir de la cual una reserva se marca como cancelación probable.",
        )
        predicted = (p >= threshold).astype(int)
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())
        tn = int(((predicted == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        m = st.columns(4)
        m[0].metric(
            "Alertas",
            fmt_number(tp + fp),
            help=f"{fmt_number(tp)} cancelaciones detectadas, {fmt_number(fp)} falsas alertas y "
            f"{fmt_number(fn)} omitidas sobre {fmt_number(len(y))} reservas.",
        )
        m[1].metric("Precisión", fmt_pct(precision))
        m[2].metric("Sensibilidad", fmt_pct(recall))
        m[3].metric("F1", fmt_number(f1, 3))
        left, right = st.columns(2, gap="medium")
        with left:
            plot(charts.confusion_heatmap([[tn, fp], [fn, tp]]), key="cm")
        with right:
            auc = {k: cls["metrics"][k]["validation"]["roc_auc"] for k in ["random_forest", "logistic"]}
            plot(charts.roc_curves(cls["roc"], auc), key="roc")
    with st.expander("Tablas de métricas y referencia de la Fase 3", icon=":material/table:"):
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        st.markdown("**Recalculadas** · 88.923 entren. / 14.821 valid. · umbral 0,50")
        st.dataframe(
            metric_table(cls["metrics"], keys, {"validation": keys, "train": ["f1", "roc_auc"]}),
            hide_index=True,
        )
        st.markdown("**Reportadas en la Fase 3** · validación, umbral 0,50")
        st.dataframe(reference_table(phase3["classification"]), hide_index=True)
        st.caption(
            f"AUC en entrenamiento {fmt_number(cls['metrics']['random_forest']['train']['roc_auc'], 3)} frente a "
            f"{fmt_number(rf['roc_auc'], 3)} en validación. XGBoost y las redes MLP se entrenaron en la Fase 3; aquí "
            "se despliega el Random Forest seleccionado (53 árboles, profundidad 18, raíz cuadrada de las variables, "
            "8 reservas por hoja), que reproduce su matriz de confusión. Importancia: caída del AUC al mezclar cada "
            "variable."
        )

with cluster_tab:
    ref = phase3["clustering"]
    k6 = next(row for row in clu["k_selection"] if row["k"] == clu["k"])
    best = max(clu["k_selection"], key=lambda row: row["silhouette"])
    with st.container(key="krow-clu"):
        cols = st.columns(3)
    kpi(
        cols[0],
        "clu-k",
        "Grupos (K)",
        str(clu["k"]),
        "violet",
        "hub",
        "Codo de la inercia y perfiles interpretables.",
    )
    kpi(
        cols[1],
        "clu-sil",
        "Silueta valid.",
        fmt_number(clu["silhouette_validation"], 3),
        "violet",
        "bubble_chart",
        "Coeficiente de silueta en 2.500 reservas de validación (−1 a 1).",
        f"entren. {fmt_number(clu['silhouette_train'], 3)}",
        arrow=False,
    )
    kpi(
        cols[2],
        "clu-ari",
        "Estabilidad (ARI)",
        fmt_number(k6["stability_ari_mean"], 3),
        "violet",
        "verified",
        "Índice de Rand ajustado medio entre tres semillas: 1 = misma partición.",
    )
    left, right = st.columns(2, gap="medium")
    with left, card("elbow"):
        plot(charts.k_selection(clu["k_selection"], "inertia", clu["k"]), key="elbow")
    with right, card("sil"):
        plot(charts.k_selection(clu["k_selection"], "silhouette", clu["k"]), key="sil")
    db = clu["dbscan"]
    section("Piloto DBSCAN", "scatter_plot", "5.000 reservas de entrenamiento")
    with st.container(key="krow-db"):
        cols = st.columns(3)
    for col, (key, name, value, reference, glyph) in zip(
        cols,
        [
            ("clusters", "Grupos", str(db["clusters"]), str(ref["dbscan"]["clusters"]), "hub"),
            ("noise", "Ruido", fmt_pct(db["noise_share"]), fmt_pct(ref["dbscan"]["noise_share"]), "grain"),
            (
                "silhouette",
                "Silueta",
                fmt_number(db["silhouette"], 3),
                fmt_number(ref["dbscan"]["silhouette"], 3),
                "bubble_chart",
            ),
        ],
        strict=True,
    ):
        kpi(
            col,
            f"db-{key}",
            name,
            value,
            "brand",
            glyph,
            f"Radio ε = {db['eps']}, mínimo {db['min_samples']} vecinos. La silueta solo describe las reservas "
            "asignadas y no es comparable con la de K-Means.",
            f"Fase 3: {reference}",
            arrow=False,
        )
    with st.expander("Tamaños frente a la Fase 3 y notas", icon=":material/table:"):
        own = {row["segment"]: row["size"] for row in clu["profiles_train"]}
        phase3_order = ["short_lead", "children", "baby", "loyal", "long_lead", "serial_cancellers"]
        st.dataframe(
            pl.DataFrame(
                {
                    "Segmento": [config.SEGMENTS[k]["name"] for k in phase3_order],
                    "Grupo Fase 3": list(range(6)),
                    "Fase 3": ref["sizes_train"],
                    "Recalculado": [own[k] for k in phase3_order],
                }
            ),
            hide_index=True,
        )
        st.caption(
            f"Silueta Fase 3: {fmt_number(ref['silhouette_train'], 3)} (entren.) y "
            f"{fmt_number(ref['silhouette_validation'], 3)} (valid.). La silueta máxima aparece con K = {best['k']} "
            f"({fmt_number(best['silhouette'], 3)}), pero esa partición cambia entre semillas (ARI medio "
            f"{fmt_number(best['stability_ari_mean'], 3)}) y separa casi solo la anticipación; K = {clu['k']} añade "
            "perfiles interpretables. DBSCAN separa sobre todo por composición de huéspedes."
        )

with method_tab:
    blocks = [
        (
            "database",
            "Datos",
            "119.390 reservas de un hotel urbano (Lisboa) y uno vacacional (Algarve), llegadas 07/2015–08/2017. "
            "Tras la limpieza de la Fase 2: **118.564** (sin ADR negativa, cero huéspedes ni cero noches).",
        ),
        (
            "call_split",
            "Particiones",
            "75 / 12,5 / 12,5 %, sin separar reservas con la misma huella. Cancelación 88.923 / 14.821 / 14.820 "
            "(estratificada); tarifa, solo mantenidas, 55.791 / 9.299 / 9.298; segmentación con partición propia.",
        ),
        (
            "shield",
            "Sin fuga de información",
            "Se excluyen habitación asignada, estado final y su fecha y, para cancelación, la ADR. Imputación, "
            "estandarización y one-hot se ajustan solo con entrenamiento dentro del pipeline.",
        ),
        (
            "forest",
            "Modelos desplegados",
            f"Tarifa: Random Forest, {reg['params']['n_estimators']} árboles, profundidad {reg['params']['max_depth']}. "
            f"Cancelación: {cls['params']['n_estimators']} árboles, profundidad {cls['params']['max_depth']}. "
            f"Segmentación: K-Means (K = {clu['k']}) y PCA a tres componentes solo para visualizar.",
        ),
        (
            "tune",
            "Predictores",
            f"Numéricos: {', '.join(config.FEATURE_LABELS[f].lower() for f in config.NUMERIC_FEATURES)}. "
            f"Categóricos: {', '.join(config.FEATURE_LABELS[f].lower() for f in config.CATEGORICAL_FEATURES)}.",
        ),
        (
            "replay",
            "Reproducibilidad",
            f"`training/train.py` regenera modelos y métricas (semilla {config.SEED}). "
            f"{metrics['generated_at']} · Python {metrics['environment']['python']} · "
            f"scikit-learn {metrics['environment']['scikit_learn']}.",
        ),
    ]
    for start in (0, 3):
        cols = st.columns(3, gap="medium")
        for col, (glyph, title, text) in zip(cols, blocks[start : start + 3], strict=True):
            with col, card(f"method-{glyph}", height="stretch"):
                st.markdown(f":blue[:material/{glyph}:] **{title}**")
                st.markdown(text)
    st.caption(f"Fuente: {config.DATASET_CITATION}")
