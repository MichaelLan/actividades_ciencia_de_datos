"""Models and methodology: metrics vs baselines and Phase 3, importance and diagnostics."""

import numpy as np
import polars as pl
import streamlit as st

from hotel_dashboard import charts, config
from hotel_dashboard.data import load_metrics, load_validation_predictions
from hotel_dashboard.kpis import fmt_number, fmt_pct
from hotel_dashboard.ui import header, plot

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
    "Metodología y evaluación",
    "Modelos y su desempeño",
    "Se reentrenaron los modelos seleccionados en la Fase 3, con su configuración ganadora, "
    "las mismas particiones de la Fase 2 y los mismos predictores (7 numéricos y 8 categóricos, "
    "preprocesados dentro del pipeline). Las métricas se recalculan en validación y coinciden con "
    "las del cuaderno de la Fase 3, que se muestran como referencia. La partición de prueba solo "
    "se usó una vez, con los modelos ya congelados.",
)

adr_tab, cancel_tab, cluster_tab, method_tab = st.tabs(
    ["Tarifa (ADR)", "Cancelación", "Segmentación", "Metodología"]
)

with adr_tab:
    rf = reg["metrics"]["random_forest"]["validation"]
    lin = reg["metrics"]["linear"]["validation"]
    dummy = reg["metrics"]["dummy"]["validation"]
    st.markdown(
        f"Random Forest reduce el RMSE de **{fmt_number(lin['rmse'], 2)}** (lineal) a "
        f"**{fmt_number(rf['rmse'], 2)}** u.m. por noche (−{fmt_pct(1 - rf['rmse'] / lin['rmse'], 0)}) y "
        f"explica el **{fmt_pct(rf['r2'], 0)}** de la varianza; la constante se queda en RMSE "
        f"{fmt_number(dummy['rmse'], 2)}."
    )
    left, right = st.columns([7, 5], gap="medium")
    with left, st.container(border=True, height="stretch"):
        st.markdown("**Métricas recalculadas** · 55.791 entren. / 9.299 valid.")
        keys = ["mae", "rmse", "r2"]
        st.dataframe(
            metric_table(reg["metrics"], keys, {"validation": keys, "train": ["rmse", "r2"]}), hide_index=True
        )
        gap = reg["metrics"]["random_forest"]["train"]["r2"] - rf["r2"]
        st.caption(
            f"Brecha de R² entre entrenamiento y validación del bosque: {fmt_number(gap, 3)}. "
            f"La profundidad máxima ({reg['params']['max_depth']}) y el mínimo de "
            f"{reg['params']['min_samples_leaf']} reservas por hoja limitan el sobreajuste."
        )
    with right, st.container(border=True, height="stretch"):
        st.markdown("**Referencia reportada en la Fase 3** · validación")
        st.dataframe(reference_table(phase3["regression"]), hide_index=True)
        st.caption(
            "Valores del cuaderno de la Fase 3 (Colab). El bosque desplegado usa la configuración ganadora de "
            "aquella búsqueda (58 árboles, profundidad 18, 50 % de variables por división, 6 reservas por hoja) "
            "y reproduce sus métricas."
        )
    adr_pred = predictions.filter(pl.col("task") == "regression")
    actual = adr_pred["adr"].to_numpy()
    predicted = adr_pred["adr_random_forest"].to_numpy()
    sample = np.random.default_rng(config.SEED).choice(len(actual), min(4000, len(actual)), replace=False)
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(charts.actual_vs_predicted(actual[sample], predicted[sample]), key="avp")
        st.caption("Las tarifas altas quedan por debajo de la diagonal: el bosque tiende a subestimarlas.")
    with right, st.container(border=True):
        plot(charts.residual_histogram(actual, predicted), key="resid")
        st.caption("Los errores se concentran cerca de cero, con una cola de residuos positivos.")
    with st.container(border=True):
        plot(
            charts.importance_bars(
                reg["permutation_importance"], "RMSE", "Importancia por permutación · tarifa"
            ),
            key="imp_adr",
        )
        st.caption(
            "Aumento del RMSE al mezclar cada variable (800 reservas de validación, 3 repeticiones). "
            "El mes de llegada y el hotel dominan, igual que en la Fase 3; mide dependencia del modelo, no causalidad."
        )

with cancel_tab:
    rf = cls["metrics"]["random_forest"]["validation"]
    log = cls["metrics"]["logistic"]["validation"]
    st.markdown(
        f"El bosque ordena mejor el riesgo que la regresión logística (AUC **{fmt_number(rf['roc_auc'], 3)}** "
        f"frente a {fmt_number(log['roc_auc'], 3)}). Con el umbral de 0,50 casi nueve de cada diez alertas "
        "son correctas, pero se escapan seis de cada diez cancelaciones. Mueva el umbral para ver el equilibrio."
    )
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    with st.container(border=True):
        st.markdown("**Métricas recalculadas** · 88.923 entren. / 14.821 valid. · umbral 0,50")
        st.dataframe(
            metric_table(cls["metrics"], keys, {"validation": keys, "train": ["f1", "roc_auc"]}),
            hide_index=True,
        )
        st.caption(
            f"AUC en entrenamiento {fmt_number(cls['metrics']['random_forest']['train']['roc_auc'], 3)} frente a "
            f"{fmt_number(rf['roc_auc'], 3)} en validación: sobreajuste moderado y controlado."
        )
    with st.container(border=True):
        st.markdown("**Referencia reportada en la Fase 3** · validación, umbral 0,50")
        st.dataframe(reference_table(phase3["classification"]), hide_index=True)
        st.caption(
            "XGBoost y las redes MLP (PyTorch) se entrenaron en la Fase 3; aquí se despliega el Random Forest "
            "seleccionado (53 árboles, profundidad 18, raíz cuadrada de las variables por división, 8 reservas por hoja), "
            "que reproduce exactamente su matriz de confusión."
        )

    probs = predictions.filter(pl.col("task") == "classification")
    y = probs["is_canceled"].to_numpy().astype(int)
    p = probs["p_random_forest"].to_numpy()
    with st.container(border=True):
        threshold = st.slider(
            "Umbral de decisión",
            0.05,
            0.95,
            config.THRESHOLD,
            0.05,
            format="%.2f",
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
        m[0].metric("Alertas", fmt_number(tp + fp))
        m[1].metric("Precisión", fmt_pct(precision))
        m[2].metric("Sensibilidad", fmt_pct(recall))
        m[3].metric("F1", fmt_number(f1, 3))
        left, right = st.columns(2, gap="medium")
        with left:
            plot(charts.confusion_heatmap([[tn, fp], [fn, tp]]), key="cm")
        with right:
            auc = {k: cls["metrics"][k]["validation"]["roc_auc"] for k in ["random_forest", "logistic"]}
            plot(charts.roc_curves(cls["roc"], auc), key="roc")
        st.caption(
            f"Con umbral {fmt_number(threshold, 2)}: {fmt_number(tp)} cancelaciones detectadas, {fmt_number(fp)} "
            f"falsas alertas y {fmt_number(fn)} cancelaciones omitidas sobre {fmt_number(len(y))} reservas."
        )
    with st.container(border=True):
        plot(
            charts.importance_bars(
                cls["permutation_importance"], "AUC", "Importancia por permutación · cancelación"
            ),
            key="imp_cls",
        )
        st.caption(
            "Caída del AUC al mezclar cada variable. Anticipación, tipo de cliente, segmento y "
            "cancelaciones previas encabezan la lista, como en la Fase 3 (allí el tipo de cliente quedó primero)."
        )

with cluster_tab:
    ref = phase3["clustering"]
    k6 = next(row for row in clu["k_selection"] if row["k"] == clu["k"])
    best = max(clu["k_selection"], key=lambda row: row["silhouette"])
    st.markdown(
        f"Se adoptó **K = {clu['k']}** como solución descriptiva: coincide con el codo de la inercia, es "
        f"estable entre semillas (ARI medio {fmt_number(k6['stability_ari_mean'], 3)}) y su silueta en validación "
        f"({fmt_number(clu['silhouette_validation'], 3)}) es similar a la de entrenamiento "
        f"({fmt_number(clu['silhouette_train'], 3)})."
    )
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        plot(charts.k_selection(clu["k_selection"], "inertia", clu["k"]), key="elbow")
    with right, st.container(border=True):
        plot(charts.k_selection(clu["k_selection"], "silhouette", clu["k"]), key="sil")
    st.caption(
        f"La silueta máxima aparece con K = {best['k']} ({fmt_number(best['silhouette'], 3)}), pero esa partición "
        f"cambia entre semillas (ARI medio {fmt_number(best['stability_ari_mean'], 3)}) y separa casi solo la "
        "anticipación; K = 6 añade perfiles interpretables (familias, clientes fieles, historial de cancelaciones)."
    )
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("**Tamaños en entrenamiento: recalculado frente a Fase 3**")
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
            f"{fmt_number(ref['silhouette_validation'], 3)} (valid.), con muestras de 2.500 reservas, como aquí."
        )
    with right, st.container(border=True):
        db = clu["dbscan"]
        st.markdown("**Piloto DBSCAN** · 5.000 reservas de entrenamiento")
        a, b, c = st.columns(3)
        a.metric(
            "Grupos",
            db["clusters"],
            f"Fase 3: {ref['dbscan']['clusters']}",
            delta_color="off",
            delta_arrow="off",
        )
        b.metric(
            "Ruido",
            fmt_pct(db["noise_share"]),
            f"Fase 3: {fmt_pct(ref['dbscan']['noise_share'])}",
            delta_color="off",
            delta_arrow="off",
        )
        c.metric(
            "Silueta",
            fmt_number(db["silhouette"], 3),
            f"Fase 3: {fmt_number(ref['dbscan']['silhouette'], 3)}",
            delta_color="off",
            delta_arrow="off",
        )
        st.caption(
            f"Radio ε = {db['eps']} y mínimo de {db['min_samples']} vecinos, como en la Fase 3. DBSCAN separa sobre "
            "todo por composición de huéspedes; la silueta solo describe las reservas asignadas y no es comparable con la de K-Means."
        )

with method_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
**Datos y unidad de análisis.** Hotel Booking Demand: 119.390 reservas de un hotel urbano
(Lisboa) y uno vacacional (Algarve) con llegadas entre julio de 2015 y agosto de 2017. Tras la
limpieza de la Fase 2 quedan 118.564 (se excluyen ADR negativa, cero huéspedes y cero noches).

**Particiones.** 75 / 12,5 / 12,5 % (estratificado por clase en cancelación), sin separar reservas con la misma
huella original. Cancelación: 88.923 / 14.821 / 14.820. Segmentación: partición propia de la Fase 2, sin
estratificar y con los mismos tamaños. Tarifa (solo reservas mantenidas): 55.791 / 9.299 / 9.298.

**Prevención de fuga de información.** Se excluyen la habitación asignada, el estado y la fecha
de estado de la reserva y, para cancelación, la propia ADR. Imputación (mediana / moda),
estandarización y one-hot se ajustan solo con entrenamiento dentro del pipeline.
"""
        )
    with right:
        st.markdown(
            f"""
**Predictores.** Numéricos: {", ".join(config.FEATURE_LABELS[f].lower() for f in config.NUMERIC_FEATURES)}.
Categóricos: {", ".join(config.FEATURE_LABELS[f].lower() for f in config.CATEGORICAL_FEATURES)}.

**Modelos desplegados.** Configuraciones ganadoras de la búsqueda aleatoria de la Fase 3 (3 candidatos ×
3 pliegues agrupados por huella). Tarifa: Random Forest de {reg["params"]["n_estimators"]} árboles, profundidad
{reg["params"]["max_depth"]}, {fmt_pct(reg["params"]["max_features"], 0)} de las variables por división y
{reg["params"]["min_samples_leaf"]} reservas por hoja. Cancelación: {cls["params"]["n_estimators"]} árboles,
profundidad {cls["params"]["max_depth"]}, raíz cuadrada de las variables y {cls["params"]["min_samples_leaf"]}
reservas por hoja. Segmentación: K-Means (K = {clu["k"]}, 10 inicializaciones) sobre las siete variables numéricas
estandarizadas y PCA a tres componentes solo para visualizar.

**Reproducibilidad.** `training/train.py` regenera modelos, métricas y el conjunto compacto
de la aplicación a partir de los Parquet de la Fase 2 (semilla {config.SEED}).
Generado: {metrics["generated_at"]} · Python {metrics["environment"]["python"]} ·
scikit-learn {metrics["environment"]["scikit_learn"]}.
"""
        )
    st.caption(f"Fuente: {config.DATASET_CITATION}")
