#import "presentacion.typ": *
#let e = d.eda
#let rft = test.regression.random_forest
#let cft = test.classification.random_forest
#let cv = cls.random_forest.validation
#let p2reg = p2.regression.metrics
#let p2cls = p2.classification.metrics
#let span(body) = table.cell(colspan: 6, text(style: "italic", body))
#let span4(body) = table.cell(colspan: 4, text(style: "italic", body))
#let r3(x) = fmt(x, digits: 3)
#let nbr = nb.models.random_forest_regressor
#let nbe = nb.models.elastic_net
#let nbc = nb.models.random_forest_classifier
#let k-row(k) = clu.k_selection.find(r => r.k == k)
#let kd = clu.k_decision
= Resultados

== Hallazgos exploratorios
La ADR tiene una cola derecha pronunciada: media de #fmt(e.adr.mean), mediana de #fmt(e.adr.median), asimetría de #fmt(e.adr.skewness) y un máximo de #fmt(e.adr.max, digits: 0) (fig. @fig:dist). La mediana del hotel urbano (#fmt(e.hotel.at(0).adr_median, pad: false)) supera a la del vacacional (#fmt(e.hotel.at(1).adr_median, pad: false)). Se canceló el #pct(e.coverage.cancellation_prevalence) de las reservas: #pct(e.hotel.at(0).cancellation_rate) en el hotel urbano y #pct(e.hotel.at(1).cancellation_rate) en el vacacional. La tasa crece con la anticipación, desde el #pct(e.lead_time_bins.at(0).cancellation_rate) con siete días o menos hasta el #pct(e.lead_time_bins.at(5).cancellation_rate) con más de un año (Spearman~=~#r3(e.spearman_lead_cancel)). El depósito mostró la asociación más fuerte (V~=~#fmt(e.chi_deposit.cramers_v)), pero su registro puede ser posterior al alta y se excluyó de los predictores.

#plot("distributions", [Distribución de ADR y de la anticipación (truncadas a 300 y 500) y ADR por hotel sin valores extremos. Fuente: Fase 1 @fase1.], wide: true) <fig:dist>

== Estimación de la tarifa diaria
Random Forest redujo el RMSE de validación de #fmt(reg.linear.validation.rmse) (regresión lineal) a #fmt(reg.random_forest.validation.rmse), un #pct(1 - reg.random_forest.validation.rmse / reg.linear.validation.rmse) menos (tabla @tab:reg). ElasticNet no mejoró a la regresión lineal: la limitación del modelo lineal es de forma funcional, no de varianza. En validación cruzada, el bosque obtuvo un RMSE medio de #fmt(nbr.cv.cv_rmse_mean) (DE #fmt(nbr.cv.cv_rmse_std)) frente a #fmt(nbe.cv.cv_rmse_mean) (DE #fmt(nbe.cv.cv_rmse_std)) de ElasticNet @fase3nb. El reentrenamiento de la Fase 4 reprodujo exactamente las métricas de validación del cuaderno. El MAE de validación equivale al #pct(reg.random_forest.validation.mae / d.business.mean_adr_validation) de la ADR media de esa partición; en prueba fue de #fmt(rft.mae) unidades por noche.

#tab(
  ([Modelo], [MAE], [RMSE], [$R^2$]),
  (
    span4[Validación (#fmt(m.regression.rows.validation) reservas)],
    ([Constante], fmt(reg.dummy.validation.mae), fmt(reg.dummy.validation.rmse), r3(calc.abs(reg.dummy.validation.r2))),
    ([Regresión lineal], fmt(reg.linear.validation.mae), fmt(reg.linear.validation.rmse), r3(reg.linear.validation.r2)),
    ([ElasticNet (F3)], fmt(p3.regression.ElasticNet.mae), fmt(p3.regression.ElasticNet.rmse), r3(p3.regression.ElasticNet.r2)),
    ([*Random Forest (F3/F4)*], fmt(reg.random_forest.validation.mae), fmt(reg.random_forest.validation.rmse), r3(reg.random_forest.validation.r2)),
    span4[Prueba (#fmt(test.regression.rows) reservas)],
    ([Regresión lineal (F2)], fmt(p2reg.linear.test.mae), fmt(p2reg.linear.test.rmse), r3(p2reg.linear.test.r2)),
    ([*Random Forest (F4)*], fmt(rft.mae), fmt(rft.rmse), r3(rft.r2)),
  ),
  [Regresión de ADR en reservas mantenidas. F2–F4 indican la fase que calculó la métrica; F3/F4: el reentrenamiento reproduce la Fase 3.],
) <tab:reg>

Los errores se concentran cerca de cero, pero las tarifas altas tienden a subestimarse (fig. @fig:adr). El intervalo _bootstrap_ del RMSE en prueba fue [#fmt(test.regression.rmse_ci95.at(0)), #fmt(test.regression.rmse_ci95.at(1))].

#plot("adr_predictions", [ADR observada frente a predicha por Random Forest (densidad logarítmica) y residuos en validación. La diagonal indica una estimación exacta.]) <fig:adr>

== Predicción de cancelaciones
Los modelos no lineales superaron el AUC de la regresión logística (#r3(cls.logistic.validation.roc_auc)); entre ellos las diferencias fueron menores de 0,01 (tabla @tab:cls, fig. @fig:models). Con el umbral de 0,50, Random Forest priorizó la precisión: en validación emitió #fmt(cv.confusion_matrix.at(0).at(1) + cv.confusion_matrix.at(1).at(1)) alertas, de las que #fmt(cv.confusion_matrix.at(1).at(1)) fueron cancelaciones reales, y omitió #fmt(cv.confusion_matrix.at(1).at(0)) (fig. @fig:cls). En la Fase 3, XGBoost detectó #fmt(p3r.xgb_confusion.tp - p3r.rf_confusion.tp) cancelaciones más que el bosque a cambio de #fmt(p3r.xgb_confusion.fp - p3r.rf_confusion.fp) falsas alertas adicionales @fase3. La MLP pequeña alcanzó la mayor sensibilidad y la balanceada (#p3r.mlp.hidden.at(0) y #p3r.mlp.hidden.at(1) neuronas ocultas) el mayor AUC de las redes; ninguna superó al bosque en AUC. Se eligió Random Forest por su AUC, su precisión y su menor coste de despliegue. Su AUC medio en validación cruzada fue de #r3(nbc.cv.cv_roc_auc_mean) (DE #r3(nbc.cv.cv_roc_auc_std)) @fase3nb.

#tab(
  ([Modelo], [Exact.], [Prec.], [Sens.], [F1], [AUC]),
  (
    span[Validación (#fmt(m.classification.rows.validation) reservas)],
    ([Clase mayoritaria], r3(cls.dummy.validation.accuracy), [—], r3(0.0), r3(0.0), r3(0.5)),
    ([Logística], r3(cls.logistic.validation.accuracy), r3(cls.logistic.validation.precision), r3(cls.logistic.validation.recall), r3(cls.logistic.validation.f1), r3(cls.logistic.validation.roc_auc)),
    ..(("XGBoost", "XGBoost (F3)"), ("MLP pequeña", "MLP pequeña (F3)"), ("MLP balanceada", "MLP balanc. (F3)"), ("MLP regularizada", "MLP regul. (F3)")).map(((k, label)) => {
      let r = p3.classification.at(k)
      ([#label], r3(r.accuracy), r3(r.precision), r3(r.recall), r3(r.f1), r3(r.roc_auc))
    }),
    ([*R. Forest (F3/F4)*], r3(cv.accuracy), r3(cv.precision), r3(cv.recall), r3(cv.f1), r3(cv.roc_auc)),
    span[Prueba (#fmt(test.classification.rows) reservas)],
    ([Logística (F2)], r3(p2cls.logistic.test.accuracy), r3(p2cls.logistic.test.precision), r3(p2cls.logistic.test.recall), r3(p2cls.logistic.test.f1), r3(p2cls.logistic.test.roc_auc)),
    ([*R. Forest (F4)*], r3(cft.accuracy), r3(cft.precision), r3(cft.recall), r3(cft.f1), r3(cft.roc_auc)),
  ),
  [Clasificación de cancelaciones con umbral de 0,50 (clase positiva: cancelada).],
  columns: (1.9fr, 1fr, 1fr, 1fr, 1fr, 1fr),
) <tab:cls>

#plot("model_comparison", [Comparación en validación. La barra azul es el modelo seleccionado y la línea discontinua, su resultado en prueba. \*Métrica de la Fase 3 @fase3.]) <fig:models>

#plot("classification", [Curvas ROC y matriz de confusión de Random Forest en validación.]) <fig:cls>

== Sobreajuste y generalización
Random Forest muestra una brecha moderada entre entrenamiento y validación: $R^2$ de #fmt(reg.random_forest.train.r2, digits: 3) frente a #fmt(reg.random_forest.validation.r2, digits: 3) en ADR, y AUC de #r3(cls.random_forest.train.roc_auc) frente a #r3(cv.roc_auc) en cancelación (fig. @fig:overfit). La profundidad máxima y el mínimo de observaciones por hoja limitan la memorización. Los modelos lineales casi no presentan brecha, pero su desempeño es menor. En las MLP, la pérdida de entrenamiento siguió bajando después de la época #p3r.mlp.best_epoch, mientras que la de validación se estabilizó en #fmt(p3r.mlp.best_val_loss, digits: 3); se conservaron los pesos de esa época @fase3. En prueba no hubo deterioro: el AUC fue de #r3(cft.roc_auc) (IC 95~%: #r3(test.classification.roc_auc_ci95.at(0))–#r3(test.classification.roc_auc_ci95.at(1))), superior al de validación. La regresión logística de la Fase 2 también mejoró en prueba (#r3(p2cls.logistic.test.roc_auc) frente a #r3(cls.logistic.validation.roc_auc)), lo que apunta a diferencias entre particiones más que a un efecto del modelo.

#plot("overfitting", [Métricas de los Random Forest seleccionados en entrenamiento, validación y prueba.]) <fig:overfit>

== Interpretabilidad
Al permutar el mes de llegada, el RMSE del bosque de ADR aumentó #fmt(m.regression.permutation_importance.at(0).mean) unidades; le siguieron el hotel, el segmento de mercado y la habitación reservada (fig. @fig:imp). En cancelación, el tipo de cliente, la anticipación, el segmento y las cancelaciones previas produjeron las mayores caídas de AUC. Con los mismos modelos y la misma muestra de validación, los valores coinciden con los de la Fase 3 @fase3nb. Estas medidas indican cuánto depende el modelo de cada variable; no establecen la dirección ni el efecto causal de la asociación.

#plot("importance", [Importancia por permutación en validación (#p3r.permutation.rows reservas, #p3r.permutation.repeats repeticiones); las barras de error son la desviación estándar.]) <fig:imp>

== Segmentación de las reservas
La silueta alcanzó su máximo con $K = #kd.max_silhouette_k$ (#r3(kd.max_silhouette)), pero esa partición cambió entre semillas (ARI medio de #r3(k-row(kd.max_silhouette_k).stability_ari_mean)); el codo apareció en $K = #kd.elbow_k$, con un ARI medio de #r3(k-row(6).stability_ari_mean) (fig. @fig:k). Ningún $K$ cumplió a la vez los tres criterios de la regla automática, como en la Fase 3 @fase3nb, y se adoptó $K = 6$ como solución descriptiva, estable e interpretable. Su silueta fue de #r3(clu.silhouette_train) en entrenamiento, #r3(clu.silhouette_validation) en validación y #r3(test.clustering.silhouette) en prueba: una separación moderada y similar en las tres particiones.

#plot("k_selection", [Inercia, silueta y ARI medio entre tres semillas de K-Means en entrenamiento para $K = 2, dots, 10$. La línea discontinua marca la solución adoptada; la punteada, el ARI mínimo exigido.]) <fig:k>

#let prof(key) = clu.profiles_train.find(p => p.segment == key)
#let ntrain = clu.profiles_train.map(p => p.size).sum()
#let names = (short_lead: "Corto plazo", long_lead: "Anticipada", children: "Familias con niños", baby: "Familias con bebé", loyal: "Clientes fieles", serial_cancellers: "Historial de cancel.")
#tab(
  ([Segmento], [Reservas], [%], [Antic.], [Noches], [Cancel.]),
  ("short_lead", "long_lead", "children", "baby", "loyal", "serial_cancellers").map(k => {
    let p = prof(k)
    (names.at(k), fmt(p.size), fmt(100 * p.size / ntrain, digits: 1), fmt(p.lead_time_median, digits: 0), fmt(p.total_nights_median, digits: 0), [#fmt(100 * p.cancel_rate, digits: 1)~%])
  }),
  [Perfiles de K-Means en entrenamiento: tamaño, medianas de anticipación (días) y noches, y tasa de cancelación observada (no usada para agrupar).],
  columns: (1.9fr, 1.1fr, 0.7fr, 0.8fr, 0.8fr, 0.9fr),
) <tab:seg>

Los dos grupos mayoritarios suman el #pct((prof("short_lead").size + prof("long_lead").size) / ntrain) de las reservas de entrenamiento (tabla @tab:seg, fig. @fig:pca). El de corto plazo tiene una anticipación mediana de #fmt(prof("short_lead").lead_time_median, digits: 0) días y el anticipado, de #fmt(prof("long_lead").lead_time_median, digits: 0); este último cancela el #pct(prof("long_lead").cancel_rate), frente al #pct(prof("short_lead").cancel_rate). Los cuatro grupos restantes se definen por niños, bebés o historial; el de cancelaciones previas reúne solo #prof("serial_cancellers").size reservas, casi todas canceladas. El plano de las dos primeras componentes principales explica el #pct(clu.pca_explained_variance.at(0) + clu.pca_explained_variance.at(1)) de la varianza, por lo que los grupos se solapan en la proyección aunque estén separados en siete dimensiones.

#plot("pca_clusters", [Reservas limpias en las dos primeras componentes principales, coloreadas por segmento (hasta 2500 por grupo).]) <fig:pca>

El piloto DBSCAN eligió un radio de #fmt(clu.dbscan.eps, digits: 3) con #clu.dbscan.min_samples vecinos mínimos: #clu.dbscan.clusters grupos, #clu.dbscan.sizes.at(0) reservas (#pct(clu.dbscan.noise_share)) como ruido y una silueta de #r3(clu.dbscan.silhouette) en las observaciones asignadas; el mayor grupo reunía reservas de dos adultos sin niños. El reentrenamiento reprodujo los radios candidatos y la selección de la Fase 3 @fase3nb. DBSCAN separa sobre todo por composición de huéspedes y es sensible a la muestra; K-Means ofrece perfiles más útiles para describir la demanda.

== Tablero interactivo
La aplicación Streamlit @streamlit (fig. @fig:dash) tiene cuatro páginas (Resumen, Predicción, Segmentación y Modelos) que cubren los módulos exigidos. (i) _Filtros dinámicos_: rango de fechas de llegada; hotel, segmento, tipo de cliente, depósito y canal; y rangos de anticipación, ADR y noches. Los filtros recalculan indicadores, gráficos y mapa. (ii) _Indicadores ejecutivos_: reservas, tasa de cancelación, ADR media, ingreso, ingreso perdido por cancelaciones y anticipación media, junto con el $R^2$, el RMSE, el AUC, el F1 y la silueta, con su diferencia frente a las referencias. (iii) _Predicción en vivo_: un formulario recibe solo variables disponibles antes de la llegada y devuelve la probabilidad de cancelación, la ADR estimada con el rango entre los percentiles 10 y 90 de los árboles y el segmento asignado. (iv) _Segmentación_: mapa PCA 2D/3D, tamaños, perfil medio en un mapa de calor y posición de la reserva simulada. La página de modelos muestra métricas, importancias, residuos, ROC y una matriz de confusión con umbral ajustable. El tablero está disponible en #dashboard-link.

#plot("dashboard", [Tablero: resumen ejecutivo con filtros e indicadores (izquierda) y predicción en vivo de una reserva (derecha).], wide: true, ext: "png") <fig:dash>

La aplicación solo necesita `data/bookings.parquet` (#fmt(d.business.bookings) reservas limpias, con segmento y coordenadas PCA) y los modelos serializados. El entorno se gestiona con uv; `requirements.txt` fija 42 dependencias con versiones exactas (Python 3.13, scikit-learn #m.environment.scikit_learn, Streamlit 1.67.0) para Streamlit Community Cloud. Doce pruebas `pytest` cubren la carga de datos, los filtros, los indicadores y la inferencia, y `ruff` controla el estilo.
