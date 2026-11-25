#import "presentacion.typ": *
#let e = d.eda
#let rft = test.regression.random_forest
#let cft = test.classification.random_forest
#let cmt = cft.confusion_matrix
#let cv = cls.random_forest.validation
#let p2reg = p2.regression.metrics
#let p2cls = p2.classification.metrics
#let span(body) = table.cell(colspan: 6, text(style: "italic", body))
#let span4(body) = table.cell(colspan: 4, text(style: "italic", body))
#let r3(x) = fmt(x, digits: 3)
#let k-row(k) = clu.k_selection.find(r => r.k == k)
#let kd = clu.k_decision
#let shot(name, caption, width: 100%, wide: true) = figure(
  block(stroke: 0.4pt + luma(200), width: width, image("imagenes/" + name + ".png", width: 100%)),
  caption: caption, kind: image,
  placement: top,
  scope: if wide { "parent" } else { "column" },
)
= Resultados

== Hallazgos exploratorios
La ADR tiene una cola derecha pronunciada: media de #fmt(e.adr.mean) unidades monetarias (u.m.) por noche, mediana de #fmt(e.adr.median) y algunos valores muy altos (fig. @fig:dist). La mediana del hotel urbano (#fmt(e.hotel.at(0).adr_median, pad: false)) supera a la del vacacional (#fmt(e.hotel.at(1).adr_median, pad: false)). Se canceló el #pct(e.coverage.cancellation_prevalence) de las reservas: el #pct(e.hotel.at(0).cancellation_rate) en el hotel urbano y el #pct(e.hotel.at(1).cancellation_rate) en el vacacional. La tasa crece con la anticipación, desde el #pct(e.lead_time_bins.at(0).cancellation_rate) cuando se reserva con siete días o menos hasta el #pct(e.lead_time_bins.at(5).cancellation_rate) cuando se reserva con más de un año. El tipo de depósito mostró la asociación más fuerte con la cancelación (V~=~#fmt(e.chi_deposit.cramers_v)), pero puede registrarse después del alta y se excluyó de los predictores.

#plot("distributions", [Distribución de la ADR y de la anticipación (truncadas a 300 u.m. y 500 días) y ADR por hotel, sin valores extremos.], wide: true) <fig:dist>

== Estimación de la tarifa diaria
Random Forest redujo el RMSE de validación de #fmt(reg.linear.validation.rmse) (regresión lineal) a #fmt(reg.random_forest.validation.rmse) u.m., un #pct(1 - reg.random_forest.validation.rmse / reg.linear.validation.rmse) menos (tabla @tab:reg). ElasticNet no mejoró a la regresión lineal: la limitación del modelo lineal no es el sobreajuste, sino su incapacidad para representar relaciones no lineales. En la partición de prueba, el bosque obtuvo un $R^2$ de #r3(rft.r2): explica aproximadamente el #pct(rft.r2, digits: 0) de la variación de la tarifa entre las reservas mantenidas de prueba. Su error absoluto medio (MAE) fue de #fmt(rft.mae) u.m. por noche, en torno al #pct(rft.mae / d.business.mean_adr_validation, digits: 0) de la ADR media, y su RMSE, de #fmt(rft.rmse) (IC 95~%: #fmt(test.regression.rmse_ci95.at(0))–#fmt(test.regression.rmse_ci95.at(1))). Los errores se concentran cerca de cero, pero las tarifas más altas tienden a subestimarse (fig. @fig:adr).

#tab(
  ([Modelo], [MAE], [RMSE], [$R^2$]),
  (
    span4[Validación (#fmt(m.regression.rows.validation) reservas)],
    ([Constante (media)], fmt(reg.dummy.validation.mae), fmt(reg.dummy.validation.rmse), r3(calc.abs(reg.dummy.validation.r2))),
    ([Regresión lineal], fmt(reg.linear.validation.mae), fmt(reg.linear.validation.rmse), r3(reg.linear.validation.r2)),
    ([ElasticNet], fmt(p3.regression.ElasticNet.mae), fmt(p3.regression.ElasticNet.rmse), r3(p3.regression.ElasticNet.r2)),
    ([*Random Forest*], fmt(reg.random_forest.validation.mae), fmt(reg.random_forest.validation.rmse), r3(reg.random_forest.validation.r2)),
    span4[Prueba (#fmt(test.regression.rows) reservas)],
    ([Regresión lineal], fmt(p2reg.linear.test.mae), fmt(p2reg.linear.test.rmse), r3(p2reg.linear.test.r2)),
    ([*Random Forest*], fmt(rft.mae), fmt(rft.rmse), r3(rft.r2)),
  ),
  [Regresión de la ADR en reservas mantenidas (MAE y RMSE en u.m. por noche).],
) <tab:reg>

#plot("adr_predictions", [ADR observada frente a predicha por Random Forest (densidad logarítmica) y residuos en validación. La diagonal indica una estimación exacta.]) <fig:adr>

== Predicción de cancelaciones
Los modelos no lineales superaron el AUC de la regresión logística (#r3(cls.logistic.validation.roc_auc)), y entre ellos las diferencias fueron menores de 0,01 (tabla @tab:cls). XGBoost y la MLP pequeña detectaron algo más de cancelaciones a cambio de más falsas alertas; ninguna red superó al bosque en AUC. Se eligió Random Forest por su AUC, su precisión y su sencillez de uso en la aplicación. En prueba alcanzó un AUC de #r3(cft.roc_auc) (IC 95~%: #r3(test.classification.roc_auc_ci95.at(0))–#r3(test.classification.roc_auc_ci95.at(1))). Con el umbral de 0,50 emitió #fmt(cmt.at(0).at(1) + cmt.at(1).at(1)) alertas, de las que #fmt(cmt.at(1).at(1)) fueron cancelaciones reales: una precisión del #pct(cft.precision), es decir, casi nueve de cada diez reservas señaladas se cancelaron. Su sensibilidad fue del #pct(cft.recall): identificó menos de la mitad de las cancelaciones y dejó sin señalar #fmt(cmt.at(1).at(0)). La curva ROC y la matriz de confusión de validación (fig. @fig:cls) muestran el mismo patrón: el modelo ordena bien el riesgo, pero el umbral de 0,50 es conservador.

#tab(
  ([Modelo], [Exact.], [Prec.], [Sens.], [F1], [AUC]),
  (
    span[Validación (#fmt(m.classification.rows.validation) reservas)],
    ([Clase mayoritaria], r3(cls.dummy.validation.accuracy), [—], r3(0.0), r3(0.0), r3(0.5)),
    ([Logística], r3(cls.logistic.validation.accuracy), r3(cls.logistic.validation.precision), r3(cls.logistic.validation.recall), r3(cls.logistic.validation.f1), r3(cls.logistic.validation.roc_auc)),
    ..(("XGBoost", "XGBoost"), ("MLP pequeña", "MLP pequeña"), ("MLP balanceada", "MLP balanceada"), ("MLP regularizada", "MLP regularizada")).map(((k, label)) => {
      let r = p3.classification.at(k)
      ([#label], r3(r.accuracy), r3(r.precision), r3(r.recall), r3(r.f1), r3(r.roc_auc))
    }),
    ([*Random Forest*], r3(cv.accuracy), r3(cv.precision), r3(cv.recall), r3(cv.f1), r3(cv.roc_auc)),
    span[Prueba (#fmt(test.classification.rows) reservas)],
    ([Logística], r3(p2cls.logistic.test.accuracy), r3(p2cls.logistic.test.precision), r3(p2cls.logistic.test.recall), r3(p2cls.logistic.test.f1), r3(p2cls.logistic.test.roc_auc)),
    ([*Random Forest*], r3(cft.accuracy), r3(cft.precision), r3(cft.recall), r3(cft.f1), r3(cft.roc_auc)),
  ),
  [Clasificación de cancelaciones con umbral de 0,50 (clase positiva: cancelada). Exact.: exactitud; Prec.: precisión; Sens.: sensibilidad.],
  columns: (1.9fr, 1fr, 1fr, 1fr, 1fr, 1fr),
) <tab:cls>

#plot("classification", [Curvas ROC y matriz de confusión de Random Forest en validación.]) <fig:cls>

== Sobreajuste y generalización
Los bosques muestran una brecha moderada entre entrenamiento y validación: $R^2$ de #r3(reg.random_forest.train.r2) frente a #r3(reg.random_forest.validation.r2) en la tarifa y AUC de #r3(cls.random_forest.train.roc_auc) frente a #r3(cv.roc_auc) en la cancelación (fig. @fig:overfit). La profundidad máxima y el mínimo de observaciones por hoja limitan esa memorización. En las MLP, la pérdida de entrenamiento siguió bajando después de la época #p3r.mlp.best_epoch, mientras que la de validación se estabilizó en #fmt(p3r.mlp.best_val_loss, digits: 3)\; la parada temprana conservó los pesos de esa época. En prueba, los resultados fueron similares a los de validación: el $R^2$ prácticamente no cambió y el AUC fue algo mayor, como también ocurrió con la regresión logística, lo que apunta a diferencias entre particiones más que a una ventaja del modelo.

#plot("overfitting", [Métricas de los Random Forest seleccionados en entrenamiento, validación y prueba.]) <fig:overfit>

== Variables más influyentes
Al permutar el mes de llegada, el RMSE del bosque de tarifa aumentó #fmt(m.regression.permutation_importance.at(0).mean) u.m.; le siguieron el hotel, el segmento de mercado y la habitación reservada (fig. @fig:imp). En cancelación, el tipo de cliente, la anticipación, el segmento y las cancelaciones previas produjeron las mayores caídas de AUC. Estas medidas indican cuánto depende el modelo de cada variable, no la dirección ni el efecto causal de la asociación.

#plot("importance", [Importancia por permutación en validación (#p3r.permutation.rows reservas, #p3r.permutation.repeats repeticiones); las barras de error son la desviación estándar.]) <fig:imp>

== Segmentación de las reservas
La silueta fue máxima con $K = #kd.max_silhouette_k$ (#r3(kd.max_silhouette)), pero esa partición cambiaba según la semilla (ARI medio de #r3(k-row(kd.max_silhouette_k).stability_ari_mean)). Con $K = #kd.elbow_k$ coincidieron el codo de la inercia y una estabilidad casi perfecta (ARI medio de #r3(k-row(6).stability_ari_mean)), además de perfiles fáciles de interpretar (fig. @fig:k); por eso se adoptaron seis segmentos como solución descriptiva. Su silueta fue de #r3(clu.silhouette_train) en entrenamiento, #r3(clu.silhouette_validation) en validación y #r3(test.clustering.silhouette) en prueba: una separación moderada, pero consistente entre particiones.

#plot("k_selection", [Inercia, silueta y ARI medio entre tres semillas de K-Means en entrenamiento para $K = 2, dots, 10$. La línea discontinua marca la solución adoptada; la punteada, un umbral de referencia de estabilidad (ARI = 0,85).]) <fig:k>

#let prof(key) = clu.profiles_train.find(p => p.segment == key)
#let ntrain = clu.profiles_train.map(p => p.size).sum()
#let names = (short_lead: "Corto plazo", long_lead: "Anticipada", children: "Familias con niños", baby: "Familias con bebé", loyal: "Clientes fieles", serial_cancellers: "Historial de cancel.")
#tab(
  ([Segmento], [Reservas], [%], [Antic.], [Noches], [Cancel.]),
  ("short_lead", "long_lead", "children", "baby", "loyal", "serial_cancellers").map(k => {
    let p = prof(k)
    (names.at(k), fmt(p.size), fmt(100 * p.size / ntrain, digits: 1), fmt(p.lead_time_median, digits: 0), fmt(p.total_nights_median, digits: 0), [#fmt(100 * p.cancel_rate, digits: 1)~%])
  }),
  [Segmentos de K-Means en entrenamiento: tamaño, medianas de anticipación (días) y noches, y tasa de cancelación observada (no usada para agrupar).],
  columns: (1.9fr, 1.1fr, 0.7fr, 0.8fr, 0.8fr, 0.9fr),
) <tab:seg>

Dos segmentos reúnen el #pct((prof("short_lead").size + prof("long_lead").size) / ntrain) de las reservas (tabla @tab:seg). El de corto plazo tiene una anticipación mediana de #fmt(prof("short_lead").lead_time_median, digits: 0) días y cancela el #pct(prof("short_lead").cancel_rate)\; el de planificación anticipada reserva con unos #fmt(prof("long_lead").lead_time_median, digits: 0) días de antelación y cancela el #pct(prof("long_lead").cancel_rate), aunque la cancelación no intervino en el agrupamiento. Los cuatro restantes son pequeños y se definen por la presencia de niños o bebés, por el historial de reservas mantenidas (clientes fieles) o por cancelaciones previas; este último reúne solo #prof("serial_cancellers").size reservas, casi todas canceladas. DBSCAN formó #clu.dbscan.clusters grupos y dejó el #pct(clu.dbscan.noise_share) de la muestra como ruido (silueta de #r3(clu.dbscan.silhouette) en las reservas asignadas), pero separó sobre todo por composición de huéspedes; K-Means ofreció perfiles más útiles para describir la demanda.

== Aplicación interactiva
Los resultados se integraron en una aplicación web desarrollada con Streamlit @streamlit, organizada en cuatro páginas: Resumen, Predicción, Segmentación y Modelos. Todas las cifras y gráficos se calculan sobre las #fmt(d.business.bookings) reservas depuradas con los modelos ya entrenados. #dashboard-status

*Filtros e indicadores.* Un panel lateral filtra las reservas por rango de fechas de llegada; por hotel, segmento de mercado, tipo de cliente, tipo de depósito y canal; y por rangos de anticipación, ADR y noches. Al cambiar un filtro se recalculan todos los indicadores y gráficos (fig. @fig:dash-resumen), incluido el mapa de segmentos (fig. @fig:dash-mapa). Los indicadores principales son el número de reservas, la tasa de cancelación, la ADR media de las reservas mantenidas y la anticipación media, con su diferencia frente al total cuando hay filtros activos y su evolución mensual. Dos indicadores expresan el valor económico: el _valor mantenido_ y el _valor cancelado_, calculados como la suma de ADR × noches de las reservas mantenidas y canceladas, respectivamente. El segundo es un valor estimado de las reservas canceladas, no una pérdida realizada, porque los datos no registran si la habitación se revendió ni si se cobró alguna penalización. Unas tarjetas resumen hallazgos de la selección y, debajo, se muestran el $R^2$, el RMSE, el AUC, el F1 y la silueta de validación de los modelos, con la mejora frente a las referencias cuando procede.

#shot("dash_resumen", [Página de resumen con el filtro «Hotel urbano (Lisboa)» activo: panel de filtros completo (izquierda), cabecera, seis indicadores con su diferencia frente al total y su evolución mensual, tarjetas de hallazgos y el inicio de las pestañas con los dos primeros gráficos (derecha).]) <fig:dash-resumen>

#shot("dash_mapa", [Mapa de segmentos en las dos primeras componentes principales (muestra de reservas que mantiene visibles los grupos pequeños).], wide: false, width: 70%) <fig:dash-mapa>

*Predicción en vivo.* Un formulario recoge las características de una reserva nueva —hotel, mes, anticipación, noches, huéspedes, segmento, canal, tipo de cliente, régimen, habitación e historial— y los modelos responden al instante (fig. @fig:dash-pred). Para la cancelación se muestra la probabilidad estimada por el bosque en un indicador con bandas de riesgo (bajo, moderado y alto), cuántas veces supera a la tasa media de cancelación y la decisión con el umbral de 0,50. Esta probabilidad sirve para ordenar reservas por riesgo, pero su calibración no se verificó, por lo que no debe leerse como una frecuencia exacta. Para la tarifa se muestran la ADR estimada, el rango entre los percentiles 10 y 90 de los árboles (una medida de dispersión del bosque, no un intervalo de confianza), el valor de la estancia si no se cancela y un histograma de reservas reales comparables. La página indica además el segmento asignado a la reserva.

#shot("dash_prediccion", [Página de predicción: formulario completo de la reserva (izquierda); probabilidad de cancelación con bandas de riesgo, ADR estimada con su rango entre árboles, valor de la estancia, histograma de reservas comparables y segmento asignado (derecha).]) <fig:dash-pred>

*Segmentación.* La página de segmentos presenta una tarjeta por grupo (peso, tasa de cancelación y ADR media), un mapa interactivo de las reservas en las componentes principales, en 2D o 3D (fig. @fig:dash-mapa), y un mapa de calor con el perfil numérico medio de cada segmento (fig. @fig:dash-perfiles). El usuario puede situar en el mapa la reserva simulada en la página de predicción. Las dos primeras componentes explican solo el #pct(clu.pca_explained_variance.at(0) + clu.pca_explained_variance.at(1)) de la varianza, de modo que el mapa es una ayuda visual: grupos que se solapan en el plano pueden estar separados en las siete dimensiones originales. Por último, la página de modelos reúne las métricas frente a las referencias, la importancia de las variables, los residuos, la curva ROC, una matriz de confusión con umbral ajustable y los criterios de selección de $K$.

#shot("dash_perfiles", [Perfil medio de cada segmento en la aplicación. El número es la media del segmento y el color, cuántas desviaciones típicas se aleja de la media general.], width: 78%) <fig:dash-perfiles>
