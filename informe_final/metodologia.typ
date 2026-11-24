#import "presentacion.typ": *
#let cl = p2.classification
#let rg = p2.regression
= Metodología

== Datos y diagnóstico inicial
El archivo combinado de Kaggle (licencia CC BY 4.0) contiene #fmt(d.eda.quality.rows) filas y #d.eda.quality.columns columnas: #fmt(d.eda.coverage.hotel_counts.at(0).len) reservas del hotel urbano y #fmt(d.eda.coverage.hotel_counts.at(1).len) del vacacional @antonio2019 @kaggle. La Fase 1 calculó estadísticas univariadas de las 16 variables cuantitativas y frecuencias de las categóricas, correlaciones de Spearman (preferidas por las colas largas y las relaciones monótonas) y contrastes $chi^2$ con la V de Cramér como tamaño del efecto @fase1. Se trataron como asociaciones exploratorias: las reservas repetidas pueden violar la independencia y no se corrigió por multiplicidad.

== Preparación de los datos
*Limpieza.* Se aplicaron reglas de dominio, no umbrales estadísticos. Se excluyeron #fmt(d.cleaning.removed_rows) reservas (#pct(d.cleaning.removed_rows / d.cleaning.input_rows, digits: 2)): #d.cleaning.rules.negative_adr con ADR negativa, #d.cleaning.rules.zero_guests sin huéspedes conocidos y #d.cleaning.rules.zero_nights sin noches; las reglas se solapan, por lo que sus conteos no se suman. Quedaron #fmt(d.cleaning.retained_rows) reservas. Se conservaron las ADR nulas (posibles cortesías), los extremos positivos finitos y los #fmt(d.eda.quality.duplicates_extra) registros repetidos adicionales, porque una fila idéntica puede corresponder a otra reserva real.

*Valores atípicos.* La regla de Tukey ($Q_1 - #[1,5] thin "IQR"$, $Q_3 + #[1,5] thin "IQR"$) @tukey1977 se usó como bandera descriptiva. En los conteos inflados en cero el IQR es nulo y la regla no es informativa. Como prueba de sensibilidad se recortó la ADR de entrenamiento en su percentil 99 (252, calculado solo con entrenamiento): en validación, el RMSE de la regresión lineal aumentó en 0,042 y el MAE bajó en 0,047. Se mantuvo la ADR original, porque el objetivo es estimar también tarifas altas plausibles.

*Ausencias.* De las cuatro variables con nulos, `agent` (#pct(d.eda.quality.missing.agent.percent / 100)) y `company` (#pct(d.eda.quality.missing.company.percent / 100)) indican «no aplicable» según la fuente y no se imputaron. Los #d.eda.quality.missing.children.count valores ausentes de `children` se concentran en cancelaciones del hotel urbano: un mecanismo MAR es plausible, sin excluir MNAR @rubin1976. Se imputó su mediana dentro del flujo de entrenamiento. `country` se excluyó porque puede registrarse en el check-in.

*Variables derivadas y fuga de información.* Se crearon `total_nights` (noches de fin de semana más laborables), que sustituye a sus componentes, y `total_guests`, conservada para trazabilidad pero fuera de los predictores por ser redundante con adultos, niños y bebés. Se excluyeron el estado de la reserva y su fecha (desenlace), la habitación asignada, los cambios, la lista de espera, el depósito, las solicitudes y el aparcamiento, porque pueden registrarse después del alta. Los modelos supervisados usan siete variables numéricas y ocho categóricas: hotel, mes de llegada, régimen, segmento, canal, cliente repetido, habitación reservada y tipo de cliente. Un `ColumnTransformer` imputa y estandariza las numéricas y codifica las categóricas con _one-hot_ (ignorando categorías nuevas). Se ajusta solo con entrenamiento, dentro de cada `Pipeline` y de cada pliegue @pedregosa2011.

== Diseño de las particiones
Se fijó una proporción de 75/12,5/12,5~% para entrenamiento, validación y prueba. Las reservas con la misma huella de las 32 columnas originales se asignaron íntegras a una sola partición, de modo que ninguna fila repetida aparece en dos conjuntos (solapamiento de huellas = 0). En clasificación, la asignación se estratificó por clase: la prevalencia de cancelación fue #pct(cl.splits.train.cancellation_prevalence, digits: 2) en las tres particiones (#fmt(cl.splits.train.rows), #fmt(cl.splits.validation.rows) y #fmt(cl.splits.test.rows) reservas). La regresión de ADR se restringió a las #fmt(rg.population_rows) reservas mantenidas (#fmt(rg.splits.train.rows)/#fmt(rg.splits.validation.rows)/#fmt(rg.splits.test.rows)), porque la ADR de una reserva cancelada no representa un ingreso realizado. La clase minoritaria supera el 10~%, por lo que no se aplicó remuestreo.

#figure(
  image("imagenes/pipeline.svg", width: 100%),
  caption: [Ciclo de vida del proyecto. Cada fase consume los artefactos versionados de la anterior.],
  placement: top, scope: "parent",
) <fig:pipeline>

== Modelos supervisados
Las referencias fueron la media o la clase mayoritaria, la regresión lineal y la regresión logística @fase2. En la Fase 3 se añadieron ElasticNet, que combina penalizaciones L1 y L2 para comprobar si la regularización mejora el modelo lineal @zou2005; Random Forest, que capta interacciones no lineales entre mes, hotel y habitación sin requerir escalamiento @breiman2001; XGBoost @chen2016; y tres perceptrones multicapa (MLP) en PyTorch @paszke2019. ElasticNet, los dos bosques y XGBoost se ajustaron con `RandomizedSearchCV` (semilla 42): #p3r.configurations_per_model configuraciones muestreadas de un espacio predefinido y #p3r.cv_folds pliegues que mantienen juntos los grupos de huella (`GroupKFold` en regresión y `StratifiedGroupKFold` en clasificación). El reajuste final usó el candidato de menor RMSE para la tarifa y de mayor AUC para la cancelación (tabla @tab:search). Las MLP (#p3r.mlp.candidates.map(h => h.map(str).join("–")).join(", ") neuronas ocultas; AdamW, lotes de #p3r.mlp.batch, hasta #p3r.mlp.max_epochs épocas) se detuvieron cuando la pérdida de validación dejó de mejorar durante #p3r.mlp.patience épocas y se ordenaron por AUC de validación @fase3 @fase3nb. La importancia de las variables se estimó por permutación en #p3r.permutation.rows reservas de validación con #p3r.permutation.repeats repeticiones @breiman2001.

#let sr = m.regression.search.space
#let sc = m.classification.search.space
#let rp = m.regression.params
#let cp = m.classification.params
#let mf(v) = if v == "sqrt" { $sqrt(p)$ } else if v == "log2" { $log_2 p$ } else { fmt(v, digits: if v == 1 { 1 } else { 2 }, pad: v == 1) }
#tab(
  ([Hiperparámetro], [Espacio], [ADR], [Cancel.]),
  (
    ([Árboles], [35–90], str(rp.n_estimators), str(cp.n_estimators)),
    ([Profundidad máx.], sr.max_depth.map(str).join(", "), str(rp.max_depth), str(cp.max_depth)),
    ([Mín. para dividir], [2–20], str(rp.min_samples_split), str(cp.min_samples_split)),
    ([Mín. por hoja], [1–10], str(rp.min_samples_leaf), str(cp.min_samples_leaf)),
    ([Variables por nodo], [#sr.max_features.map(mf).join("; ") (ADR) \ #sc.max_features.map(mf).join("; ") (cancel.)], mf(rp.max_features), mf(cp.max_features)),
    ([Peso de clase], [ninguno, balanceado], [—], [ninguno]),
    ([Criterio de CV], [3 candidatos × 3 pliegues], [RMSE], [AUC]),
  ),
  [Búsqueda aleatoria de los Random Forest en la Fase 3 y configuración ganadora; $p$ es el número de columnas tras la codificación.],
  columns: (1.35fr, 1.7fr, 0.6fr, 0.65fr),
) <tab:search>

== Segmentación
K-Means @lloyd1982 se aplicó a las siete variables numéricas estandarizadas (anticipación, adultos, niños, bebés, noches, cancelaciones previas y reservas previas mantenidas), sin la ADR ni la cancelación, sobre la partición de segmentación de la Fase 2: mismos tamaños, pero sin estratificar por la clase. Para $K = 2, dots, 10$ se examinaron la inercia (codo), la silueta @rousseeuw1987 en una muestra de #fmt(clu.silhouette_sample_size) reservas y la estabilidad entre #p3r.kmeans.seeds semillas mediante el índice de Rand ajustado (ARI) @hubert1985, con #clu.kmeans_settings.n_init inicializaciones por ajuste. La regla automática exigía un $K$ vecino del codo, con silueta a menos de 0,02 del máximo y ARI medio de al menos 0,85. Como segundo método se exploró DBSCAN @ester1996 en #fmt(p3r.dbscan.rows) reservas de entrenamiento: los cuatro radios candidatos fueron los cuantiles 0,85–0,975 de la distancia al décimo vecino y se combinaron con 5, 10 y 20 vecinos mínimos. Se eligió la configuración de mayor silueta entre las que formaban de 2 a 20 grupos con al menos un 50~% de cobertura.

== Reentrenamiento y evaluación final
El cuaderno de la Fase 3 (copia local en `fase_3/Fase_3_Hotel_Booking.ipynb`) registra los tres candidatos de cada búsqueda, pero no imprime el ganador. Se reajustaron los tres candidatos de cada bosque en entrenamiento: solo uno reprodujo las métricas de validación del cuaderno, y también ocupaba el primer lugar en validación cruzada (tabla @tab:search). En la Fase 4, `training/train.py` reentrena esos bosques, sus referencias, K-Means con $K = #clu.k$ y el piloto DBSCAN con la misma semilla, el mismo preprocesador y scikit-learn #m.environment.scikit_learn, la versión del cuaderno. Escribe además los artefactos y las métricas de entrenamiento y validación. Con los modelos ya congelados, `training/evaluate_test.py` los carga y los evalúa una sola vez en la partición de prueba, sin reajustar hiperparámetros ni el umbral de 0,50. Los intervalos del 95~% se obtuvieron con #fmt(1000) remuestreos _bootstrap_ por filas.
