#import "presentacion.typ": *
#let cft = test.classification.random_forest
#let cmt = cft.confusion_matrix
#let rft = test.regression.random_forest
#let prof(key) = clu.profiles_train.find(p => p.segment == key)
= Discusión

== Respuesta a las preguntas de investigación
*P1 (calidad).* Los datos son completos en casi todas las variables, pero no están libres de problemas: ausencias no aplicables en agente y empresa, una ADR muy asimétrica con valores imposibles o extremos, reservas sin noches ni huéspedes y #fmt(d.eda.quality.duplicates_extra) registros repetidos adicionales. Además, varias columnas se registran después del alta. El problema más relevante para el modelado no fue la falta de datos, sino la disponibilidad temporal de los predictores.

*P2 (asociaciones).* La tarifa depende sobre todo del calendario, el hotel y el producto reservado. La cancelación crece con la anticipación y difiere según el tipo de cliente, el segmento y el historial del huésped. Las importancias por permutación confirman en los modelos multivariables lo observado en el análisis exploratorio. Ninguno de estos resultados permite una interpretación causal.

*P3 (utilidad de los modelos).* Random Forest explica tres cuartas partes de la varianza de la ADR en reservas no vistas y ordena el riesgo de cancelación con un AUC superior a 0,8. Ambos superan claramente a sus referencias. Sin embargo, con el umbral por defecto solo detecta menos de la mitad de las cancelaciones. La pregunta principal queda respondida dentro del diseño retrospectivo: se puede estimar la tarifa y ordenar el riesgo sin usar el desenlace, aunque con predictores cuya disponibilidad al crear la reserva no está garantizada.

== Impacto para la gestión hotelera
Las cancelaciones representan el #pct(d.business.lost_share) del valor reservado (ADR por noches) en los datos limpios. En prueba, el bosque emitió #fmt(cmt.at(0).at(1) + cmt.at(1).at(1)) alertas, de las que #fmt(cmt.at(1).at(1)) fueron cancelaciones reales: una lista corta y fiable para acciones de retención o confirmación. A cambio, omitió #fmt(cmt.at(1).at(0)) cancelaciones. Si el objetivo es dimensionar la sobreventa, conviene bajar el umbral y aceptar más falsas alertas. Esa decisión depende del coste relativo de cada error, que los datos no contienen; el tablero permite explorarla con el umbral ajustable. El error medio de la tarifa, #fmt(rft.mae) u.m. por noche, sirve para previsiones agregadas y rangos de referencia, pero no para fijar precios individuales. El segmento anticipado, con casi un tercio de las reservas y una cancelación del #pct(prof("long_lead").cancel_rate), es el candidato natural para políticas de depósito o confirmación.

== Cumplimiento de los objetivos
La tabla @tab:smart resume la evaluación de los objetivos específicos.

#tab(
  ([Obj.], [Indicador verificable], [Estado]),
  (
    ([O1], [32 variables descritas; nulos, repetidos y anomalías cuantificados @fase1], [Cumplido]),
    ([O2], [#fmt(d.cleaning.removed_rows) exclusiones trazadas; 2 variables derivadas; particiones por huella sin solapamiento; referencias en validación y prueba @fase2], [Cumplido]),
    ([O3], [6 clasificadores y 3 regresores comparados con referencias constantes; K-Means y DBSCAN; tablero con los 4 módulos; evaluación única en prueba], [Cumplido; URL pública pendiente]),
  ),
  [Evaluación de los objetivos SMART.],
  columns: (0.45fr, 2.6fr, 1fr),
) <tab:smart>

== Limitaciones
La fotografía previa a la llegada impide afirmar que el modelo de cancelación sirva como alerta al crear la reserva: la habitación reservada, el segmento o el tipo de cliente pueden haberse actualizado después. Las particiones son aleatorias por grupos y no temporales, por lo que no miden el desempeño en periodos futuros ni en otros hoteles; además, #fmt(p2.classification.x_pattern_overlap.train_validation) perfiles de predictores coinciden entre entrenamiento y validación aunque correspondan a reservas distintas. La regresión se limita a reservas mantenidas, una población seleccionada por el propio desenlace, y la ADR se expresa en las unidades monetarias de la fuente, sin moneda acreditada. La búsqueda de la Fase 3 evaluó solo tres configuraciones por modelo, por lo que los bosques son los mejores de esa muestra y no necesariamente los óptimos; además, el cuaderno no guardó el ganador y hubo que identificarlo reajustando los candidatos. La reproducción exacta depende de la versión de scikit-learn: con la 1.9 el bosque de ADR varía en la tercera cifra decimal y K-Means converge a otra solución con $K = 6$. La elección de $K$ es descriptiva: la silueta prefiere $K = 4$ y DBSCAN solo se evaluó en una muestra de #fmt(p3r.dbscan.rows) reservas. Los intervalos _bootstrap_ ignoran la dependencia entre reservas del mismo grupo. Finalmente, la interpretabilidad se basó en la importancia por permutación; no se calcularon valores SHAP.
