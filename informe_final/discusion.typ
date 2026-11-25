#import "presentacion.typ": *
#let cft = test.classification.random_forest
#let cmt = cft.confusion_matrix
#let rft = test.regression.random_forest
#let prof(key) = clu.profiles_train.find(p => p.segment == key)
= Discusión

== Tarifas y cancelaciones
La tarifa depende sobre todo del calendario, el hotel y el producto reservado: el mes de llegada fue la variable más influyente del modelo de ADR, en línea con la fuerte estacionalidad observada en el análisis exploratorio. La cancelación, en cambio, se asocia sobre todo con el comportamiento de la reserva: crece con la anticipación y difiere según el tipo de cliente, el segmento de mercado y el historial del huésped. Ambas relaciones son asociaciones observadas en dos hoteles, no efectos causales.

== Ventaja frente a las referencias
Random Forest mejoró claramente a los modelos de referencia en las dos tareas: redujo en una cuarta parte el error de la tarifa respecto a la regresión lineal y elevó el AUC de cancelación de #fmt(m.classification.metrics.logistic.validation.roc_auc, digits: 3) a #fmt(cls.random_forest.validation.roc_auc, digits: 3) en validación. Que ElasticNet no mejorara al modelo lineal y que XGBoost y las redes neuronales quedaran a menos de 0,01 de AUC del bosque indica que la ganancia procede de modelar relaciones no lineales e interacciones, no de un algoritmo concreto. Aun así, ninguno de los dos modelos es infalible. Un error medio de #fmt(rft.mae) u.m. por noche basta para previsiones agregadas y rangos de referencia, pero no para fijar el precio de una reserva individual. En cancelación, la lista de alertas es fiable (#pct(cft.precision) de precisión), pero incompleta: con el umbral de 0,50 se omitieron #fmt(cmt.at(1).at(0)) de #fmt(cmt.at(1).at(0) + cmt.at(1).at(1)) cancelaciones de prueba. Un umbral más bajo detectaría más cancelaciones a cambio de más falsas alertas; la elección depende del coste relativo de cada error, que los datos no contienen.

== Qué representan los segmentos
Los seis segmentos son una descripción útil de la demanda, no categorías naturales: la silueta indica una separación moderada y el número de grupos se eligió por estabilidad e interpretabilidad. La distinción principal es temporal. Las reservas de corto plazo y las de planificación anticipada concentran casi toda la demanda y difieren mucho en su tasa de cancelación (#pct(prof("short_lead").cancel_rate) frente a #pct(prof("long_lead").cancel_rate)), aunque esta variable no se usó para agrupar. El segmento anticipado es, por tanto, un candidato razonable para estudiar políticas de depósito o confirmación; su eficacia no puede deducirse de estos datos y debería evaluarse con un experimento o un seguimiento posterior.

== Uso de los resultados en la aplicación
La aplicación traslada estos resultados a decisiones concretas. Los filtros permiten comparar hoteles, canales o temporadas y ver cómo cambian la cancelación y la tarifa. El valor cancelado dimensiona la exposición: en los datos depurados, las reservas canceladas suman el #pct(d.business.lost_share) del valor reservado (ADR × noches). Esta cifra es el valor estimado de las reservas canceladas y no debe interpretarse como ingreso perdido, porque parte de esas habitaciones pudo revenderse o generar penalizaciones. El simulador permite comprobar cómo varían el riesgo y la tarifa al modificar una reserva, y la matriz de confusión con umbral ajustable muestra el equilibrio entre alertas correctas y cancelaciones omitidas. En conjunto, el trabajo responde a la pregunta principal dentro de su diseño retrospectivo: es posible estimar la tarifa y ordenar el riesgo de cancelación sin usar el desenlace, con un desempeño claramente superior al de las referencias, y comunicarlo de forma interactiva.

== Limitaciones
*Representatividad.* Los datos proceden de dos hoteles portugueses entre 2015 y 2017; los resultados no se pueden generalizar sin más a otros hoteles, mercados o periodos.

*Disponibilidad temporal.* Los registros son una fotografía previa a la llegada. Variables como la habitación reservada, el segmento o el tipo de cliente pueden haberse actualizado después de crear la reserva, por lo que el modelo de cancelación no acredita por sí solo que funcione como alerta temprana en el momento del registro.

*Validación.* Las particiones son aleatorias, con las filas idénticas agrupadas, y no cronológicas. La similitud entre validación y prueba indica estabilidad dentro del periodo estudiado, pero no demuestra el rendimiento en reservas futuras.

*Regresión.* La ADR se evaluó solo en reservas no canceladas, una población seleccionada por el propio desenlace, y se expresa en las unidades monetarias de la fuente.

*Segmentación.* Los perfiles son descriptivos y su separación es moderada; la proyección PCA de la aplicación conserva solo una parte de la varianza.
