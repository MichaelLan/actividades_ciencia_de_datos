#import "@preview/charged-ieee:0.1.4": ieee
#import "presentacion.typ": *

// Traducción local: el paquete no expone un parámetro de idioma.
#show: body => {
  set text(lang: "es")
  set page(numbering: "1")
  show "Abstract": "Resumen"
  show "Index Terms": "Palabras clave"
  show "TABLE": "TABLA"
  body
}
#show: ieee.with(
  title: [Demanda hotelera: estimación de tarifas, predicción de cancelaciones y segmentación de reservas con un tablero interactivo],
  abstract: [
    Se estudian #fmt(d.eda.quality.rows) reservas de dos hoteles portugueses para estimar la tarifa diaria media (ADR) de las estancias mantenidas, ordenar el riesgo de cancelación y describir perfiles de reserva. Tras excluir #fmt(d.cleaning.removed_rows) registros inválidos y dividir los datos por grupos de registros idénticos, se compararon modelos de referencia, regresión regularizada, Random Forest, XGBoost y perceptrones multicapa. Los modelos seleccionados se reentrenaron con la configuración ganadora de la Fase 3, reproduciendo exactamente sus métricas de validación, y se evaluaron una sola vez en la partición de prueba. Random Forest alcanzó en prueba un RMSE de ADR de #fmt(test.regression.random_forest.rmse) unidades monetarias ($R^2$ = #fmt(test.regression.random_forest.r2, digits: 3)) y un AUC de cancelación de #fmt(test.classification.random_forest.roc_auc, digits: 3); con umbral de 0,50, el #pct(test.classification.random_forest.precision) de sus alertas fue correcto, pero solo detectó el #pct(test.classification.random_forest.recall) de las cancelaciones. K-Means con seis grupos separó la demanda de corto plazo de la anticipada (silueta de #fmt(test.clustering.silhouette, digits: 3) en prueba). Los resultados se integran en un tablero Streamlit con filtros, indicadores, predicción en vivo y un mapa de segmentos. El diseño es retrospectivo y describe estos dos hoteles en 2015–2017.
  ],
  authors: (
    (name: "Michael Lan Alvarez"),
    (name: "Carlos Arnulfo Cogua Laverde"),
  ),
  index-terms: ([demanda hotelera], [cancelación], [tarifa diaria media], [Random Forest], [K-Means], [tablero interactivo]),
  bibliography: bibliography("refs.bib", title: [Referencias], style: "ieee"),
  figure-supplement: [Fig.],
)
#show raw.where(block: false): set text(size: 7.5pt)
#set figure(gap: 0.6em)
// Las referencias cruzadas muestran solo el número; el texto escribe «fig.» o «tabla».
#show ref: it => {
  let el = it.element
  if el != none and el.func() == figure {
    link(el.location(), numbering(el.numbering, ..el.counter.at(el.location())))
  } else { it }
}

#include "introduccion.typ"
#include "metodologia.typ"
#include "resultados.typ"
#include "discusion.typ"
#include "conclusiones.typ"
#include "trabajo_futuro.typ"
