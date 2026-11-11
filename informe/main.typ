#import "@preview/charged-ieee:0.1.4": ieee
#import "presentacion.typ": eda, fmt

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
  title: [Demanda hotelera: planteamiento y análisis exploratorio de Hotel Booking Demand],
  abstract: [Se analizan #eda.quality.rows reservas de dos hoteles portugueses para estudiar su tarifa diaria media (ADR) y su cancelación. El diagnóstico combina estadísticas descriptivas, visualizaciones, asociaciones y evaluación de calidad de las 32 variables. ADR muestra una distribución asimétrica y una mediana mayor en el hotel urbano. La cancelación afecta al #fmt(100 * eda.coverage.cancellation_prevalence) % de las reservas y aumenta con la anticipación. Las ausencias en intermediarios, los valores extremos y los patrones repetidos orientan la preparación de los datos. Los resultados fundamentan el modelado posterior, considerando que los registros describen una fotografía previa a la llegada y no el momento de creación de la reserva.],
  authors: (
    (name: "Michael Lan Alvarez"),
    (name: "Carlos Arnulfo Cogua Laverde"),
  ),
  index-terms: ([demanda hotelera], [análisis exploratorio], [calidad de datos], [cancelación], [tarifa diaria media]),
  bibliography: bibliography("refs.bib", title: [Referencias], style: "ieee"),
  figure-supplement: [Fig.],
)
#show raw.where(block: false): set text(size: 8pt)

#include "introduccion.typ"
#include "problema.typ"
#include "objetivos.typ"
#include "descripcion_dataset.typ"
#include "analisis_exploratorio.typ"
#include "conclusiones.typ"

#import "descripcion_dataset.typ": dataset-dictionary
#import "analisis_exploratorio.typ": categorical-tables
#dataset-dictionary
#categorical-tables
