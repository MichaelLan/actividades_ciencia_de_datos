#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A Typesetting System to Untangle the Scientific Writing Process],
  abstract: [  ],
  authors: (
    (
      name: "Michael Lan Alvarez",
    ),
    (
      name: "Carlos Arnulfo Cogua Laverde",
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#include "introduccion.typ"
#include "problema.typ"
#include "objetivos.typ"
#include "descripcion_dataset.typ"
#include "analisis_exploratorio.typ"
#include "conclusiones.typ"