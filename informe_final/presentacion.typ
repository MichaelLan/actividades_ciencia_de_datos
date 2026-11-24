// Datos y utilidades de presentación. Las cifras proceden de datos.json, generado por
// generate_report_data.py a partir de los JSON de las fases 1–4; no se escriben a mano.
#let d = json("datos.json")
#let m = d.metrics
#let reg = m.regression.metrics
#let cls = m.classification.metrics
#let clu = m.clustering
#let p3 = m.phase3_reference
#let p3r = d.phase3_report
#let nb = d.phase3_notebook
#let test = d.test
#let p2 = d.phase2.tasks

// URL pública del tablero: sustituir "PENDIENTE" por la dirección de Streamlit Community Cloud.
#let dashboard-url = "PENDIENTE"
#let dashboard-link = if dashboard-url == "PENDIENTE" {
  text(fill: rgb("#d03b3b"))[[URL de despliegue pendiente]]
} else { link(dashboard-url, raw(dashboard-url)) }

// Número decimal con coma; enteros con espacio fino de miles a partir de 10 000.
#let fmt(value, digits: 2, pad: true) = {
  if value == none { [—] }
  else if type(value) == int or (type(value) == float and digits == 0) {
    let s = str(int(calc.round(value)))
    let neg = s.starts-with("-")
    if neg { s = s.slice(1) }
    if s.len() > 4 {
      let out = ""
      let n = s.len()
      for (i, c) in s.clusters().enumerate() {
        if i > 0 and calc.rem(n - i, 3) == 0 { out += "\u{202F}" }
        out += c
      }
      s = out
    }
    if neg { "−" + s } else { s }
  } else {
    let r = calc.round(value, digits: digits)
    let s = str(r)
    if not pad { }
    else if not s.contains(".") and digits > 0 { s += "." + "0" * digits }
    else if s.contains(".") {
      let dec = s.split(".").at(1).len()
      if dec < digits { s += "0" * (digits - dec) }
    }
    s.replace(".", ",").replace("-", "−")
  }
}
#let pct(value, digits: 1) = [#fmt(100 * value, digits: digits)#sym.space.nobreak%]

#let tab(headers, rows, caption, columns: auto, wide: false, align: left) = figure(
  {
    set text(size: 7.5pt, hyphenate: false)
    set par(first-line-indent: 0pt, leading: 0.4em, justify: false)
    table(
      columns: if columns == auto { (1.6fr, ..range(headers.len() - 1).map(_ => 1fr)) } else { columns },
      inset: (x: 2.5pt, y: 2.4pt), align: align,
      stroke: none,
      table.hline(stroke: 0.6pt),
      table.header(..headers.map(h => strong(h))),
      table.hline(stroke: 0.4pt),
      ..rows.flatten(),
      table.hline(stroke: 0.6pt),
    )
  }, kind: table, caption: caption,
  placement: if wide { top } else { none },
  scope: if wide { "parent" } else { "column" },
)
#let plot(name, caption, wide: false, ext: "svg", placement: auto) = figure(
  image("imagenes/" + name + "." + ext, width: 100%),
  caption: caption, kind: image,
  placement: if wide { top } else if placement == auto { none } else { placement },
  scope: if wide { "parent" } else { "column" },
)
