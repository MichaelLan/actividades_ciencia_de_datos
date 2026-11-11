#import "datos.typ": results
#let eda = results.eda
#let fmt(value, digits: 2) = {
  if value == none { [—] }
  else if type(value) == int { str(value) }
  else { str(calc.round(value, digits: digits)).replace(".", ",") }
}
#let variable(name) = text(hyphenate: false, name.replace("_", "_\u{200b}"))
#let aliases = (
  "lead_time": "Anticipación", "stays_in_weekend_nights": "Noches fin sem.",
  "stays_in_week_nights": "Noches labor.", "adults": "Adultos",
  "children": "Niños", "babies": "Bebés", "previous_cancellations": "Cancel. previas",
  "previous_bookings_not_canceled": "No cancel. prev.", "booking_changes": "Cambios",
  "days_in_waiting_list": "Espera", "adr": "ADR",
  "required_car_parking_spaces": "Aparcamiento", "total_of_special_requests": "Solicitudes",
  "arrival_date_year": "Año llegada", "arrival_date_week_number": "Semana llegada",
  "arrival_date_day_of_month": "Día llegada",
)
#let tab(headers, rows, caption, columns: auto, wide: false) = {
  set text(size: 8pt, hyphenate: false)
  set par(first-line-indent: 0pt, leading: 0.4em, justify: false)
  figure(
    table(
      columns: if columns == auto { (1.5fr, ..range(headers.len() - 1).map(_ => 1fr)) } else { columns },
      inset: 2.5pt, align: left,
      stroke: (bottom: 0.3pt + luma(180)),
      table.header(..headers.map(h => strong(h))),
      ..rows.flatten(),
    ), kind: table, caption: caption,
    placement: if wide { top } else { none },
    scope: if wide { "parent" } else { "column" },
  )
}
#let plot(name, caption, wide: false) = figure(
  image("imagenes/" + name + ".svg", width: 100%),
  caption: caption, kind: image,
  placement: if wide { top } else { none },
  scope: if wide { "parent" } else { "column" },
)
#let domain(name) = {
  let d = eda.dictionary.at(name).observed_domain
  if type(d) == dictionary { [#fmt(d.min)–#fmt(d.max)] }
  else if name == "reservation_status_date" { [#d.first()–#d.last()] }
  else if name == "arrival_date_month" { [#d.len() meses del calendario] }
  else if name in ("country", "agent", "company") { [#d.len() códigos nominales] }
  else { d.map(v => str(v)).join(", ") }
}
