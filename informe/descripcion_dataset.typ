#import "presentacion.typ": eda, fmt, variable, tab
= Descripción del conjunto de datos
*Nombre y repositorio.* Hotel Booking Demand, distribuido por Jesse Mostipak en Kaggle @kaggle.

*URL de acceso.* #link("https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand")[https://www.kaggle.com/datasets/jessemostipak/\ hotel-booking-demand].

*Procedencia científica.* Antonio, de Almeida y Nunes, _Data in Brief_ (2019) @antonio2019. Los datos proceden de los sistemas de gestión de un hotel de Lisboa (City Hotel) y otro del Algarve (Resort Hotel), Portugal.

*Licencia.* Creative Commons Attribution 4.0 International (CC BY 4.0), según el repositorio; requiere atribución.

*Extracción.* La fecha de descarga original no puede verificarse; no se le atribuye una fecha de consulta o de procesamiento.

*Cobertura.* Llegadas previstas entre el #eda.coverage.arrival_min y el #eda.coverage.arrival_max, incluidas reservas canceladas.

*Dimensiones y unidad.* #eda.quality.rows registros y #eda.quality.columns variables. Cada fila representa una reserva. El diccionario completo se presenta en el anexo A.

#let entries = (
  ("hotel", "N", [Tipo de establecimiento.], [City Hotel; Resort Hotel.]),
  ("is_canceled", "N", [Indicador de cancelación.], [0: no; 1: sí.]),
  ("lead_time", "CD", [Días entre alta de la reserva y llegada prevista.], [Enteros ≥ 0.]),
  ("arrival_date_year", "CD", [Año de llegada prevista.], [2015–2017.]),
  ("arrival_date_month", "N", [Mes de llegada prevista.], [January–December (12 meses).]),
  ("arrival_date_week_number", "CD", [Semana de llegada prevista.], [1–53.]),
  ("arrival_date_day_of_month", "CD", [Día del mes de llegada.], [1–31, según mes y año.]),
  ("stays_in_weekend_nights", "CD", [Noches reservadas en sábado o domingo.], [Enteros ≥ 0.]),
  ("stays_in_week_nights", "CD", [Noches reservadas de lunes a viernes.], [Enteros ≥ 0.]),
  ("adults", "CD", [Número de adultos.], [Enteros ≥ 0.]),
  ("children", "CD", [Número de niños.], [Enteros ≥ 0.]),
  ("babies", "CD", [Número de bebés.], [Enteros ≥ 0.]),
  ("meal", "N", [Régimen de alimentación.], [BB: desayuno; HB: media pensión; FB: pensión completa; SC/Undefined: sin paquete.]),
  ("country", "N", [País de origen.], [Códigos de país en letras.]),
  ("market_segment", "N", [Segmento comercial; TA: agencia; TO: operador.], [Aviation, Complementary, Corporate, Direct, Groups, Offline TA/TO, Online TA, Undefined.]),
  ("distribution_channel", "N", [Canal de distribución.], [Corporate, Direct, GDS, TA/TO, Undefined.]),
  ("is_repeated_guest", "N", [Cliente con reservas anteriores.], [0: no; 1: sí.]),
  ("previous_cancellations", "CD", [Reservas previas canceladas del mismo perfil.], [Enteros ≥ 0.]),
  ("previous_bookings_not_canceled", "CD", [Reservas previas no canceladas del mismo perfil.], [Enteros ≥ 0.]),
  ("reserved_room_type", "N", [Tipo de habitación reservado, anonimizado.], [A, B, C, D, E, F, G, H, L, P.]),
  ("assigned_room_type", "N", [Tipo de habitación asignado, anonimizado.], [A, B, C, D, E, F, G, H, I, K, L, P.]),
  ("booking_changes", "CD", [Cambios hasta la llegada o cancelación.], [Enteros ≥ 0.]),
  ("deposit_type", "N", [Pago previo respecto al coste de la estancia.], [No Deposit: ninguno; Non Refund: al menos el total; Refundable: menor que el total.]),
  ("agent", "N", [Código anonimizado de agencia.], [Códigos nominales; NULL: no aplicable.]),
  ("company", "N", [Código de empresa que reserva o paga.], [Códigos nominales; NULL: no aplicable.]),
  ("days_in_waiting_list", "CD", [Días en espera hasta confirmar la reserva.], [Enteros ≥ 0.]),
  ("customer_type", "N", [Modalidad de cliente.], [Contract: contrato; Group: grupo; Transient: individual; Transient-Party: vinculado a otras reservas.]),
  ("adr", "CC", [Tarifa diaria media: ingresos de alojamiento por noche.], [Valor monetario; sin límites normativos publicados.]),
  ("required_car_parking_spaces", "CD", [Plazas de aparcamiento solicitadas.], [Enteros ≥ 0.]),
  ("total_of_special_requests", "CD", [Número de solicitudes especiales.], [Enteros ≥ 0.]),
  ("reservation_status", "N", [Último estado de la reserva.], [Canceled; Check-Out; No-Show.]),
  ("reservation_status_date", "O", [Fecha del último estado.], [Fecha válida (AAAA-MM-DD).]),
)

#let dataset-dictionary = [
  = Anexo A. Diccionario de datos
  *N*: cualitativa nominal; *O*: cualitativa ordinal (fecha ordenada); *CD*: cuantitativa discreta; *CC*: cuantitativa continua. Los componentes de calendario se describen por su función temporal. Los rangos de conteos son semánticos; no son umbrales de exclusión. Los códigos de agencia, empresa y habitación no tienen magnitud ni jerarquía. Definiciones basadas en la fuente científica @antonio2019.
  #for i in range(4) {
    tab(([Variable], [Tipo], [Descripción funcional], [Rango o valores permitidos]),
      entries.slice(i * 8, (i + 1) * 8).map(r => (variable(r.at(0)), r.at(1), r.at(2), r.at(3))),
      [Diccionario de las 32 variables (#(i + 1)/4). Fuente: Hotel Booking Demand @antonio2019.],
      columns: (1.3fr, 0.45fr, 1.4fr, 1.6fr))
  }
  #let calendar = ("arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month")
  #let measures = (("mean", [Media]), ("median", [Mediana]), ("mode", [Moda]), ("std", [DE]), ("skewness", [Asimetría]), ("kurtosis", [Curtosis]))
  #tab(([Medida], [Año], [Semana], [Día del mes]),
    measures.map(((key, label)) => (label, ..calendar.map(name => {
      let value = eda.quantitative.at(name).at(key)
      if key == "mode" { value.map(x => fmt(x)).join(", ") } else { fmt(value) }
    }))),
    [Resumen complementario del calendario de llegada. Sus componentes describen la cobertura temporal; no se interpretan como magnitudes económicas ni se incluyen en la correlación. Curtosis: exceso de Fisher.])
]
