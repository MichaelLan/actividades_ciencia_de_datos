#import "presentacion.typ": eda, fmt
= Conclusiones
Las tarifas son heterogéneas y presentan una cola derecha pronunciada: la media de ADR (#fmt(eda.quantitative.adr.mean)) supera la mediana (#fmt(eda.quantitative.adr.median)), con extremos que requieren revisión. El hotel urbano registra una mediana mayor que el vacacional (#fmt(eda.relationships.hotel.at(0).adr_median) frente a #fmt(eda.relationships.hotel.at(1).adr_median)). Las asociaciones débiles con duración y anticipación no respaldan una relación monótona fuerte de estas características con la tarifa.

Las cancelaciones afectan al #fmt(100 * eda.coverage.cancellation_prevalence) % de las reservas y son más frecuentes en City Hotel que en Resort Hotel. La frecuencia aumenta con la anticipación y difiere entre segmentos comerciales. El depósito presenta la asociación más marcada de los contrastes seleccionados, aunque su interpretación está condicionada por el momento en que se registra el pago.

La preparación de los datos debe distinguir ausencias no aplicables de información desconocida, revisar ADR anómalos y reservas sin noches o huéspedes conocidos, y controlar los #eda.quality.duplicates_extra registros repetidos adicionales. La atipicidad estadística y la repetición no justifican una eliminación automática.

El conjunto permite plantear modelos básicos de tarifa y cancelación y una futura segmentación con variables de reserva. Su utilidad predictiva deberá comprobarse con particiones independientes y una selección temporalmente válida de características. El diagnóstico proporciona esa base, pero todavía no permite afirmar qué desempeño alcanzarán los modelos.
