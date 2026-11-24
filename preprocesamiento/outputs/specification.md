# T1 — Requisitos y evidencia para fases 1 y 2

**Resultado:** guía completa inspeccionada y extraída en `guide.txt` (8 páginas físicas). Este documento cubre S1 y fundamenta S3/S6/S9; no certifica implementación, descarga del CSV ni resultados experimentales. Fuentes públicas consultadas el **2026-11-10 UTC**. Leer primero el checklist; después, las precauciones temporales antes de elegir predictores.

## Checklist exacto de 3.1 y 3.2

Transcripción del PDF, páginas físicas 3–4: se normalizan únicamente acentos tipográficos y cortes de línea; cada casilla corresponde al texto de una viñeta. Casillas abiertas = requisitos pendientes de demostrar en T2/T3, no fallos de esta inspección. `guide.txt` conserva la extracción original sin esta normalización.

### 3.1. FASE 1: Planteamiento del Problema, Descripción y EDA (Entregable 1 – 20%)

Fechas de Trabajo: 20 Oct – 01 Nov — Fecha Límite de Entrega: Domingo 01 Noviembre.
Requerimientos Técnicos del Informe PDF:

**1. Formulación del Problema y Objetivos:**
- [ ] Contextualización del dominio del problema e impacto potencial de la solución analítica.
- [ ] Formulación de 1 pregunta de investigación principal y 3 preguntas secundarias.
- [ ] Objetivos del proyecto (1 General y 3 Específicos) en formato SMART.

**2. Ficha Técnica del Dataset:**
- [ ] Nombre del dataset, repositorio de origen, URL exacta de acceso, fecha de extracción y tipo de licencia libre.
- [ ] Diccionario de datos exhaustivo: Nombre de la columna, tipo de variable (cuantitativa continua/discreta, cualitativa nominal/ordinal), descripción funcional y rango/valores permitidos.

**3. Análisis Exploratorio de Datos (EDA) Exhaustivo:**
- [ ] Análisis Univariado: Medidas de tendencia central, dispersión, asimetría y curtosis para variables cuantitativas. Tablas de frecuencias para variables cualitativas.
- [ ] Análisis Bivariado y Multivariado: Matriz de correlación (Pearson o Spearman), diagramas de dispersión (scatter plots), y pruebas de independencia (χ2).
- [ ] Diagnóstico de Calidad de Datos: Porcentaje exacto de datos faltantes por variable, detección de valores atípicos (outliers) mediante regla IQR o Z-score, e identificación de incoherencias o duplicados.

### 3.2. FASE 2: Preprocesamiento, Ingeniería de Variables y Modelo Base (Entregable 2 – 25%)

Fechas de Trabajo: 03 Nov – 15 Nov — Fecha Límite de Entrega: Domingo 15 Noviembre.
Requerimientos Técnicos del Cuaderno (Jupyter Notebook / Script Executable):

**Limpieza e Imputación de Datos:**
- [ ] Justificación técnica del tratamiento de valores faltantes (eliminación vs. imputación mediante media, mediana, KNNImputer o IterativeImputer). Prohibido imputar sin analizar el mecanismo de falta (MCAR, MAR, MNAR).
- [ ] Tratamiento razonado de valores atípicos (outliers) mediante remoción, transformación logarítmica o recorte por percentiles (winsorization).

**Ingeniería y Transformación de Características:**
- [ ] Creación de al menos 2 variables derivadas respaldadas en el conocimiento del dominio.
- [ ] Codificación de categóricas (One-Hot Encoding, Target Encoding u Ordinal Encoding) previniendo estrictamente la fuga de datos (Data Leakage).
- [ ] Escalamiento/Estandarización (StandardScaler, MinMaxScaler o RobustScaler) ajustado únicamente sobre el conjunto de entrenamiento.

**Estrategia de Partición de Datos e Imbalance:**
- [ ] División de datos en Entrenamiento (70–80%), Validación y Prueba (20–30%). En clasificación, aplicar división estratificada (Stratified Split).
- [ ] Tratamiento de desbalance severo de clases si aplica (técnicas SMOTE, ADASYN o ajuste de class weight).

**Modelo Base (Baseline):**
- [ ] Implementación de un modelo de referencia sencillo (Regresión Lineal Simple o Regresión Logística básica / Dummy Classifier) para establecer el umbral mínimo de rendimiento.

## Discrepancias y límites de autoridad

- La sección 2.2 exige **mínimo 600** registros limpios; el checklist de sección 5 dice **más de 600**. Cumplir >600 satisface ambos. Exige ≥5 variables mixtas y viabilidad de regresión, clasificación y clustering; esto último no obliga a ejecutar clustering en fase 2.
- Sección 5 menciona «dataset didáctico prohibido», pero la guía completa no enumera datasets prohibidos. Hotel Booking Demand procede de datos reales de PMS y publicación científica; describir esa procedencia, sin inventar una lista.
- La sección 3.1 pide PDF; sección 5 permite informe o cuadernillo. Entregar PDF, opción más estricta. La sección 3.2 permite explícitamente script ejecutable: no hace falta notebook.
- La guía solo pide un baseline sencillo en 3.2; los dos problemas, los Dummy y las métricas detalladas son requisitos adicionales S6/S7, no citas de 3.2. Las métricas completas aparecen en 3.3.
- MLP/PyTorch, SHAP y búsqueda de hiperparámetros figuran en rúbrica/checklist de fase 3 sin desarrollo equivalente en 3.3. Registrar inconsistencia, sin ampliar fases 1/2 (S12).
- Estructura pública /data,/notebooks,/src,/dashboard y requirements/environment se mencionan para fase 4; S2/S10 adelantan reproducibilidad, sin autorizar renombrar la estructura existente.
- «Licencia dataset distinta publicación» se interpreta como **verificación independiente**, no como conclusión: las dos fuentes observadas declaran **CC BY 4.0**, no licencias diferentes. No atribuir CC0 a este dataset.

## Fuentes verificadas y hechos citables

### [F1] Publicación primaria y diccionario

https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ — HTTP 200, artículo completo inspeccionado (146048 caracteres HTML).

N. Antonio, A. de Almeida y L. Nunes, “Hotel booking demand datasets,” *Data in Brief*, vol. 22, pp. 41–49, 2019, doi: **10.1016/j.dib.2018.11.126**. Publicación electrónica: 29 noviembre 2018; colección: febrero 2019. Licencia del artículo: CC BY 4.0, indicada en copyright. No confundir año del DOI/publicación en línea con año del volumen.

Abstract/specifications: H1 Algarve, resort, **40060** reservas; H2 Lisboa, ciudad, **79330**; **31 variables por hotel**. Unidad = reserva, no huésped ni noche. Llegadas previstas entre **2015-07-01 y 2017-08-31**, incluidas cancelaciones. Total de referencia: **119390** reservas; las **32 columnas** esperadas del CSV combinado incorporan `hotel`. Verificar esos valores en el archivo descargado, no presentarlos como medición local.

Tabla 1: `agent` y `company` son identificadores categóricos. El texto explica que `NULL` en estas columnas significa **no aplicable**, no ausencia desconocida: no imputar su media/mediana. Distinguir semántica original de la representación nula que produzca el lector del CSV. `country` y `children` deben diagnosticarse en el CSV, sin afirmar ausencia cero por la frase general del artículo sobre integridad del PMS.

Tablas 4/7 ya muestran ADR -6.38 y 5400: anomalía no equivale automáticamente a error. Son referencias publicadas, no estadísticas locales. El diccionario contiene erratas (`RequiredCardParkingSpaces`, ISO «3155–3»); usar nombres reales del CSV y no reproducir erratas como estándares verificados.

### [F2] Kaggle: licencia y procedencia independientes

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand — HTTP 200; JSON-LD `license.name` = **Attribution 4.0 International (CC BY 4.0)** y `license.url` = https://creativecommons.org/licenses/by/4.0/.

https://www.kaggle.com/api/v1/datasets/view/jessemostipak/hotel-booking-demand — HTTP 200; `licenseName` confirma CC BY 4.0, `currentVersionNumber` = 1, `lastUpdated` = 2020-02-13T01:27:20.8Z. Redistribuidor: Jesse Mostipak. Descripción atribuye origen a Antonio/Almeida/Nunes y descarga/limpieza intermedia a Thomas Mock y Antoine Bichat para TidyTuesday, 2020-02-11. **Fecha de consulta no es fecha de descarga**: T2 debe registrar esta última y versión/hash reales. La respuesta API devuelve `files: []`; no prueba dimensiones, esquema ni descarga exitosa del CSV.

### [F3] API oficial charged-ieee 0.1.4

Fuentes del registro oficial Typst (ambas HTTP 200):
- https://raw.githubusercontent.com/typst/packages/main/packages/preview/charged-ieee/0.1.4/README.md
- https://raw.githubusercontent.com/typst/packages/main/packages/preview/charged-ieee/0.1.4/lib.typ

Exporta `ieee(title:, authors:, abstract:, index-terms:, paper-size:, bibliography:, figure-supplement:, body)`. `body` es posicional; los demás argumentos son nombrados. `authors` = tupla de diccionarios con `name` obligatorio; opcionales `department`, `organization`, `location`, `email`. `index-terms` = tupla de contenidos; `bibliography` recibe `bibliography("refs.bib")` o `none`. Valores predeterminados: autores vacíos, abstract none, papel `us-letter`, suplemento `[Fig.]`.

`#import "@preview/charged-ieee:0.1.4": ieee` y `#show: ieee.with(...)` son válidos. Implementación fija dos columnas, texto 10 pt TeX Gyre Termes, citas IEEE numéricas y bibliografía 8 pt. Encabezados Abstract/Index Terms/References y TABLE están en inglés; no existe argumento `language` ni `columns`. Para español, estudiar reglas locales de presentación y comprobar PDF: no inventar opciones del paquete. No se compiló ni verificó visualmente el informe en T1.

## S6 — Disponibilidad temporal y recomendación de predictores

**Hecho F1:** specifications y sección 1 definen captura **día previo a llegada**, no creación. Cuando había cambios se consultó el log BO/BL. El artículo advierte que personas, duración y habitación pueden cambiar al llegar/durante estancia, y nacionalidad puede conocerse solo al check-in. No hay historial de versiones por reserva en el CSV. Una partición por fecha de llegada no es una validación por fecha de creación.

| Variable | Evidencia F1, tabla 1 | Recomendación conservadora |
|---|---|---|
| `adr` | Suma de transacciones de alojamiento dividida entre noches; BO, BL y TR | Objetivo de regresión en no canceladas; excluir como predictor de clasificación: no está probado que sea tarifa cotizada disponible al crear reserva. Nunca incluirlo en X de regresión. |
| `reservation_status`, `reservation_status_date` | Último estado y fecha de establecimiento; incluyen cancelación, salida, no-show | Exclusión absoluta de X para ambos problemas: resultado/tiempo posterior. |
| `assigned_room_type` | Habitación asignada, puede diferir por operación o petición | Excluir del baseline; no afirmar disponible al reservar. |
| `booking_changes` | Cambios desde alta hasta check-in o cancelación | Excluir: acumulado depende del transcurso/terminación. |
| `days_in_waiting_list` | Días hasta confirmación, BO | Excluir: no disponible antes de confirmar, sin prueba individual de corte. La descripción verbal de resta tiene signo ambiguo: verificar valores, no regenerar fechas. |
| `deposit_type` | Derivada de pagos TR anteriores a llegada **o cancelación**, no mera política contractual | Excluir: información financiera acumulada dependiente del desenlace. |
| `required_car_parking_spaces` | Plazas requeridas por cliente, BO/BL | No confundir con capacidad disponible del hotel. Excluir conservadoramente: falta timestamp individual. |
| `total_of_special_requests` | Suma de solicitudes del cliente, BO/BL | Excluir conservadoramente: solicitudes acumuladas, no comprobadas al alta. |

Lista inicial razonable para ambos X (recomendación, no garantía causal): `hotel`, calendario de llegada, `lead_time`, noches, `adults/children/babies`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `customer_type`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `total_nights`, `total_guests`. Los antecedentes se definen previos a la reserva actual. Excluir `country` en versión estricta por advertencia explícita del artículo; omitir `agent/company` del baseline simplifica cardinalidad, pero pueden ser nominales en EDA. Mes es cíclico/categórico, no magnitud ordinal arbitraria. En regresión omitir también `is_canceled` de X: solo determina población.

Esto sigue siendo **análisis retrospectivo**, no un servicio prospectivo de cancelación al crear reserva: algunas reservas ya estarían canceladas antes del día previo a llegada. Regresión en no canceladas estima ADR condicional a ese desenlace y tiene sesgo de selección; no generalizar a todas las reservas. No puede acreditarse disponibilidad prospectiva de ADR solo por la declaración general del corte temporal del artículo.

## Ampliación T1 para T3: diccionario documentado de las 32 columnas

Fuente de las **31 variables originales**: [F1], **Table 1, “Variables description”**, recuperada nuevamente completa el 2026-11-10 UTC. La columna combinada `hotel` se fundamenta en el abstract/Ficha de F1 y la descripción de F2, **no es una fila de Table 1**. Los nombres snake_case son el mapeo al CSV combinado esperado; T2 debe contrastarlos con su encabezado real.

La columna «tipo» es una **clasificación estadística recomendada para el informe**, no una transcripción literal del almacenamiento: F1 usa Numeric, Integer, Categorical y Date. Las fechas/calendario necesitan explicación propia; no forzar una fecha a nominal/ordinal sin aclarar su estructura temporal. No confundir códigos nominales numéricos con cantidades. Se incluyen solo categorías explícitamente documentadas; **rangos, valores extremos y dominios observados del CSV los proporciona T2**. Los códigos de habitaciones, segmentos, países, agencias y empresas no quedan exhaustivamente enumerados en Table 1.

| Columna CSV | Original F1 / tipo recomendado | Definición y categorías documentadas (resumen fiel) |
|---|---|---|
| `hotel` | H1/H2 en abstract / cualitativa nominal | Hotel de la reserva: H1 resort de Algarve; H2 hotel urbano de Lisboa. F2 describe city hotel y resort hotel; verificar literales del CSV con T2. |
| `is_canceled` | IsCanceled / cualitativa nominal binaria | Indica si la reserva fue cancelada: 1 sí, 0 no. |
| `lead_time` | LeadTime / cuantitativa discreta | Días entre entrada de la reserva en el PMS y fecha de llegada: llegada menos entrada. No es antigüedad desde extracción. |
| `arrival_date_year` | ArrivalDateYear / temporal discreta | Año de la fecha de llegada. Los años efectivamente presentes los mide T2. |
| `arrival_date_month` | ArrivalDateMonth / cualitativa cíclica | Mes de llegada; 12 categorías de January a December. No imponer distancia lineal entre meses. |
| `arrival_date_week_number` | ArrivalDateWeekNumber / temporal discreta | Número de semana de la fecha de llegada. F1 no especifica en Table 1 una convención ISO ni un rango permitido. |
| `arrival_date_day_of_month` | ArrivalDateDayOfMonth / temporal discreta | Día del mes de la fecha de llegada; validar coherencia con mes/año. |
| `stays_in_weekend_nights` | StaysInWeekendNights / cuantitativa discreta | Noches de sábado o domingo que el huésped permaneció o reservó para permanecer; contadas del total de noches. |
| `stays_in_week_nights` | StaysInWeekNights / cuantitativa discreta | Noches de lunes a viernes que permaneció o reservó para permanecer; contadas del total. |
| `adults` | Adults / cuantitativa discreta | Número de adultos. |
| `children` | Children / cuantitativa discreta | Número de niños: suma de niños de pago y sin cargo. Que el CSV pueda almacenarla como float por nulos no cambia su naturaleza de conteo. |
| `babies` | Babies / cuantitativa discreta | Número de bebés. |
| `meal` | Meal / cualitativa nominal | Régimen reservado: Undefined/SC = sin comidas; BB = alojamiento y desayuno; HB = media pensión, desayuno y otra comida normalmente cena; FB = pensión completa, desayuno/almuerzo/cena. Undefined y SC no son valores faltantes por su mera etiqueta. |
| `country` | Country / cualitativa nominal | País de origen; F1 lo presenta mediante códigos. Table 1 imprime «ISO 3155–3:2013», mientras referencia 6 dice ISO 3166-3:2013: discrepancia editorial, no una norma corregida/verificada por este trabajo. |
| `market_segment` | MarketSegment / cualitativa nominal | Designación del segmento de mercado. TA significa Travel Agents; TO, Tour Operators. Table 1 no enumera todos sus niveles. |
| `distribution_channel` | DistributionChannel / cualitativa nominal | Canal de distribución de la reserva. TA = Travel Agents; TO = Tour Operators; sin lista exhaustiva en Table 1. |
| `is_repeated_guest` | IsRepeatedGuest / cualitativa nominal binaria | 1 huésped repetido, 0 no. Se considera repetido cuando existe perfil del cliente creado antes del alta de la reserva en PMS. |
| `previous_cancellations` | PreviousCancellations / cuantitativa discreta | Reservas anteriores del mismo perfil, creadas antes de la actual y canceladas. Sin perfil asociado se asigna 0. |
| `previous_bookings_not_canceled` | PreviousBookingsNotCanceled / cuantitativa discreta | Reservas del mismo perfil creadas antes de la actual y no canceladas. Sin perfil asociado se asigna 0. |
| `reserved_room_type` | ReservedRoomType / cualitativa nominal | Código del tipo de habitación reservada; anonimizado en lugar de denominación. Sin orden ni lista exhaustiva documentados. |
| `assigned_room_type` | AssignedRoomType / cualitativa nominal | Código del tipo de habitación asignada, anonimizado. Puede diferir del reservado por razones operativas, por ejemplo sobreventa, o petición del cliente. |
| `booking_changes` | BookingChanges / cuantitativa discreta | Cambios desde entrada en PMS hasta check-in o cancelación; suma de iteraciones únicas que modifican personas, llegada, noches, tipo reservado o comida. |
| `deposit_type` | DepositType / cualitativa nominal | Depósito para garantizar reserva, derivado de pagos TR previos a llegada o cancelación. No Deposit = sin pagos; Non Refund = pago igual o superior al costo total; Refundable = pago inferior al costo total. Son nombres del dataset: no deducir una política contractual de reembolso no documentada. |
| `agent` | Agent / cualitativa nominal | ID anonimizado de la agencia que hizo la reserva. Párrafo posterior a Table 1: NULL = no aplicable, reserva sin agencia. |
| `company` | Company / cualitativa nominal | ID anonimizado de empresa/entidad que reservó o es responsable de pagar. NULL puede ser categoría no aplicable, según párrafo posterior a Table 1. |
| `days_in_waiting_list` | DaysInWaitingList / cuantitativa discreta | Días en lista de espera antes de confirmación al cliente. La frase Source/Engineering describe una resta en orden aparentemente invertido; mantener definición semántica y no reconstruir fechas con esa frase. |
| `customer_type` | CustomerType / cualitativa nominal | Contract = reserva con allotment u otro contrato; Group = asociada a grupo; Transient = ni grupo ni contrato ni asociada a otra reserva transient; Transient-party = transient asociada al menos a otra transient. |
| `adr` | ADR / cuantitativa continua | Average Daily Rate: suma de todas las transacciones de alojamiento dividida por número total de noches de estancia; fuentes BO/BL/TR. Table 1 no indica explícitamente moneda ni afirma que sea cotización disponible al crear reserva. |
| `required_car_parking_spaces` | RequiredCardParkingSpaces [sic] / cuantitativa discreta | Número de plazas de estacionamiento requeridas por el cliente, **no capacidad disponible** del hotel. La d extra de Card es errata en nombre original. |
| `total_of_special_requests` | TotalOfSpecialRequests / cuantitativa discreta | Número total de peticiones especiales del cliente; ejemplos: camas separadas o piso alto. |
| `reservation_status` | ReservationStatus / cualitativa nominal | Último estado: Canceled = cancelada por cliente; Check-Out = cliente ingresó y ya salió; No-Show = no hizo check-in. F1 añade literalmente «and did inform the hotel of the reason why» para No-Show; no corregir silenciosamente esa frase ni inferir conducta de notificación desde el CSV. |
| `reservation_status_date` | ReservationStatusDate / fecha temporal | Fecha de establecimiento del último estado. Junto al estado permite conocer cuándo se canceló o cuándo salió el cliente; no equivale a fecha de creación. |

### Fragmentos textuales breves para citas de T3

[F1] https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ — HTML completo, HTTP 200; espacios HTML normalizados, palabras y puntuación preservadas:

- Specifications table, Experimental factors: “The data point time for each observation was defined as the day prior to each booking׳s arrival”.
- Sección 1, Data: “when available, the variables’ values were extracted from the bookings change log, with a timestamp relative to the day prior to arrival date (for all the bookings created before their arrival date).”
- Copyright and License information: “This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).”
- Después de Table 7: “It is also common for hotels not to know the correct nationality of the customer until the moment of check-in.”
- Table 1, ADR, Source/Engineering: “Calculated by dividing the sum of all lodging transactions by the total number of staying nights”.
- Después de Table 1, fragmento literal con sus comillas internas: `This should not be considered a missing value, but rather as “not applicable”.`

[F2] https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand — HTTP 200; fragmento del objeto JSON-LD de la página, exacto en contenido (espacios de serialización no significativos):

```json
{"@type": "CreativeWork", "name": "Attribution 4.0 International (CC BY 4.0)", "url": "https://creativecommons.org/licenses/by/4.0/"}
```

Corroboración independiente: https://www.kaggle.com/api/v1/datasets/view/jessemostipak/hotel-booking-demand — HTTP 200; `licenseName` = `Attribution 4.0 International (CC BY 4.0)`. No sustituir la evidencia de licencia de Kaggle por la licencia del artículo aunque coincidan.

### Método de extracción reproducible sin depender de `<article>`

El método exitoso usa `urllib.request.urlopen` + `html.parser.HTMLParser` sobre **todo el HTML**. No presupone que el contenido esté dentro de una etiqueta `<article>` ni requiere BeautifulSoup/lxml. Ignora script/style, decodifica entidades, concatena texto y localiza marcadores visibles con `index` (falla explícitamente si faltan, en vez de devolver un recorte vacío). Ejecutable con `python3`, sin instalar paquetes ni escribir archivos:

```python
from html.parser import HTMLParser
from urllib.request import urlopen
import re

class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if tag in ("p", "tr", "h1", "h2", "h3", "li"):
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/"
with urlopen(url, timeout=40) as response:
    print("HTTP", response.status)
    parser = Text()
    parser.feed(response.read().decode("utf-8"))
text = re.sub(r"[ \t\r\f\v]+", " ", " ".join(parser.parts))
start = text.index("Variables description.")
end = text.index("Summary statistics for both hotels", start)
print(text[start:end])
for needle in ("The data point time", "when available,",
               "This is an open access article", "It is also common"):
    start = text.index(needle)
    print(text[start:start + 700])
```

Si PMC entrega un challenge en lugar del artículo, los marcadores ausentes deben considerarse fallo de recuperación; no declarar tabla vacía ni inventar definiciones. Este método fue observado funcionando en las dos recuperaciones de T1, pero no garantiza superar bloqueos futuros del servidor.

## Precisión metodológica final T1 — fuentes realmente consultadas para S4/S9

### Correspondencia con el código inspeccionado

Lectura de `preprocesamiento/src/eda.py` en esta continuación: `explore` calcula Spearman mediante **Polars `pl.corr(..., method="spearman")`**, no invoca `scipy.stats.spearmanr` ni calcula sus p-valores. `chi_square` sí invoca **`scipy.stats.chi2_contingency(observed, correction=False)`** y calcula manualmente **V = sqrt(χ² / [n min(r−1,c−1)])**. `scipy.stats.contingency.association` sirve de documentación y contraste de esa fórmula, **no es una llamada del pipeline**. No describir funciones consultadas como funciones ejecutadas.

El indicador `assumptions_met` es exactamente `expected.min() >= 1 and (expected < 5).mean() <= 0.2`. Solo comprueba dos umbrales de frecuencias esperadas, **no todos los supuestos**. El programa igualmente devuelve χ² y p cuando el indicador es falso; no ejecuta automáticamente Fisher, permutaciones o Monte Carlo. Repetidos/reservas grupales pueden romper independencia; no se corrige multiplicidad. Presentar asociaciones exploratorias, no inferencia causal ni garantía de validez por superar umbrales.

### [M1] SciPy: documentación oficial actual, tres entradas leídas

Consultadas por urllib/HTMLParser, HTTP 200, el 2026-11-10 UTC. El enlace `[source]` de la documentación actual de `association` apunta a **v1.18.0**; no atribuir estas páginas móviles sin más a la versión instalada 1.18.1.

1. **chi2_contingency** — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
   - Descripción: “Chi-square test of independence of variables in a contingency table.”
   - Explica frecuencias esperadas a partir de marginales bajo independencia; estadístico predeterminado de Pearson. `correction=True` aplica Yates si grados de libertad = 1; el código usa False, por lo que no lo aplica.
   - Notes, cita literal: “An often quoted guideline for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.”
   - **Esta página NO respalda literalmente la regla mínimo 1 / hasta 20% <5**: enuncia otra guía, más estricta, que incluye frecuencias observadas y esperadas. Advierte inexactitud de aproximación asintótica con tablas de suma pequeña y propone remuestreo; aquí no se implementa ese recurso.
2. **association** — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html
   - “Calculates degree of association between two nominal variables.” Permite `method="cramer"` (predeterminado); describe V entre 0 (sin asociación) y 1 (asociación perfecta).
   - Código oficial enlazado, también recuperado HTTP 200: https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/contingency.py#L425-L526 ; contenido leído desde https://raw.githubusercontent.com/scipy/scipy/v1.18.0/scipy/stats/contingency.py . Define `phi2 = chi2_stat.statistic / arr.sum()`, divide por `min(n_cols - 1, n_rows - 1)` y retorna raíz cuadrada. Confirma la fórmula manual usada en EDA, sin afirmar haber ejecutado una comparación numérica adicional.
3. **spearmanr** — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
   - Cita literal: “The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between two datasets.”
   - Documenta rango −1 a +1, signo de asociación monótona y resultado indefinido con entrada constante. Su advertencia de aproximación de p-valores para muestras pequeñas corresponde a `spearmanr`; **el mapa EDA es descriptivo y no utiliza esos p-valores**.
   - Respalda describir relación monótona basada en rangos, no causalidad ni inmunidad general a outliers/dependencia. La elección de Polars es un detalle de implementación distinto de la definición metodológica consultada.

### [M2] Manual oficial R stats: evidencia de la heurística de esperadas

https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html — título **“Fisher's Exact Test for Count Data”**, entrada `fisher.test`, paquete stats versión 4.6.0 observada al pie. Página completa leída, HTTP 200, 2026-11-10 UTC. No se ejecutó R ni Fisher: se consulta únicamente para documentar el criterio de aproximación.

- Usage: `hybridPars = c(expect = 5, percent = 80, Emin = 1)`.
- Details, pasaje literal relevante: “if no cell has expected counts less than 1 ( = Emin ) and more than 80% ( = percent ) of the cells have expected counts at least 5 ( = expect ), otherwise the exact calculation is used.” Espacios matemáticos normalizados desde HTML.
- Contexto específico: decisión híbrida para tablas mayores que 2×2, también evaluada sobre subtablas; **no una certificación universal para cualquier tabla de Pearson**. El manual las llama “Cochran conditions” (o versión modificada).
- **Frontera importante:** el texto dice **más del 80%**, mientras el código admite **al menos 80%** (equivalente a **≤20%** por debajo de 5). Esta fuente respalda los umbrales y la familia de heurística, pero **no respalda literalmente la igualdad del 20%**. Esa frontera inclusiva es una convención local que debe declararse, no atribuirse como cita exacta del manual o de Cochran. Ninguna fuente consultada en esta tarea establece literalmente esa versión inclusiva para todos los tamaños de tabla.
- Redacción recomendada: «Se informan el mínimo y porcentaje de frecuencias esperadas inferiores a 5. Como cribado heurístico local se exige mínimo ≥1 y ≤20% de celdas <5; inspirado en las condiciones documentadas en R, con frontera inclusiva propia. SciPy enuncia además la guía más estricta de todas las frecuencias observadas y esperadas ≥5. Superar el cribado no prueba independencia de observaciones ni causalidad».

No fue necesario incorporar un tercer artículo: **no se leyó ni se verificó aquí el artículo de Cochran de 1954, ni su DOI**. No generar una entrada Cochran ni presentarlo como fuente primaria leída. Dos referencias documentales verificadas son suficientes; M1 agrupa tres páginas del mismo manual y M2 delimita la heurística y su discrepancia de frontera.

### BibTeX propuesto para T3 (dos referencias, sin autores/años/DOI inventados)

No se asigna año de publicación a páginas actualizadas sin fecha editorial; `urldate` es fecha de consulta, no publicación. Metadatos de títulos tomados de los encabezados visibles. La referencia SciPy agrupa explícitamente las tres entradas leídas; las URL específicas anteriores permiten comprobar cada afirmación.

```bibtex
@misc{scipy_stats_methods,
  title = {{SciPy} API reference: chi2\_contingency, association, and spearmanr},
  organization = {The SciPy community},
  howpublished = {Official online API documentation},
  url = {https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html},
  urldate = {2026-11-10},
  note = {Also consulted: scipy.stats.contingency.association and scipy.stats.spearmanr in the same official API reference}
}

@misc{r_fisher_conditions,
  title = {Fisher's Exact Test for Count Data},
  howpublished = {R Documentation, stats package, fisher.test},
  url = {https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html},
  urldate = {2026-11-10},
  note = {Development manual; stats version 4.6.0 observed. Used for documented hybrid approximation conditions, not as evidence of running Fisher's test}
}
```

## Inspección de archivos preexistentes y próximos pasos

Inspeccionados antes de escribir salidas:
- `README.md`: solo título. `.gitignore`: solo `.atl/`.
- `informe/main.typ`: import correcto 0.1.4, título/términos de ejemplo, abstract vacío; dos autores ya existentes, no inventados por este trabajo. Preservar identidades; actualizar metadatos académicos en T3.
- `informe/{introduccion,problema,objetivos,descripcion_dataset,analisis_exploratorio,conclusiones}.typ`: vacíos. `informe/refs.bib`: dos referencias Net Wok de plantilla, no evidencia científica; sustituir en T3 por fuentes verificadas.
- `preprocesamiento/src/main.py`: únicamente Hello World; `src/cleaning.py`: vacío. No función existente de limpieza/descarga que reutilizar.
- `preprocesamiento/pyproject.toml`, `.python-version`, `uv.lock` completo: único proyecto Python ≥3.14, fija 3.14; dependencias base presentes, faltan kagglehub y herramientas de desarrollo. PyPI consultado confirma versiones actuales numpy 2.5.4, polars 2.1.0, scipy 1.18.1, sklearn 1.9.1 y matplotlib 3.11.2; no se ejecutó sync ni se demostró instalación del proyecto.
- `informe/imagenes/.gitkeep`, `preprocesamiento/data/{raw,processed}/.gitkeep`: estructura vacía descubierta; sin dataset ni figuras. No CI ni pruebas en estructura inspeccionada.
- Estado inicial: PDF y `odd/` no versionados; preservados. `uv` y `typst` en PATH; no `pdftotext`; `python` no existe en PATH, sí `python3`. Extracción con PyMuPDF mediante uv aislado, sin modificar pyproject/lock.

**T2:** descargar real, validar 119390×32 como expectativa, separar diagnóstico del original de limpieza y fit train-only; conservar no-aplicable de IDs; analizar mecanismo de falta sin declarar MCAR/MAR/MNAR demostrado; justificar outliers y auditar repetidos entre splits. **T3:** diccionario de las 32 columnas desde CSV + tabla 1, cifras locales verificadas y bibliografía F1/F2; distinguir hallazgos descriptivos de fases posteriores. **T4/T5:** reproducibilidad y compilación reales quedan pendientes, fuera de las dos salidas autorizadas de T1.
