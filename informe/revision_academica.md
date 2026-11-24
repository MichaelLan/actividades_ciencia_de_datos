# Revisión académica independiente — T5 sobre T3

## Alcance y versión

Revisión genérica independiente; no constituye aprobación nativa ni certificación de todo S11. Se leyeron la guía (incluida 3.1), S1/S9/S10/S11, `outputs/specification.md`, `eda.json`, `source.json`, todos los `informe/*.typ`, `refs.bib`, `generar_datos.py` y el código EDA. No se ejecutaron el pipeline, el adaptador ni la compilación; no se modificaron PDF, datos o fuentes del informe.

**Resultado de contenido:** sin discrepancias numéricas ni incumplimientos académicos nuevos encontrados. **Cierre visual del PDF final: met** tras extracción y lectura de los 12 renders propios. No se usaron solamente controles de cajas para descartar solapamientos.

Huellas de evidencia:
- CSV: `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`.
- EDA: `a97ff27e670c7b21c24af0f571e9ec8b1f7a3f69104a4f27f0bced68ce7287af`.
- Source: `1eedc3f00244e52436b4c132a96b0f53cd0fb5223da48bf92469be2d7593aea2`.
- PDF inicial inspeccionado: `c685b0ce47f8b0f0d567dfacb1c1f32cf8f092dd0da772262abcb860c59742dc`, 11 páginas.
- PDF intermedio posterior: `8671947a606f94e984cd09db953aaa5eacda643822c78cd334406eb0b624afbc`, 12 páginas. No certificado: presentaba solapamiento de figura 7 con tabla XXVI.
- **PDF final comprobado:** `eed83a5a3e1642b3cf7ec83026e17c835ffaf49fd6c33d446fed124cdaf49c6b`, 12 páginas, 32 tablas y siete figuras. Figura 7 aislada en página 10; tabla XXVI íntegra en página 11.

## Matriz de cumplimiento

| Requisito / probe independiente | Estado | Evidencia |
|---|---|---|
| 3.1.1 contextualización e impacto acotado | met | Introducción: planificación hotelera; reserva como unidad; dos establecimientos, no población hotelera general. |
| 3.1.1 exactamente 1 pregunta principal y 3 secundarias | met | Problema: calidad, asociaciones y referencias predictivas; tercera pregunta reservada expresamente a fase 2. |
| 3.1.1 exactamente 1 objetivo general y 3 SMART | met (criterio de alcance aclarado) | Objetivos con productos verificables, 32 variables, ≥2 derivados y métricas especificadas; fechas 01/11, 15/11 y consolidación 28/11/2026. Se leyó la continuación del objetivo 3 en página 2. Allí se mencionan modelos avanzados, segmentación y despliegue como no realizados en esta entrega; la guía los sitúa después. Se acepta esa mención futura conforme a aclaración del coordinador, sin exigir implementar fase 3. No existe un objetivo SMART específico medible de segmentación: los tres detallan fases 1/2 y consolidación. |
| 3.1.2 ficha y procedencia; S9 | met | URL Kaggle exacta y fuente primaria en bibliografía; autores, 119390×32, hoteles 79330/40060, reserva, llegadas 2015-07-01–2017-08-31. Adquisición local y verificación separadas de descarga desconocida; ambas licencias CC BY 4.0 verificadas independientemente según evidencia T1, no CC0. |
| 3.1.2 diccionario exhaustivo; S9 | met | Tablas I–IV: exactamente 32 nombres reales. IDs nominales, niños como conteo aunque Float64, mes cíclico y fecha temporal, no magnitud. Definiciones contrastadas con evidencia científica; rangos observados y categorías coinciden con CSV/JSON, sin inventar reglas de exclusión ni moneda. Catálogos extensos remitidos explícitamente al JSON. |
| 3.1.3 univariado cuantitativo completo | met | Tablas V–VII: 16 variables; válidos, media, mediana, todas las modas, DE/varianza muestrales, extremos, cuartiles, IQR, asimetría corregida y exceso de Fisher. Calendario descrito como componente temporal, excluido de correlaciones. |
| 3.1.3 frecuencias cualitativas | met | Tablas VIII–XXII: todas las categorías de baja cardinalidad; cinco principales de país/agente/empresa declaradas como selección. Denominador original, nulos incluidos; frecuencias íntegras en JSON. Un 0 % redondeado con conteo positivo no significa inexistencia. |
| 3.1.3 correlación, dispersión, ADR por hotel/características | met | Spearman justificado por colas/monotonía, Pearson como contraste, sin IDs/calendario; figuras 4–6 y tablas XXV–XXX. Dispersión n=4000, semilla 42; no se extrapola su tasa a población. |
| 3.1.3 χ², supuestos y efecto | met | Tablas XXXI–XXXII: siete contrastes; χ²/gl/V/esperadas comprobados. Market segment no cumple mínimo de 1; canal alcanza 20 % de celdas <5. Advierte dependencia, multiplicidad, p subdesbordado y ausencia de causalidad. |
| 3.1.3 calidad de datos | met | Tabla XXIII: 32 nulos y porcentajes; XXIV: anomalías. Distingue 31994 repetidos extra de 40165 miembros, sin eliminación. Regla IQR, cero noches/huéspedes, ADR negativo/cero/extremo y temporalidad interpretados, no errores automáticos. |
| S9 conclusiones y separación de fases | met | Responde calidad y asociaciones; propone fase 2 sin inventar respuesta predictiva ni importar métricas heredadas. Reconoce selección en regresión no cancelada y clasificación retrospectiva. |
| S9 español, IEEE, estructura y referencias | met | Charged-ieee 0.1.4, dos columnas, resumen/términos, secciones solicitadas y dos referencias reales [1]–[2], DOI/volumen/páginas correctos; sin plantilla bibliográfica ni portada. |
| Adaptador sin nuevo cálculo ni acoplamiento estadístico | met | Solo stdlib: carga JSON, SHA-256 y doble serialización JSON dentro de `json(bytes(...))`. Igualdad profunda de payload y hashes comprobada; no ejecutado. Estadísticas en `src/eda.py`; destino gráfico parametrizado, no lectura del documento para calcular. |
| S10 PDF: texto, cantidades y geometría básica | met (versiones indicadas) | PyMuPDF: 32 tablas, figuras 1…7; todas las páginas contienen texto; ninguna palabra fuera de página. Se extrajo texto completo; no se hallaron ejemplos/plantillas. Esto NO prueba ausencia de colisiones internas. |
| S10 PDF final legible y sin solapamientos | met | Recheck propio del hash final: lectura de 12 PNG, siete figuras legibles a ancho completo; figura 7/p10 y tabla XXVI/p11 separadas, sin recortes ni solapamientos observados. |
| S10 compilación propia / README completo / CI; S11 integral | unverified | Fuera de esta revisión de contenido: no se ejecutaron compilación, checks Python ni CI. README leído, pero todavía contiene pendientes de confirmación final. No se certifican sus comandos desde este trabajo. |

No hay requisitos de contenido clasificados **unmet** con la evidencia examinada. La aceptación integral de S11 permanece fuera de esta certificación acotada. La conclusión visual se apoya en renders, no en ausencia de errores en las cajas.

## Probes reproducibles y resultados observados

Se ejecutaron scripts de lectura/cálculo sin exportar datos al repositorio:

1. **Serialización:** decodificar las dos capas JSON del literal de `datos.typ`, comparar profundamente `eda` y `source` con los JSON originales y comprobar SHA-256 de bytes. **PASS**; no basta con comparar las huellas impresas.
2. **Origen independiente:** `csv.DictReader` sobre el CSV original, `Counter` sobre las 32 columnas y comprobación de su SHA. **PASS**: esquema, 32 dominios observados, nulos y conteos de todos los niveles categóricos; 31994 extras y 40165 miembros.
3. **Estadística independiente stdlib:** `statistics.mean/median/variance`, cuartiles interpolados en posición `(n−1)p`, modas empatadas, IQR y outliers. Asimetría `sqrt(n(n−1))/(n−2)·m3/m2^(3/2)`; exceso corregido `(n−1)/((n−2)(n−3))·((n+1)(m4/m2²−3)+6)`. **PASS** para las 16 variables, tolerancia 1e−9. También PASS para todas las medias/medianas/tasas agrupadas de ADR y cancelación; los 715 registros de cero noches efectivamente tienen ADR cero.
4. **Denominadores:** todas las frecuencias suman 119390; relativas = conteo/119390 y nulos % = 100·conteo/119390. **PASS**. No nulos adicionales: niños 4 (0,003350 %), país 488 (0,408744 %), agente 16340 (13,686238 %), empresa 112593 (94,306893 %); otras 28 variables, cero.
5. **PDF frente a cifras, no solo hashes:** normalizar espacios, guiones Unicode y separadores invisibles del texto PyMuPDF; construir y localizar **230 filas numéricas completas** desde JSON, con redondeo decimal a 2/6/3 posiciones según tabla y tasas multiplicadas por 100. **PASS, cero filas ausentes** en PDF intermedio `8671947a…624afbc`: 48 filas cuantitativas, todas las frecuencias mostradas, 32 nulos, comparaciones agrupadas, intervalos y ambos diagnósticos χ². Ejemplos sensibles: ADR mediana 94,575 → 94,58; habitación C 162,995 → 163; IQR ADR 56,71; umbral 211,065. Unidades monetarias no inferidas.
6. **χ² independiente desde contingencias JSON:** esperadas = total fila·total columna/n; χ² = Σ(O−E)²/E; V = √(χ²/n); gl = (filas−1). **PASS** para siete pruebas y mínimos/porcentajes de esperadas; inferencia cautelosa coherente con los resultados. Los valores p del PDF concuerdan con JSON, sin recalcular su función de supervivencia.
7. **Prosa:** se contrastaron cifras incrustadas y rutas de interpolación: cancelación 37,041628 % (resumen 37,04 %), anticipación 104,01/69 días, ADR 101,83/94,58/5400 y curtosis 1013,19; Spearman anticipación–cancelación 0,317 (Pearson 0,293), ADR–cancelación 0,051, noches laborales–ADR 0,094, adultos–ADR 0,280. **Sin discrepancias**. Anomalías 1/1959/3793/715/180/74460/0/0 coinciden; fechas/objetivos contrastados con guía/manifiesto, no con fecha de descarga supuesta.

Los gráficos se contrastaron con su implementación y claves de resultados (distribución original, serie mensual, nulos, Spearman, muestra fija, cajas ADR, tasas categóricas), no se presentan como nueva recomputación de todo el pipeline. Reejecutar `all` no fue autorizado ni necesario para estos probes.

## Recheck visual final y límites

**Conocido y resuelto por T3, no hallazgo nuevo:** figura 7 de la versión intermedia invadía el espacio de tabla XXVI; una caja dentro de página no lo detectaba. El render propio de aquella página 10 confirmó la tabla cortada al pie.

En el hash final `eed83a5a…af49c6b` se extrajo nuevamente todo el texto y se renderizaron/leyeron **las 12 páginas**. Resultado: 32 tablas, siete figuras en orden 1…7, cero páginas blancas y cero palabras fuera de página; no se observaron solapamientos ni columnas desbordadas. La figura 7 ocupa página 10 sin tabla debajo; tabla XXVI incluye sus ocho niveles íntegros en página 11. Las otras seis figuras conservan ancho completo y etiquetas legibles; matriz de correlación permite leer los nombres largos. Página 9 tiene espacio inferior sin texto: no es una página blanca ni un recorte.

El multiconjunto de tokens del texto final coincide exactamente con el intermedio auditado, quitando únicamente folios de página y normalizando espacios/guiones: **cero tokens añadidos o retirados**. Se preservan las cifras de las 230 filas comprobadas; el cambio es paginación. No se recalculó el EDA ni se recompiló desde esta revisión.

T2 está corrigiendo incidencias de modelado/particiones fuera del alcance del PDF de fase 1. Una eventual actualización de `source.json/verified_at_utc` requiere regenerar transporte y revisar nueva huella/procedencia si se quiere sincronizar el informe con ese manifiesto posterior; esta revisión queda ligada a los hashes enumerados y no certifica futuras mutaciones.

Scratch externo conservado para coordinación (no parte de la entrega):
- `/tmp/gphxzMQG/ga47oXcO/hotel-academic-lyjy0b7j/`: texto y 11 PNG del primer PDF.
- `/tmp/gphxzMQG/ga47oXcO/hotel-academic-final-756b8zku/`: texto y 12 PNG del PDF intermedio, **no final**.
- `/tmp/gphxzMQG/ga47oXcO/hotel-academic-confirmed-yiv6cf3v/`: texto y 12 PNG del PDF **final confirmado**, inspeccionados visualmente.

El coordinador debe retirar estos scratch cuando deje de necesitarlos.

## Recheck final limitado — referencias y procedencia (versión vigente)

Este apartado actualiza el estado para **S9/S10**, sin repetir la auditoría estadística anterior ni certificar S11 integral. Sustituye las huellas de versión y el número de referencias anteriores; los resultados históricos siguen vinculados a sus respectivos hashes.

**Conclusión: met en el alcance delegado; ningún bloqueador concreto encontrado.**

- **PDF vigente:** `c5fe717cc475b91e09cc6385710d0fc0dc22729f606151d13b7a6dd4e30ead8f`, comprobado leyendo sus bytes: 12 páginas, 32 tablas, siete figuras en orden 1…7 y cuatro referencias [1]–[4].
- **EDA sin cambios:** `a97ff27e670c7b21c24af0f571e9ec8b1f7a3f69104a4f27f0bced68ce7287af`. Se reutiliza la evidencia previa de 230 filas y cálculos; no se recalcularon.
- **Source vigente:** `4fe14405d91f95a869e96cae01dee1ff34734a20cd7284bfad881057474e2c58`. Payload de `datos.typ` decodificado por completo: igualdad profunda con ambos JSON y SHA-256 internos correctos (**PASS**), sin ejecutar adaptador.
- **Procedencia, PASS:** adquisición original `2026-11-10T02:36:59.972054+00:00`, `local_cached`, descarga original nula y mtime separados, sin alteración. Únicamente cambia verificación a `2026-11-10T02:55:52.103330+00:00`. Prueba fuerte: sustituir solo ese timestamp en los bytes actuales recupera exactamente SHA-256 previo `1eedc3f0…593aea2`. La página 2 refleja correctamente la nueva verificación, sin convertirla en descarga.
- **Referencias nuevas, met:** [3] agrupa explícitamente las tres entradas de la API oficial SciPy (`chi2_contingency`, `association`, `spearmanr`); [4] corresponde al manual oficial R `fisher.test`. Títulos, URL y alcance coinciden con M1/M2 de `outputs/specification.md`. Fecha de consulta 10/11/2026, no año editorial inventado ni DOI/autores fabricados. No se añadió una referencia Cochran no verificada. Se contrastó con la evidencia documental M1/M2 suministrada, sin nueva consulta web.
- **Citas metodológicas, met:** página 8 cita [3] para Spearman basado en rangos y distingue cálculo Polars de documentación SciPy. Página 11 cita [3] para χ² ejecutado con `correction=False` y fórmula manual de V; declara que no llama `association`. Página 12 cita [4] para condiciones híbridas R en tablas mayores que 2×2 y [3] para la guía más estricta de todas las frecuencias observadas/esperadas ≥5. Quedan explícitos el criterio local mínimo ≥1 y ≤20 % de esperadas <5, su frontera inclusiva propia frente a **más del 80 %** de R y la ausencia de ejecución R/Fisher/remuestreo. No confunde superar ese cribado con satisfacer todos los supuestos.
- **Regresiones directas, met:** extracción textual completa sin páginas vacías ni palabras fuera de página. Lectura visual propia acotada a páginas **2/8/11/12**, más página **10** como control del arreglo anterior: texto/citas/tablas legibles, sin solapamientos ni recortes observados. Figura 7 permanece sola en p10; tabla XXVI íntegra, ocho niveles, en p11. No se incorporaron resultados numéricos de modelos de fase 2; sus métricas solo se mencionan como objetivos futuros, manteniendo la separación de entregas.

**Límite de comparación:** el scratch textual del PDF anterior ya no estaba disponible en esta sesión; no se afirma un nuevo diff automático completo de todo el texto. Se reutilizan la revisión previa y el hash EDA idéntico, y se comprobaron los pasajes modificados y sus regresiones directas, conforme al recheck limitado autorizado. No se ejecutaron `all`, compilación, adaptador, R ni pruebas estadísticas nuevas.

Scratch de este recheck: `/tmp/gphxzMQG/ga0gKc1v/hotel-refs-recheck-fv2i06_z/` contiene texto completo y PNG de páginas 2/8/10/11/12; el coordinador puede retirarlo tras revisión. Los anteriores scratch se enumeran históricamente, sin garantizar que sigan disponibles. La conclusión se limita al PDF y JSON con estas huellas; no certifica mutaciones posteriores.
