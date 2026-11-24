# Checklist de entrega — Hotel Booking Demand, fases 1 y 2

## Fuente y criterio de lectura

Fuente literal: `guia_detallada_proyecto_ciencia_datos.pdf`, secciones **3.1 y 3.2**, páginas físicas 3–4. Los 16 textos siguientes proceden de `preprocesamiento/outputs/specification.md`, sección «Checklist exacto de 3.1 y 3.2»; solo se normalizaron acentos tipográficos y cortes de línea del PDF. La extracción original completa está en `preprocesamiento/outputs/guide.txt`. Todas las rutas de evidencia se expresan desde la raíz del repositorio.

`[x]` indica evidencia disponible en fuentes/código/resultados y verificaciones comunicadas; `[ ]` mantiene una limitación o control pendiente. PDF final compilado e inspeccionado: 12 páginas, 32 tablas, siete figuras y cuatro referencias, sin blancos ni recortes; solapamiento corregido. Las revisiones independientes técnica y académica, incluidos los rechecks finales, terminaron sin bloqueadores. Permanecen explícitas las limitaciones de procedencia y la ausencia de ejecución de CI remota; no hay controles técnicos/académicos pendientes del entregable.

## 3.1 — Fase 1 (8 requisitos)

1. [x] **Contextualización del dominio del problema e impacto potencial de la solución analítica.**
   - Evidencia: `informe/introduccion.typ` explica gestión de capacidad, reservas y límites causales; `informe/problema.typ` delimita ADR y cancelación. La unidad es reserva, no huésped.

2. [x] **Formulación de 1 pregunta de investigación principal y 3 preguntas secundarias.**
   - Evidencia: `informe/problema.typ`, bloques «Pregunta principal» y «Preguntas secundarias»: una principal y tres entradas secundarias.

3. [x] **Objetivos del proyecto (1 General y 3 Específicos) en formato SMART.**
   - Evidencia: `informe/objetivos.typ`, un objetivo general y tres específicos con productos verificables y fechas de fases. Los plazos se presentan como metas académicas, no fechas de ejecución atribuidas.

4. [ ] **Nombre del dataset, repositorio de origen, URL exacta de acceso, fecha de extracción y tipo de licencia libre.**
   - Evidencia disponible: `informe/descripcion_dataset.typ`, «Ficha técnica y procedencia»; `preprocesamiento/outputs/source.json`: dataset, URL, versión Kaggle 1, licencia CC BY 4.0, atribución, SHA-256, `acquisition` y `verified_at_utc`. Función: `src.data.prepare`.
   - Limitación explícita: adquisición/inspección local `2026-11-10T02:36:59.972054+00:00`, origen `local_cached`; `downloaded_at_utc: null`. La descarga original no tiene fecha acreditada y el mtime no la prueba. La adquisición y la verificación no sustituyen silenciosamente la fecha de extracción solicitada. No se marca cumplimiento literal completo de ese dato.

5. [x] **Diccionario de datos exhaustivo: Nombre de la columna, tipo de variable (cuantitativa continua/discreta, cualitativa nominal/ordinal), descripción funcional y rango/valores permitidos.**
   - Evidencia: `informe/descripcion_dataset.typ`, cuatro tablas de ocho variables con nombre, tipo y definición/dominio semántico; `preprocesamiento/outputs/eda.json/dictionary` contiene los dominios observados de las 32 columnas. `informe/presentacion.typ` los presenta desde `informe/datos.typ`; `informe/generate_report_data.py` los serializa sin recalcular estadísticas. Fechas se explican como temporales; identificadores no se tratan como magnitudes. Catálogos nominales extensos permanecen en JSON.

6. [x] **Análisis Univariado: Medidas de tendencia central, dispersión, asimetría y curtosis para variables cuantitativas. Tablas de frecuencias para variables cualitativas.**
   - Evidencia: `src.eda.quantitative_summary` y `src.eda.explore`; `eda.json/quantitative` (16 resúmenes) y `/categorical`; `informe/analisis_exploratorio.typ`; figuras `distributions.svg` y `adr_by_hotel.svg`.
   - Pruebas: `test_summary_exact_statistics`, `test_summary_handles_entirely_missing_column` en `preprocesamiento/tests/test_pipeline.py`. Convenciones documentadas en `eda.json/statistics_conventions`: varianza/SD muestral, cuartiles lineales, asimetría corregida y curtosis de exceso.

7. [x] **Análisis Bivariado y Multivariado: Matriz de correlación (Pearson o Spearman), diagramas de dispersión (scatter plots), y pruebas de independencia (χ2).**
   - Evidencia: `eda.json/correlations` y `/chi_square`; funciones `src.eda.chi_square`, `explore` y `plot_figures`; `informe/imagenes/correlations.svg`, `relationships.svg`, `cancellation_categories.svg`; discusión en `informe/analisis_exploratorio.typ`.
   - Prueba: `test_chi_square_expected_and_effect`. Se informan frecuencias esperadas y V de Cramér; `market_segment` no cumple supuestos registrados. Repetidos pueden vulnerar independencia; no se afirma causalidad ni corrección por multiplicidad.

8. [x] **Diagnóstico de Calidad de Datos: Porcentaje exacto de datos faltantes por variable, detección de valores atípicos (outliers) mediante regla IQR o Z-score, e identificación de incoherencias o duplicados.**
   - Evidencia: `eda.json/quality/missing` para las 32 variables, `/quality/anomalies`, `/quality/duplicates_extra`, `/quality/duplicate_member_rows` y `/quantitative/*/iqr_outliers`; figura `missingness.svg` y sección de calidad del informe.
   - Comprobación independiente del original comunicada: 119390×32, nulos, 31994 repetidos adicionales, ADR negativo 1/cero 1959/IQR superior 211.065 y 3793 excesos, cero noches 715 y cero huéspedes conocidos 180. No se eliminan repetidos automáticamente.

## 3.2 — Fase 2 (8 requisitos)

9. [x] **Justificación técnica del tratamiento de valores faltantes (eliminación vs. imputación mediante media, mediana, KNNImputer o IterativeImputer). Prohibido imputar sin analizar el mecanismo de falta (MCAR, MAR, MNAR).**
   - Evidencia: `eda.json/missingness_by_group` y `/missingness_interpretation`; `src.modeling.make_preprocessor`: mediana numérica ajustada en train. `children` desconocido se conserva hasta el pipeline. Identificadores nominales `agent/company` excluidos; ausencia estructural/no aplicable se distingue de ausencia desconocida.
   - Prueba: `test_train_only_imputation_scaling_and_unknown_categories`. Las diferencias observadas sugieren hipótesis condicionales, no demuestran MCAR/MAR/MNAR; MNAR no se descarta.

10. [x] **Tratamiento razonado de valores atípicos (outliers) mediante remoción, transformación logarítmica o recorte por percentiles (winsorization).**
    - Evidencia: `src.cleaning.clean_bookings`, `cleaning.json/policy`, `/rules` y `/rejected`. Se excluyen ADR negativo, conteos negativos, cero huéspedes conocidos, cero noches y ADR ausente/no finito; se conservan ADR cero y extremos positivos finitos porque IQR no prueba error. Logaritmo/winsorización son alternativas discutidas, no optimizadas con test.
    - Resultado: 826 filas rechazadas por `row_id`, 118564 retenidas. Reglas 1/0/180/715/0; se solapan. Prueba: `test_cleaning_trace_features_and_preserved_duplicates` conserva extremos y repetidos.

11. [x] **Creación de al menos 2 variables derivadas respaldadas en el conocimiento del dominio.**
    - Evidencia: `src.cleaning.clean_bookings`: `total_nights` suma noches de semana y fin de semana; `total_guests` suma adultos/niños/bebés y conserva ausencia. Ambas están en `data/processed/cleaned.parquet` y en los Parquet de cada tarea/split. `phase2.json/tasks/*/numeric_features` incluye `total_nights`, pero excluye `total_guests`: se conserva como diagnóstico trazable, no como predictor.
    - Pruebas: `test_cleaning_trace_features_and_preserved_duplicates`, `test_known_adults_survive_unknown_children` y `test_offline_task_exports_are_target_safe`. X conserva adultos/niños/bebés conocidos, imputa niños solo en train y no imputa independientemente un total desconocido. Evita perder adultos por un total nulo. Noches usa el total sin sus componentes; originales permanecen en cleaned.

12. [x] **Codificación de categóricas (One-Hot Encoding, Target Encoding u Ordinal Encoding) previniendo estrictamente la fuga de datos (Data Leakage).**
    - Evidencia: `src.modeling.make_preprocessor` usa `OneHotEncoder(handle_unknown="ignore")`; `run_task` ajusta cada `Pipeline` solo sobre train tras `split_bookings`. `phase2.json/excluded_features` justifica exclusiones de desenlace y disponibilidad temporal. Mes e identificadores no se codifican como magnitudes arbitrarias.
    - Pruebas: `test_train_only_imputation_scaling_and_unknown_categories`, `test_features_exclude_targets_and_uncertain_information`, `test_offline_task_exports_are_target_safe`.

13. [x] **Escalamiento/Estandarización (StandardScaler, MinMaxScaler o RobustScaler) ajustado únicamente sobre el conjunto de entrenamiento.**
    - Evidencia: `src.modeling.make_preprocessor` y `run_task`: mediana seguida de `StandardScaler`, fit train-only. `phase2.json/tasks/*/transforms` registra medianas, medias y nombres transformados por modelo.
    - Prueba: `test_train_only_imputation_scaling_and_unknown_categories` comprueba que el valor extremo del holdout no cambia la media de entrenamiento ni la imputación.

14. [x] **División de datos en Entrenamiento (70–80%), Validación y Prueba (20–30%). En clasificación, aplicar división estratificada (Stratified Split).**
    - Evidencia: `src.modeling.split_bookings`, semilla 42 y objetivo 75/12.5/12.5 **por peso de filas**, dentro de cada clase para clasificación, no por cantidad de grupos. Asignación de grupos intactos por peso descendente y mayor déficit relativo entre destinos con encaje; sin encaje minimiza incremento del error relativo cuadrático. Valida train 70–80 % y holdouts no vacíos antes de fit/exportación; rechaza incompatibilidad con `ValueError`. Heurística voraz, no solución exhaustiva de factibilidad.
    - Clasificación: train 88923 (`0.75`), validación 14821 (`0.12500421713167573`), test 14820 (`0.12499578286832429`), población 118564.
    - Regresión no cancelada: train 55791 (`0.75`), validación 9299 (`0.12500672151422274`), test 9298 (`0.12499327848577728`), población 74388.
    - Huellas originales cruzadas: cero en las tres parejas y ambas tareas. Perfiles X compartidos no identifican clientes ni prueban la misma reserva: clasificación 2934/2848/807, regresión 2073/2058/587 (train–validación/train–test/validación–test).
    - `phase2.json/tasks/*/splits/*/group_sizes` incluye histogramas y repetidos: clasificación 6104/1015/1015 grupos repetidos, fracciones de filas repetidas `0.3378540984897046`/`0.33796639902840564`/`0.3379217273954116`; regresión 3667/610/610, fracciones `0.22345898083920346`/`0.22346488869770942`/`0.22348892234889223` (train/validación/test).
    - Pruebas: `test_grouped_stratified_reproducible_splits`, `test_unequal_groups_train_fraction_meets_guide`, `test_weighted_600_rows_keep_groups_and_stratify`, `test_impossible_4400_rows_rejected_before_export` y `test_repeated_groups_are_represented_in_every_split`.

15. [x] **Tratamiento de desbalance severo de clases si aplica (técnicas SMOTE, ADASYN o ajuste de class weight).**
    - Evidencia: `src.modeling.run_task` y `phase2.json/tasks/classification/imbalance`. Prevalencia positiva train `0.3725920178131642`, minoría no severa frente al umbral `0.1`: no se remuestrea ni pondera esta ejecución. La logística activa `class_weight="balanced"` si la minoría queda bajo el umbral. No se presenta SMOTE como aplicado.

16. [x] **Implementación de un modelo de referencia sencillo (Regresión Lineal Simple o Regresión Logística básica / Dummy Classifier) para establecer el umbral mínimo de rendimiento.**
    - Evidencia: `src.modeling.run_task`; DummyRegressor/media y LinearRegression para ADR en no canceladas; DummyClassifier/prior y LogisticRegression básica para cancelación retrospectiva. `phase2.json/tasks/*/metrics` separa validación/test. Pruebas: `test_metrics_exact_values`, `test_offline_task_exports_are_target_safe`.
    - Los cuatro modelos, las dos tareas y las métricas completas son **exigencias adicionales del usuario/S6**, no una transcripción de obligaciones más amplias de la sección 3.2. README contiene todos los valores finales; no se reutilizan métricas avanzadas ni se selecciona con test.

## Controles adicionales de la entrega

Estos controles corresponden al usuario y a `odd/tasks/hotel-booking-fases-1-2.md` (S2–S12), no se atribuyen literalmente a las 16 viñetas académicas.

- [x] **Python/entorno:** Python 3.14.6 verificado; único `preprocesamiento/pyproject.toml` y `uv.lock`; `uv sync --frozen` pasó. Polars, NumPy, SciPy, sklearn, matplotlib y kagglehub; pytest/Ruff/ty de desarrollo. Sin segundo proyecto ni dependencias del adaptador documental.
- [x] **Dependencias exportadas:** `preprocesamiento/requirements.txt` con versiones/hashes; cabecera `uv export --frozen --no-dev --format requirements-txt --output-file requirements.txt`. No se confunde exportación de runtime con herramientas de desarrollo.
- [x] **Ejecución real y aceptación CLI:** `prepare`, `eda`, `phase2` y `all` se ejecutaron una vez cada uno con salida 0 sobre `data/raw` real existente, sin descarga, con `--raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures"` después del verbo. `--help` también pasó. Se comprobaron source en prepare, EDA y siete SVG en eda, siete Parquet y diagnósticos en phase2, y las cuatro salidas JSON/siete Parquet/siete SVG en all. `phase2.json` completo (incluidas métricas/splits) y cleaning idénticos a producción; EDA comparó 1844 flotantes con rtol/atol `1e-12`, diferencia absoluta máxima `2.842170943040401e-14`. Hashes antes/después del CSV, source/EDA/phase2 de producción y PDF idénticos. No se atribuye esta aceptación a destinos de salida predeterminados.
- [x] **Calidad local del código final:** el coordinador comunicó salida 0 de Ruff, formato (7 archivos), ty de proyecto/adaptador y pytest, **20 passed in 4.28s**, Python 3.14.6. Secuencia de comprobación:

  ```bash
  cd preprocesamiento && uv run ruff check . ../informe/generate_report_data.py && uv run ruff format --check . ../informe/generate_report_data.py && uv run ty check && uv run ty check ../informe/generate_report_data.py && uv run pytest
  ```

- [x] **Pruebas sin Kaggle:** `preprocesamiento/tests/test_pipeline.py`; fixtures pequeños, esquema/nulos/inválidos, errores explícitos de descarga, limpieza, reproducibilidad/huellas, transformaciones train-only, métricas y exportación segura.
- [x] **CI configurada:** `.github/workflows/ci.yml`, push/pull_request, Python 3.14, checkout v7.0.1 y setup-uv v10.3.3 con releases verificadas, directorio preprocesamiento, frozen y cuatro checks sin dataset ni secretos Kaggle. Inspección estructural realizada; **ejecución remota no realizada**.
- [x] **README práctico:** entorno, comandos prepare/eda/phase2/all, cuatro opciones de rutas, semilla fija 42 sin flag inventado, exportación, serialización/compilación, estructura, resultados exactos y límites.
- [x] **No datasets pesados en Git:** inspección `git ls-files preprocesamiento/data informe/imagenes` solo muestra `.gitkeep` en raw/processed/imagenes al verificar; CSV/Parquet excluidos por `.gitignore`. No se excluyen figuras SVG ni JSON útiles de outputs. No se realizaron commits.
- [x] **Puente documental:** `informe/generate_report_data.py` serializa exclusivamente eda/source, con SHA-256, a `informe/datos.typ`; verificación comunicada de 32 entradas de diccionario y 16 cuantitativas. No recalcula estadísticas ni incorpora phase2 al EDA. Nombre operativo en inglés verificado por renombre puro, bytes/SHA-256 intactos (`690929fbd5a0cc8c0775f3123d0e293ced577fde5a28ecfed6d426bd6bbf4e01`); Ruff, formato (7 archivos) y ty del módulo renombrado pasaron. No se ejecutó `--help` porque no hay parser y regeneraría datos.typ; no se alteraron transporte, PDF, JSON ni SVG. Los reportes históricos mantienen el nombre válido en su fecha.
- [x] **Límites explícitos:** retrospectivo del día previo a llegada, no predictor al crear reserva; exclusión country/deposit_type y demás fuga temporal; ADR condicionado a no canceladas, selección y colas; repetidos conservados; perfiles iguales no prueban identidad; MCAR/MAR/MNAR no demostrados; asociaciones no causales; fecha original de descarga y moneda de ADR desconocidas.
- [x] **PDF final:** `typst compile informe/main.typ informe/fase1.pdf` salida 0; 12 páginas, 32 tablas, siete figuras y cuatro referencias. SHA-256 `c5fe717cc475b91e09cc6385710d0fc0dc22729f606151d13b7a6dd4e30ead8f`. `informe/revision_pdf.json` registra serialización coincidente, revisión visual completa previa y recheck de páginas 2/8/11/12 cambiadas por bibliografía/procedencia; cero blancos/recortes y solapamiento resuelto. No se infiere ausencia de solapes solo de bbox.
- [x] **Contraste académico numérico/visual:** `informe/revision_academica.md` documenta 230 filas numéricas PASS y lectura de los 12 renders, sin discrepancias de contenido. SciPy/R incorporadas y distinción del límite local de 20 % frente al criterio de R >80 % aclarada; total cuatro referencias.
- [x] **Build independiente reproducido:** compilación del coordinador a un directorio temporal con salida 0; comparación PyMuPDF: texto y todos los píxeles de las 12 páginas idénticos al PDF publicado. No se exige igualdad binaria de metadatos.
- [x] **Recheck independiente del pipeline final:** `preprocesamiento/outputs/review_pipeline.md`, «Cierre limitado H1/H2 y regresión directa de multiplicidad»: ambos hallazgos cerrados, S5/S7 met. Particiones, clases, multiplicidad, IDs/huellas sin solapamiento y ausencia segura PASS; phase2 aislado produjo JSON estructuralmente exacto al final y raw intacto. Refit independiente de los cuatro modelos: todas las métricas validation/test coinciden. No se atribuyen los resultados históricos al candidato actual.
- [x] **Recheck académico de los últimos cambios bibliográficos/procedencia:** `informe/revision_academica.md` confirma PDF `c5fe717c…ead8f`, 12 páginas, 32 tablas, siete figuras y cuatro referencias. Payload eda/source profundamente igual y hashes correctos; adquisición intacta, verificación actualizada sin atribuir descarga. Inspección visual 2/8/10/11/12 PASS, sin bloqueadores.

## Límites y pendientes que no deben ocultarse

La fecha de descarga original sigue desconocida, no es una casilla que pueda cerrarse inventando una fecha. CI remota no ejecutada es una limitación de evidencia, no un check fallido. La maquetación visual, el build reproducido, la calidad local, los resultados numéricos y ambos dictámenes independientes finales tienen evidencia. No quedan pendientes técnicos/académicos del entregable. La heurística de partición no es exhaustiva; valida restricciones y rechaza sin exportar si no puede satisfacerlas. Fases 3/4, modelos avanzados, clustering, MLP/SHAP y dashboard están fuera del alcance implementado.
