# Hotel Booking Demand: exploración y modelos base

Estudio académico reproducible de reservas hoteleras: calidad y relaciones del original, limpieza trazable y referencias sencillas para estimar la tarifa diaria media (ADR) y clasificar cancelaciones. Abarca **fases 1 y 2**; no incluye modelos avanzados, clustering, dashboard ni despliegue.

El pipeline completo se ejecutó sobre el dataset real con salida 0. Están disponibles EDA, limpieza, particiones finales dentro del intervalo exigido y métricas de cuatro baselines, documentados abajo desde los JSON generados. Calidad local verificada: Ruff, formato, ty y **20 pruebas aprobadas** con Python **3.14.6**. El informe final está compilado e inspeccionado; las revisiones independientes técnica y académica terminaron sin bloqueadores. CI remota no ejecutada.

## Datos y fuente científica

- Dataset: [Hotel Booking Demand en Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), `jessemostipak/hotel-booking-demand`, archivo `hotel_bookings.csv`; redistribuidor Jesse Mostipak.
- Fuente: N. António, A. de Almeida y L. Nunes, «Hotel booking demand datasets», *Data in Brief*, vol. 22, pp. 41–49, 2019. [DOI: 10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).
- Tanto Kaggle como el artículo declaran [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), verificadas independientemente; **no CC0**. Conservar atribución y procedencia. [Evidencia de fuentes y guía](preprocesamiento/outputs/specification.md).
- Validación local registrada: **119390 reservas y 32 variables**; City Hotel: **79330**, Resort Hotel: **40060**; llegadas entre **2015-07-01 y 2017-08-31**. Unidad de observación: reserva, no huésped ni noche.
- SHA-256 del CSV: `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`, registrado con esquema en [source.json](preprocesamiento/outputs/source.json). La adquisición/inspección local se registró en `2026-11-10T02:36:59.972054+00:00` (`acquisition.obtained_at_utc`, origen `local_cached`); la verificación inspeccionada es `2026-11-10T02:55:52.103330+00:00` y se actualiza al ejecutar la preparación. **La fecha de descarga original no está acreditada**: `acquisition.downloaded_at_utc` es `null`. El mtime `2026-11-10T02:31:17.667526+00:00` se conserva por separado y no prueba una descarga de red.

La descarga usa `kagglehub`, conserva el CSV original y se omite si ya existe en el directorio raw. Un fallo de descarga produce un error explícito, no datos sintéticos. El CSV no se distribuye en Git; no subir credenciales.

## Entorno y ejecución

Requisitos: **Python 3.14**, [uv](https://docs.astral.sh/uv/) y [Typst](https://typst.app/) para el informe. Un único proyecto Python, con Polars para datos y NumPy en el límite de scikit-learn. Desde la raíz:

```bash
cd preprocesamiento
uv sync --frozen
uv run python -m src.main --help
uv run python -m src.main all
```

La ayuda y los cuatro verbos `prepare`, `eda`, `phase2` y `all` se verificaron individualmente con salida 0 sobre el CSV real existente, con las cuatro opciones de rutas después del verbo y salidas aisladas en un directorio temporal. No hubo descarga ni cambios en los artefactos finales. Los checks de calidad tienen verificación independiente de la ejecución del pipeline.

### Etapas y opciones

Todos los comandos Python siguientes se ejecutan desde `preprocesamiento/`:

```bash
uv run python -m src.main prepare
uv run python -m src.main eda
uv run python -m src.main phase2
uv run python -m src.main all
```

| Comando | Efecto |
|---|---|
| `prepare` | Descarga si falta el CSV, valida dimensiones/esquema y escribe `source.json`. |
| `eda` | Prepara/valida y explora el original; escribe `eda.json` y figuras SVG. |
| `phase2` | Prepara/valida, limpia, crea features, divide y entrena cuatro baselines; escribe diagnósticos y Parquet. |
| `all` | Ejecuta las cuatro etapas anteriores sin necesidad de invocarlas por separado. |

| Opción | Ruta predeterminada, relativa a `preprocesamiento/` |
|---|---|
| `--raw-dir` | `data/raw` |
| `--processed-dir` | `data/processed` |
| `--output-dir` | `outputs` |
| `--figures-dir` | `../informe/imagenes` |

Los valores predeterminados se resuelven desde la ubicación del proyecto, no desde el directorio de trabajo. Las rutas relativas proporcionadas explícitamente se interpretan desde el directorio de ejecución. Para verificar cada etapa sin sobrescribir resultados o figuras de la entrega, esta secuencia se probó con salidas temporales:

```bash
scratch=$(mktemp -d /tmp/hotel-cli-acceptance-XXXXXX)
uv run python -m src.main prepare --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures"
uv run python -m src.main eda --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures"
uv run python -m src.main phase2 --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures"
uv run python -m src.main all --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures"
```

Se comprobaron los archivos esperados en cada etapa: cuatro JSON, siete Parquet y siete SVG al completar la secuencia. `phase2.json` y `cleaning.json` coincidieron exactamente con producción; EDA coincidió con tolerancia relativa/absoluta `1e-12` (1844 flotantes, diferencia absoluta máxima `2.842170943040401e-14`). Los hashes del CSV, source/EDA/phase2 de producción y PDF permanecieron idénticos. Esta aceptación no ejecutó las opciones con destinos predeterminados ni acredita una nueva descarga. La **semilla fija es 42**, definida en código para particiones y huellas; no existe opción `--seed`. La primera descarga requiere red; las pruebas unitarias no requieren Kaggle.

### Notebook autosuficiente de fase 2

Abrir [preprocesamiento/fase2.ipynb](preprocesamiento/fase2.ipynb) en un visor compatible con Jupyter para leer la narrativa y las salidas guardadas. **29 celdas (17 de código)** recorren la guía 3.2 en español, sin repetir el EDA completo de fase 1. Todas las definiciones necesarias están dentro: no importa `src`, no invoca la CLI ni requiere el informe o JSON anteriores.

Para reproducir desde un kernel nuevo, ejecutar desde `preprocesamiento/`:

```bash
uv sync --frozen --group notebook
uv run --group notebook python -c 'import sys, nbformat; from nbclient import NotebookClient; from jupyter_client import KernelManager; nb = nbformat.read("fase2.ipynb", as_version=4); km = KernelManager(kernel_name="python3"); km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]; NotebookClient(nb, km=km, timeout=600).execute(cleanup_kc=True); nbformat.write(nb, "fase2.ipynb")'
```

Comando probado con API real: el kernel se inicia con **el ejecutable del entorno uv**, no con un Python global elegido por el kernelspec. El cuaderno además imprime la versión y exige `sys.version_info[:2] == (3, 14)`. El grupo opcional `notebook` contiene `nbclient` (ejecución), `ipykernel` (kernel) y `nbformat` (lectura/escritura); no instala otro proyecto ni exige un servidor Jupyter. Para uso interactivo, seleccionar ese mismo entorno Python 3.14 en el editor y ejecutar todas las celdas tras reiniciar el kernel.

| Aspecto | Notebook | Script existente |
|---|---|---|
| Código | Completo en celdas didácticas | Módulos de `src` y CLI |
| Alcance | Fase 2, con diagnóstico de calidad necesario | Preparación, EDA y fase 2 |
| Resultados por defecto | `outputs/notebook_phase2/` | `outputs/` canónico |
| Parquet por defecto | `data/processed/notebook_phase2/` | `data/processed/` canónico |

La primera celda de configuración permite cambiar rutas y semilla (42 por defecto); la huella original mantiene semilla 42 para compatibilidad con el pipeline. Desde la raíz se detecta `preprocesamiento/`; en una carpeta independiente se usa el directorio actual, **sin exigir `pyproject.toml` ni `src`**. Llevar allí el notebook y `data/raw/hotel_bookings.csv`, con dependencias ya instaladas, basta. Solo si falta raw se invoca Kagglehub: no hay sustitución por CSV sintético ni descarga durante tests. El loader conserva `NA`/`NULL`/vacío, `agent/company` String y `children` Float64 como el script.

La ejecución guardada completó las 17 celdas en orden con **Python 3.14.6, sin excepciones**, tablas acotadas y una figura de confusiones. Una segunda ejecución en carpeta temporal, con solo notebook y CSV real (sin `src`, informe, pyproject ni outputs previos), terminó, tras la corrección F1, en **7.30 s** y reprodujo exactamente las tareas completas del JSON canónico: splits, métricas, predictores y diagnósticos. El contraste dentro del cuaderno es opcional: lee el canónico si existe; en aislamiento registra `available=false` y continúa.

Resultados: 826 descartes distintos, 118564 reservas limpias; clasificación **88923/14821/14820**, regresión no canceladas **55791/9299/9298**. Coincidencia exacta con `phase2.json` SHA-256 `99b5e3ffafbeb2540a4e6d245fc43ab33d31130cf146cbf2de1d94c40dbb0a6c`, incluida toda métrica de validación/test. No se copian métricas para producirlas. El CSV conserva su SHA-256 `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`; los artefactos canónicos anteriores se preservan.

Los JSON nuevos documentan procedencia, ausencia por grupos, limpieza por `row_id`, estadísticas ajustadas solo en train, desbalance, histogramas de grupos y manifiesto de versiones. `tests/test_notebook.py` lee JSON/AST con biblioteca estándar y prueba las definiciones mediante fixtures pequeños offline, sin necesitar nbclient, nbformat o IPython en CI. Incluye regresiones H1 (adultos conocidos con niños ausentes) y H2 (peso por filas, déficit relativo, repetidos en holdouts e incompatibilidad explícita). Ruff admite el `.ipynb`: **formatear antes de ejecutar**, para guardar salidas sobre el código final; `ty` se comprueba también sobre Python extraído temporalmente.

Calidad tras la primera corrección agrupada: **36 pruebas aprobadas en 2.62 s** (las 31 de base + 5 casos F1; 20 del pipeline anterior y 16 del notebook), incluso tras `uv sync --frozen` sin las bibliotecas opcionales de notebook instaladas. Ruff check/formato (8 archivos) y ty de proyecto/Python extraído pasan. [Evidencia de ejecución y 30 hashes canónicos antes/después](preprocesamiento/outputs/notebook_phase2/verification.json); `requirements.txt` mantiene su hash anterior tras reexportar, sin modificar versiones de producción.

**F1 cerrado; revisión independiente aprobada en el alcance auditado (S6 met), sin hallazgos abiertos.** Véanse el [recheck independiente](preprocesamiento/outputs/notebook_phase2/review.md#cierre-f1--recheck-limitado-de-la-primera-corrección) y la [verificación final del coordinador](preprocesamiento/outputs/notebook_phase2/coordinator_verification.json). El notebook ahora rechaza poblaciones de clasificación sin ambas clases 0/1 y clases con menos de tres grupos originales distintos. También verifica al final que cada split conserve ambas clases antes de cualquier fit o exportación. Tener tres grupos por clase es necesario, no suficiente: si el heurístico deja un holdout sin alguna clase se rechaza su asignación, sin afirmar inexistencia matemática de otra solución. Regresión no exige dos clases. Cinco pruebas por `clean_bookings` → `run_task` cubren el caso exacto de 600 filas/60 canceladas idénticas, dos grupos minoritarios, tres grupos con asignación voraz inválida y las dos poblaciones monoclase; comprueban ausencia de fit y de directorio destino. RED: cinco fallos antes del cambio; GREEN: cinco aprobados después. La figura usa anotaciones blancas sobre celdas altas/oscuras y negras sobre claras según `max(matrix)/2`. Nueva ejecución guardada: 17 celdas de código en 6.22 s, sin excepciones; tareas JSON y seis Parquet siguen exactamente iguales al canónico. La corrección conservó `src` y la entrega anterior. El revisor confirmó los cinco rechazos con cero fits, destinos ausentes y entrada intacta, además de H1/H2 válidos y regresión con todas las etiquetas 0. El coordinador confirmó por separado la ejecución aislada en **8.511 s**, Python 3.14.6, 17 celdas consecutivas sin errores, 13 docstrings, tareas completas/seis Parquet exactos y 30 hashes intactos. Su calidad final pasó: Ruff check/formato (9 archivos, incluido el adaptador del informe), ty de proyecto/Python extraído, pytest **36 passed en 3.10 s** y git diff --check. Estos tiempos corresponden a ejecuciones distintas; la salida guardada sigue siendo la de 6.22 s.

Límites: esto es clasificación retrospectiva del día previo a llegada, no predicción al crear reserva; ADR no es cotización inicial y la regresión condicionada a no cancelación tiene sesgo de selección. No demuestra causalidad, independencia entre clientes ni rendimiento futuro. Kaggle remoto y CI remota no se ejecutaron; se reutilizó el raw existente. La asignación de grupos es heurística, no exhaustiva. El kernel local emitió una advertencia TCP sin cifrar, sin error de ejecución. La revisión independiente está cerrada en el alcance documentado; no constituye un veredicto de revisión nativa.

### Dependencias exportadas

`preprocesamiento/requirements.txt` ya está exportado desde el lock, con versiones exactas y hashes. Para reproducir su generación, el comando indicado en su cabecera es:

```bash
uv export --frozen --no-dev --format requirements-txt --output-file requirements.txt
```

No editar `uv.lock` ni `requirements.txt` manualmente. `--no-dev` exporta dependencias de ejecución; las herramientas de desarrollo se instalan mediante `uv sync --frozen`. Aunque la guía exige el archivo de dependencias en fase 4, se incluye ahora por solicitud explícita para fase 2.

### Generar y compilar el informe

El adaptador documental `informe/generate_report_data.py` usa solo biblioteca estándar: transporta `eda.json` y `source.json` a `informe/datos.typ`, incorpora sus SHA-256 y **no recalcula estadísticas**. Después de generar el EDA:

```bash
# Desde preprocesamiento/:
uv run python ../informe/generate_report_data.py
# Volver a la raíz:
cd ..
typst compile informe/main.typ informe/fase1.pdf
```

El informe usa `@preview/charged-ieee:0.1.4`; una primera compilación puede requerir el registro de paquetes. La serialización se verificó: hashes de entrada, diccionario de 32 variables y 16 resúmenes cuantitativos; no incorpora resultados de fase 2 al informe de fase 1. El [PDF](informe/fase1.pdf) contiene 12 páginas, 32 tablas y siete figuras. La inspección visual independiente de las 12 páginas pasó, incluido el solapamiento corregido, y se contrastaron 230 filas numéricas sin discrepancias. La última recompilación terminó con salida 0, añade referencias metodológicas hasta cuatro entradas y sincroniza `source.json`: [registro de comprobación](informe/revision_pdf.json). SHA-256 vigente del PDF: `fdd73e9b8dad5070cce02569a2d471b29575696c91f0fd9ad8c93cd43f8df06f`. Actualización editorial posterior: únicamente la pregunta principal elegida por el usuario; recompilación y revisión visual de páginas 1/2 correctas, resto del texto equivalente y cifras intactas. Las revisiones independientes y comparaciones de píxeles descritas aquí conservan su alcance histórico sobre el PDF anterior; no se reejecutaron datos, modelos ni notebook. Las páginas modificadas 2/8/11/12 pasaron reinspección visual; no hay blancos ni recortes. Una compilación independiente a un directorio temporal terminó con salida 0 y reprodujo el texto y **todos los píxeles de las 12 páginas** del PDF publicado (comparación PyMuPDF); diferencias binarias por metadatos no se confunden con cambios visuales. El [informe académico independiente](informe/revision_academica.md) confirma también la última versión, con transporte profundo de eda/source y hashes coincidentes, cuatro referencias y recheck visual de páginas 2/8/10/11/12 aprobado.

## Estructura y artefactos

```text
informe/
  main.typ, *.typ, refs.bib       Artículo académico y bibliografía
  generate_report_data.py        Adaptador documental
  datos.typ                      Datos serializados
  imagenes/*.svg                 Figuras vectoriales versionables
  fase1.pdf                      PDF generado
preprocesamiento/
  pyproject.toml, uv.lock         Único proyecto y lock
  requirements.txt               Exportación de ejecución con versiones/hashes
  src/                           CLI, adquisición, EDA, limpieza y modelos
  tests/                         Pruebas pequeñas sin Kaggle
  data/raw/hotel_bookings.csv     Original, excluido de Git
  data/processed/                 Parquet, excluidos de Git
  outputs/                       JSON diagnósticos pequeños versionables
.github/workflows/ci.yml          Checks en push y pull_request
```

Salidas de `phase2`: `data/processed/cleaned.parquet` y `{classification,regression}_{train,validation,test}.parquet`. Diagnósticos: `outputs/source.json`, `eda.json`, `cleaning.json` y `phase2.json`. Las figuras son `adr_by_hotel.svg`, `arrivals.svg`, `cancellation_categories.svg`, `correlations.svg`, `distributions.svg`, `missingness.svg` y `relationships.svg`.

`.gitignore` excluye entornos, caches, raw/processed e intermedios voluminosos; mantiene `.gitkeep`, **figuras de `informe/imagenes/` y JSON útiles de `outputs/`**. Los JSON situados en raw/processed también se excluyen: colocar los diagnósticos versionables en outputs.

## Resultados del original y preparación

Fuentes: [eda.json](preprocesamiento/outputs/eda.json) y [cleaning.json](preprocesamiento/outputs/cleaning.json). Las cifras siguientes no proceden de modelos heredados.

| Diagnóstico original | Conteo |
|---|---:|
| Nulos en `children` | 4 |
| Nulos en `country` | 488 |
| Nulos en `agent` | 16340 |
| Nulos en `company` | 112593 |
| Repetidos adicionales respecto a filas únicas | 31994 |
| Filas pertenecientes a grupos repetidos | 40165 |
| ADR negativo / cero | 1 / 1959 |
| ADR superior al umbral IQR 211.065 | 3793 |
| Sin huéspedes conocidos / sin noches | 180 / 715 |

El JSON incluye porcentajes exactos por variable, estadísticas cuantitativas completas, frecuencias categóricas, correlaciones y χ² con frecuencias esperadas y V de Cramér. Las asociaciones son exploratorias: los repetidos pueden vulnerar independencia y no hay corrección por multiplicidad. El contraste por `market_segment` no satisface los supuestos registrados; un p-valor almacenado como cero es precisión numérica, no probabilidad matemáticamente nula.

### Ausencia, limpieza y transformaciones

- La ausencia se estudia por hotel, cancelación y segmento. Por ejemplo, los cuatro nulos de `children` están en reservas canceladas del hotel urbano. Las diferencias observadas pueden sugerir MAR condicional, pero **no demuestran MCAR/MAR/MNAR**; MNAR no se descarta. `agent`/`company` son identificadores nominales y sus NULL pueden significar no aplicable, no valores numéricos que deban promediarse. Se excluyen del baseline.
- Se descartan **826 filas distintas por `row_id`**, conservando **118564** (no se deduplican reservas). Reglas: ADR negativo **1**, conteos negativos **0**, huéspedes conocidos cero **180**, noches cero **715**, ADR ausente/no finito **0**. Los conteos **se solapan** y no deben sumarse. `cleaning.json` conserva `row_id`, huella original y motivos de cada rechazo.
- Se conservan repetidos, ADR cero y extremos positivos finitos. La cola IQR puede representar demanda real: no se recorta automáticamente ni se elige winsorización/logaritmo sobre test. Excluir el ADR negativo es una decisión de población del baseline, no prueba de un error en el original.
- Features: `total_nights = stays_in_weekend_nights + stays_in_week_nights` y `total_guests = adults + children + babies`. Los niños desconocidos siguen desconocidos en la feature; el cero auxiliar solo sirve para detectar ausencia de huéspedes conocidos.
- Tras dividir, cada pipeline ajusta en **train únicamente**: imputación numérica por mediana y `StandardScaler`; categorías nominales con One-Hot (`handle_unknown="ignore"`). La entrada categórica usa `__MISSING__` para nulos; la rama dispone además de imputación por moda. `phase2.json` registra medianas y medias aprendidas. El mes es nominal, no una magnitud ordinal arbitraria.

### Modelos y particiones finales

Se implementan `DummyRegressor(strategy="mean")` y `LinearRegression` para ADR en reservas no canceladas, y `DummyClassifier(strategy="prior")` y `LogisticRegression(max_iter=2000)` para cancelación retrospectiva. No hay selección de hiperparámetros ni ajuste sobre test. El desbalance se evalúa en train; el umbral de minoría severa es 0.10 y activa `class_weight="balanced"` en logística cuando corresponde, sin remuestreo automático.

Las huellas se calculan sobre las 32 columnas originales antes de limpiar; los repetidos se conservan juntos. La partición apunta a **75/12.5/12.5 % de filas**, dentro de cada clase para clasificación y globalmente para regresión, no a porcentajes de grupos. Ordena grupos por peso descendente con desempate de semilla 42 y asigna al mayor déficit relativo entre destinos donde encajen; si ninguno encaja, minimiza el aumento del error relativo cuadrático. Valida train en 70–80 % y holdouts no vacíos antes de entrenar/exportar; una incompatibilidad produce `ValueError`. Es una heurística voraz, **no un solucionador exhaustivo de factibilidad**: grupos indivisibles pueden impedir proporciones exactas.

X conserva `adults`, `children` y `babies`, con imputación de niños solo en train, para no perder adultos conocidos cuando faltan niños. `total_guests` se crea y conserva nulo si falta algún componente en cleaned y Parquet de cada split, pero **no se usa como predictor ni se imputa independientemente**. `total_nights` sí entra en X, en lugar de sus componentes. El One-Hot completo sigue siendo redundante con intercepto; mínimos cuadrados/SVD no implica inferencia fiable de coeficientes. Las huellas originales tienen solapamiento cero en todas las parejas de ambas tareas; el hash conserva un riesgo de colisión no nulo.

Fuente de todas las tablas siguientes: [phase2.json](preprocesamiento/outputs/phase2.json), ejecución final con semilla 42, SHA-256 `99b5e3ffafbeb2540a4e6d245fc43ab33d31130cf146cbf2de1d94c40dbb0a6c`. Se transcriben los valores almacenados, sin aproximar métricas ni proporciones. Las fracciones son proporciones de **filas dentro de cada población**, no porcentajes de grupos.

| Tarea (población) | Partición | Filas | Fracción | Huellas originales únicas |
|---|---|---:|---:|---:|
| Clasificación (118564) | train | 88923 | 0.75 | 64984 |
| Clasificación | validación | 14821 | 0.12500421713167573 | 10827 |
| Clasificación | test | 14820 | 0.12499578286832429 | 10827 |
| Regresión (74388 no canceladas) | train | 55791 | 0.75 | 46991 |
| Regresión | validación | 9299 | 0.12500672151422274 | 7831 |
| Regresión | test | 9298 | 0.12499327848577728 | 7830 |

Ambas tareas cumplen 70–80 % de filas train y 20–30 % de filas entre validación y test. La prevalencia de cancelación de clasificación es `0.3725920178131642` en train, `0.3725794480804264` en validación y `0.3726045883940621` en test. No hay desbalance severo con umbral 0.10: no se aplica remuestreo ni ponderación balanceada en esta ejecución.

| Tarea | Pareja | Huellas originales compartidas | Patrones X compartidos |
|---|---|---:|---:|
| Clasificación | train–validación | 0 | 2934 |
| Clasificación | train–test | 0 | 2848 |
| Clasificación | validación–test | 0 | 807 |
| Regresión | train–validación | 0 | 2073 |
| Regresión | train–test | 0 | 2058 |
| Regresión | validación–test | 0 | 587 |

Los patrones X cuentan perfiles de predictores idénticos: **no identifican clientes ni prueban que sea la misma reserva**. Se diagnostican, no se eliminan ni se agrupan borrando el target; la separación comprobada es por huella de la fila original completa.

| Tarea | Split | Grupos repetidos | Filas en esos grupos | Fracción de filas del split |
|---|---|---:|---:|---:|
| Clasificación | train | 6104 | 30043 | 0.3378540984897046 |
| Clasificación | validación | 1015 | 5009 | 0.33796639902840564 |
| Clasificación | test | 1015 | 5008 | 0.3379217273954116 |
| Regresión | train | 3667 | 12467 | 0.22345898083920346 |
| Regresión | validación | 610 | 2078 | 0.22346488869770942 |
| Regresión | test | 610 | 2078 | 0.22348892234889223 |

`group_sizes` en el JSON añade histogramas, mínimos, medianas y máximos. Equilibrar estas masas observadas no demuestra independencia de clientes o grupos comerciales reales.

#### Regresión de ADR

| Modelo | Holdout | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| DummyRegressor | validación | 37.39264471020401 | 49.1553696232623 | -5.464820561185135e-05 |
| DummyRegressor | test | 37.07037323659072 | 49.109540722039334 | -2.4287331943595447e-05 |
| LinearRegression | validación | 22.34694413575031 | 31.393460463643393 | 0.5920939180969427 |
| LinearRegression | test | 22.529626701934447 | 31.3703220066847 | 0.5919468302415928 |

#### Clasificación retrospectiva de cancelación

| Modelo | Holdout | Accuracy | Precisión | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| DummyClassifier | validación | 0.6274205519195736 | 0.0 | 0.0 | 0.0 | 0.5 |
| DummyClassifier | test | 0.6273954116059379 | 0.0 | 0.0 | 0.0 | 0.5 |
| LogisticRegression | validación | 0.7307199244315499 | 0.76423886779427 | 0.4009416877942774 | 0.5259532010927663 | 0.7540278814743275 |
| LogisticRegression | test | 0.7442645074224021 | 0.763542300669507 | 0.4543643607388627 | 0.5697093551316984 | 0.7746932545926504 |

Matrices de confusión: filas = clase real `[0, 1]`, columnas = predicción `[0, 1]`; clase positiva 1 = cancelación, estructura `[[TN, FP], [FN, TP]]`.

| Modelo | Holdout | Matriz |
|---|---|---|
| DummyClassifier | validación | `[[9299, 0], [5522, 0]]` |
| DummyClassifier | test | `[[9298, 0], [5522, 0]]` |
| LogisticRegression | validación | `[[8616, 683], [3308, 2214]]` |
| LogisticRegression | test | `[[8521, 777], [3013, 2509]]` |

El modelo lineal mejora los errores frente a la media de train; la logística mejora al Dummy pero todavía omite numerosas cancelaciones. La precisión decimal reproduce el artefacto, no cuantifica incertidumbre estadística. Estos son baselines retrospectivos, no resultados de modelos avanzados ni una validación prospectiva.

## Pruebas y CI

Desde `preprocesamiento/`:

```bash
uv run ruff check . ../informe/generate_report_data.py
uv run ruff format --check . ../informe/generate_report_data.py
uv run ty check
uv run ty check ../informe/generate_report_data.py
uv run pytest
```

Verificación local posterior al ajuste tipográfico: todos los comandos anteriores terminaron correctamente (salida 0); formato comprobó **7 archivos**, ty verificó proyecto y adaptador, y pytest informó **20 passed in 4.28s**, con Python **3.14.6**. `uv sync --frozen` también pasó. Las pruebas usan datos pequeños sin Kaggle: esquema, faltantes/invalidaciones, features, limpieza, particiones, exclusión de fugas, fit train-only y métricas.

[CI](.github/workflows/ci.yml) ejecuta `uv sync --frozen` y los cuatro checks del proyecto (Ruff, formato, ty y pytest) desde `preprocesamiento/`, en `push` y `pull_request`, con Python 3.14. No descarga el dataset ni ejecuta el EDA real, los modelos con datos reales o Typst; no requiere secretos de Kaggle. Tags estables completos verificados públicamente: [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) y [setup-uv v10.3.3](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.3). Se inspeccionó la estructura y las entradas de las acciones; **CI remota no ejecutada**.

## Límites de interpretación

- **Retrospectivo, no predictor al crear la reserva:** la fotografía original es del día previo a llegada; algunas reservas ya podrían estar canceladas. Una fecha de llegada y un holdout aleatorio no prueban generalización futura ni disponibilidad prospectiva.
- Se excluyen estado/fecha de estado, habitación asignada, cambios, espera, ADR como predictor, depósito, parking y solicitudes. `country` puede conocerse solo al check-in; `deposit_type` acumula pagos anteriores a llegada **o cancelación**, no solo política contractual inicial. No usar su asociación fuerte como justificación de inclusión.
- La regresión se condiciona a reservas no canceladas (**74388** tras la limpieza inspeccionada): tiene sesgo de selección y no estima el precio causal ni la tarifa inicial para todas las reservas. ADR extremo conservado hace las métricas sensibles a colas.
- La unidad monetaria de ADR no está acreditada por la evidencia fuente: MAE/RMSE se expresan en unidades de ADR, sin atribuir euros u otra moneda.
- Dos hoteles y un periodo histórico limitado; faltantes y reservas grupales limitan inferencia. Las asociaciones no son causales y las referencias simples no certifican despliegue.

## Checklist académico y evidencia

[Checklist de entrega: los 16 requisitos exactos y sus evidencias](preprocesamiento/outputs/delivery_checklist.md), a partir de la [transcripción de 3.1/3.2](preprocesamiento/outputs/specification.md#checklist-exacto-de-31-y-32) y la [extracción original](preprocesamiento/outputs/guide.txt). Los cuatro modelos, métricas completas y controles adicionales son exigencias de esta entrega, no requisitos atribuidos literalmente a la guía de fase 2. Resumen operativo, no certificación final del PDF:

- [x] Procedencia pública y licencias verificadas; esquema real de 32 variables y más de 600 registros limpios.
- [x] Pregunta principal, tres secundarias y objetivos SMART; revisión académica y visual de las 12 páginas realizada.
- [x] EDA original generado: univariado, frecuencias, correlaciones, dispersión, χ² y calidad; fuentes JSON y siete figuras SVG disponibles.
- [x] Diagnóstico de ausencia, reglas justificadas y trazabilidad de 826 descartes; repetidos conservados.
- [x] Dos features derivadas, encoding nominal y escalado/imputación ajustados en train.
- [x] Cuatro baselines implementados y ejecutados en el pipeline real; test sin selección de hiperparámetros.
- [x] `requirements.txt` exportado desde el lock para esta fase 2, aunque la guía lo sitúe en fase 4.
- [x] Registrar adquisición local y verificación diferenciadas; declarar como limitación que la descarga original tiene fecha desconocida (`null`), sin atribuírsela al mtime.
- [x] Particiones finales por filas dentro del intervalo exigido; métricas de los cuatro modelos en validación/test, desbalance y solapamientos documentados desde el JSON regenerado.
- [x] Ficha/diccionario, 230 filas numéricas y maquetación contrastados; informe de 12 páginas, 32 tablas y siete figuras.
- [x] Ruff, formato (7 archivos), ty de proyecto/adaptador y pytest (20 aprobadas en 4.28s), todos con salida 0; CI remota no ejecutada.
- [x] PDF final compilado e inspeccionado tras sincronizar procedencia y cuatro referencias; sin blancos ni recortes. Build independiente reproducido con texto y píxeles idénticos en las 12 páginas.
- [x] Revisiones independientes del [pipeline](preprocesamiento/outputs/review_pipeline.md#cierre-limitado-h1h2-y-regresión-directa-de-multiplicidad) y [académica](informe/revision_academica.md) cerradas sin bloqueadores: phase2 aislado idéntico, raw intacto, métricas de cuatro modelos recalculadas y PDF final contrastado. CI remota continúa sin ejecutar.
