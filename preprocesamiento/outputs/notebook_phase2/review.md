# Revisión independiente del notebook de fase 2

**Resultado actual: F1 corregido y cerrado mediante recheck limitado; S6 met.** Véase la sección final con los hashes nuevos y probes independientes. La auditoría original se conserva abajo como evidencia histórica: su hallazgo pendiente ya no describe la versión actual. Esta es una auditoría genérica independiente, no un veredicto de revisión nativa.

**Resultado histórico inicial:** un hallazgo de prioridad media pendiente en S6; ejecución real y autonomía aprobadas. No se recomendaba cerrar todos los requisitos hasta resolver o disponer el hallazgo.

## Alcance y versión revisada

Revisión de `odd/tasks/hotel-booking-notebook-fase2.md` completa, guía `outputs/guide.txt` (especialmente 3.2), las **29 celdas completas** de `fase2.ipynb` (Markdown, código y salidas, incluida inspección visual de la figura), `tests/test_notebook.py`, `outputs/notebook_phase2/verification.json`, `pyproject.toml` y sección del notebook de `../README.md`. No se auditó nuevamente la entrega anterior ni se modificaron notebook, código, pruebas, configuración o resultados.

- Notebook SHA-256: `74df57616952ff02620d766a263bbfb6c45516b531ad5ea4b8231443b5462844`.
- **Hash estable del código:** `ad05521cc0ce2016e4ef506c4c69943c7f2d7e15cbdebf20d493f38c2cca68b7`.
- Definición del hash estable: SHA-256 UTF-8 de `"\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")`. Excluye timestamps, metadatos y outputs; incluye comentarios/docstrings.
- Referencias a celdas: numeración **1–29** del documento, no `execution_count`.

La independencia aquí significa que el revisor no escribió el notebook y derivó sus fixtures de la guía y de la solicitud. No significa una certificación externa de procedencia/licencia ni una revisión nativa.

## Hallazgo F1 — media — S6: incompatibilidad de estratificación no rechazada

**Ubicación:** celda 17, `split_bookings`; efecto visible en celda 23, `run_task`.

La asignación se realiza por clase, pero la validación final solo comprueba fracción global de train y holdouts no vacíos. No verifica que las clases puedan estar representadas en los tres destinos. Con **600 reservas**, 60 canceladas idénticas (un único grupo original) y 540 no canceladas distintas, acepta:

| Split | Filas | Canceladas | Prevalencia |
|---|---:|---:|---:|
| train | 465 | 60 | 0.12903225806451613 |
| validation | 68 | 0 | 0 |
| test | 67 | 0 | 0 |

La población tiene 10 % de cancelaciones. Mantener el único grupo minoritario íntegro hace imposible una partición estratificada con ambas clases en los tres splits. **Debe rechazarse esa incompatibilidad explícitamente**, no presentar los holdouts aceptados como una partición estratificada válida. Esto se deriva de S6 y de la guía 3.2, no de una exigencia de proporciones exactas ni de un solver exhaustivo nuevo.

La reproducción completa mediante `clean_bookings` → `run_task` terminó sin excepción, no descartó reservas y exportó los tres Parquet. Ambos modelos devolvieron `roc_auc=None` en ambos holdouts. La función de métricas trata correctamente el AUC indefinido; el defecto es aceptar la partición antes de entrenar/exportar, no fabricar un AUC.

**Impacto acotado:** no invalida las métricas del CSV público inspeccionado; sí incumple la validación/rechazo de incompatibilidad que promete el cuaderno. El test actual de 4400 filas solo cubre un grupo mayor que el 80 % global y no detecta este caso por clase.

**Fixture/causa para el autor:** añadir una prueba offline por la interfaz completa con este grupo minoritario único; esperar `ValueError` informativo antes de entrenar/exportar y destino inexistente. Validar factibilidad/representación por clase sin romper grupos. No se propone modificar proporciones, agrupar perfiles X ni exigir un optimizador general. No se realizaron arreglos.

### Reproducción autónoma del hallazgo

Fragmento causal del segundo heredoc efectivamente ejecutado (se omiten únicamente extracción de imagen e impresión de hashes). Ejecutar desde `preprocesamiento/`, con dependencias ya disponibles; genera solo scratch temporal:

```bash
UV_NO_SYNC=true uv run --group notebook python - <<'PY'
import ast, json, tempfile
from pathlib import Path
import polars as pl
nb = json.loads(Path('fase2.ipynb').read_text())
code = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
constants = {'COLUMNS', 'CATEGORICAL', 'STRINGS', 'SCHEMA', 'DATASET', 'SEED'}
nodes = [n for n in ast.parse(code).body
         if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
         or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
s = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'review', 'exec'), s)
base = {k: 'A' if t == pl.String else 0 for k, t in s['SCHEMA'].items()}
base.update(adults=2, children=0., babies=0, stays_in_week_nights=2,
            stays_in_weekend_nights=0, adr=100., is_canceled=1)
raw = pl.DataFrame([base] * 60 + [base | {'lead_time': i + 1, 'is_canceled': 0}
                   for i in range(540)], schema_overrides=s['SCHEMA'])
frame, audit = s['clean_bookings'](raw)
destination = Path(tempfile.mkdtemp(prefix='hotel-review-repro-')) / 'splits'
result = s['run_task'](frame, 'classification', destination)
print(destination, audit['removed_rows'])
print([(k, v['rows'], v['cancellation_prevalence']) for k, v in result['splits'].items()])
print({k: {h: v['roc_auc'] for h, v in m.items()} for k, m in result['metrics'].items()})
PY
```

Este bloque documentado usa un directorio nuevo para quien lo reproduzca; la ejecución observada del revisor usó el scratch concreto registrado abajo.

## Matriz de cumplimiento histórica (antes de corregir F1)

`met` significa evidencia suficiente para este alcance; `unverified` no equivale a defecto.

| Spec | Estado | Evidencia / límite |
|---|---|---|
| S1 | unverified (histórico); met en preservación | Un solo proyecto Python >=3.14; configuración inspeccionada. El coordinador acredita 30 hashes preservados. No se puede demostrar retrospectivamente el orden de lectura del autor antes de escribir; no se amplió a auditar fase 1. |
| S2 | met | Celdas 3–8, 28: definiciones completas, rutas configurables, sin import de `src` ni CLI; comparación canónica opcional. Aislamiento real del coordinador sin pyproject/src/informe/outputs previos. |
| S3 | met | 29 celdas, 17 de código. Narrativa secuencial completa y rigurosa sin exigir extensión ornamental. Las 13 funciones, incluidas las internas `sanitize` y `x`, tienen docstring española. Comentarios explican huellas, déficit relativo y frontera sklearn; no son comentarios redundantes masivos. `run_task` coordina un flujo cohesivo, con evaluación/preprocesamiento/partición separados. |
| S4 | met | Celdas 9–13, 19–20: hipótesis MCAR/MAR/MNAR sin afirmación causal; IDs nominales/ausencia estructural; remoción razonada por dominio, retención justificada de cero/colas; 826 descartes distintos con motivos; repetidos conservados; dos derivados; Polars y pipelines sin pandas. SciPy está disponible pero no se exige una prueba estadística innecesaria para fase 2. |
| S5 | met | Celdas 1, 14–15, 23, 29: fotografía retrospectiva, regresión condicionada a no cancelación, exclusiones explícitas, adultos en X, total_guests nulo cuando corresponde y fuera de X. No suma redundante de componentes y derivados. Sin moneda/fecha de descarga inventada. Probe H1 propio aprobado. |
| S6 | **unmet** | F1. En el CSV real y fixtures factibles se cumplen pesos por filas, grupos intactos, déficit relativo, estratificación, desbalance train-only, cuatro baselines y métricas completas sin tuning de test. Falta rechazo por incompatibilidad de clase. |
| S7 | met | Celdas 24–29: tablas legibles, figura pertinente, conclusiones y límites. Directorios separados, manifiesto y trazabilidad; comparación de tareas completas sin copiar métricas. CSV y canónicos preservados según coordinador. |
| S8 | met | Salidas guardadas con counts 1..17, sin errores/rutas personales/secretos detectados. Prueba funcional aislada del coordinador reutilizada explícitamente; README documenta APIs reales y grupo notebook mínimo. Descarga remota no ejecutada: límite honesto, no fallo cuando raw existe. |
| S9 | met con evidencia atribuida | Lectura completa de pruebas: fixtures offline/AST, H1/H2, fronteras, loader y exportaciones. `verification.json` informa 31 pytest, Ruff/formato y ty proyecto/Python extraído aprobados; no se repitieron sin valor. RED histórico solo acreditado por reporte del autor, no observado personalmente. La cobertura por clase de F1 falta. |
| S10 | met para esta revisión; cierre pendiente | Auditoría independiente documentada, probes propios y ejecución funcional aislada atribuida al coordinador. No se declara corregido F1 ni aprobada una revisión nativa. |

### Guía 3.2 y autonomía

- **Ausencia y extremos: met.** El análisis precede imputación; no se presume mecanismo demostrado. Conservar extremos positivos está razonado; la guía no obliga a winsorizar todo valor IQR.
- **Dos derivados, encoding y escalado: met.** `total_guests` es una derivación honesta aunque no entre al modelo. Ajuste exclusivamente train probado mediante interfaz de tarea; categorías nuevas seguras.
- **Partición/desbalance: unmet solo por F1.** La población pública sí cumple; el límite de incompatibilidad por clase no.
- **Baseline: met.** Cuatro referencias sencillas, métricas completas adicionales a lo literalmente exigido por 3.2; no modelos de fase 3.
- **Autonomía contractual: met.** La única lectura de resultados previos es una comparación opcional posterior a calcular modelos. El entorno y dependencias deben existir: autosuficiente no significa distribuir Python o evitar librerías. El CSV real existente basta; no se exige red para demostrarlo.

## Probes independientes y resultados observados

Las definiciones se extrajeron por JSON/AST; no se importaron `src`, pruebas ni fixtures del autor. Se ejecutaron solo imports, funciones y constantes confiables, sin ejecutar las celdas de exportación canónica.

1. **H1 y derivados:** fixture propio de 120 reservas, dos filas con niños ausentes y adultos 1/2; retiene ambas, `total_guests=[None,None]`, noches=3, ningún descarte. Aprobado.
2. **Loader:** CSV pequeño de esquema completo con `NA`, `NULL` y vacío en children/agent/company; todos quedan nulos, IDs String y bytes SHA-256 idénticos. Aprobado.
3. **Fit train-only por interfaz completa:** determinar splits del fixture de 120 filas; conservar huellas/pertenencia y modificar exclusivamente lead_time a `1e12` y hotel a `HOLDOUT_ONLY` en validación/test. Ejecutar `run_task` antes/después. Los diagnósticos de **ambos modelos** son idénticos en medianas, medias del scaler y vocabulario; categoría nueva ausente de features entrenadas. Transformación/predicción terminaron sin error. No se afirma invariancia de métricas. Aprobado.
4. **H2 desigual propio:** 360 grupos con pesos 7 para los primeros 20, 3 para los siguientes 40 y 1 para los restantes: 560 filas. Regresión y clasificación producen 420/70/70, conservan masa/huellas y repetidos en los tres splits. Clasificación tiene prevalencia 0.5 en cada uno. Aprobado.
5. **Fronteras solicitadas:** 399 singleton + grupo de 201: 600 filas → 450/75/75. 399 singleton + grupo de 4001: 4400 filas → rechazo explícito antes de modelar. Aprobado.
6. **Grupo minoritario indivisible:** probe directo y luego interfaz completa de 600 filas → F1 reproducido, exportación indebida, AUC nulos. Incumplimiento observado.
7. **Contraste pasivo:** JSON `tasks` completo notebook/canónico idéntico; hash raw `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`; notebook sin salidas error. Figura visualmente legible y consistente con las cuatro matrices publicadas.

## Ejecución funcional reutilizada y comandos reales

**No se repitió el kernel completo:** el coordinador entregó y se leyó `outputs/notebook_phase2/coordinator_verification.json`. Su ejecución real informa Python 3.14.6, 29/17 celdas, counts 1..17, cero errores, **8.341 s**; solo notebook y CSV en scratch, sin src/pyproject/informe/resultados previos; tareas JSON completas y seis split Parquet exactos; **30 hashes originales intactos**. El hash de notebook coincide con el calculado por el revisor. Es evidencia funcional ajena explícitamente atribuida, no una ejecución reclamada como propia.

Comandos realmente usados por el revisor desde la raíz (heredocs de inspección/probes en memoria; ningún runner de suite o kernel completo):

- `git status --short; pwd; find preprocesamiento/outputs preprocesamiento/tests -maxdepth 3 -type f | sort` → inspección del árbol y cambios preexistentes.
- `cd preprocesamiento && uv run --group notebook python - <<'PY' … PY` → JSON/AST, hash de código, inventario de celdas/docstrings y fuentes finales; salida 0. **uv informó `Installed 33 packages in 621ms` automáticamente**. No se ejecutó `uv sync` ni instalador explícito; no se afirma que esta primera llamada fuera libre de sincronización implícita.
- `cd preprocesamiento && UV_NO_SYNC=true uv run --group notebook python - <<'PY' … PY` → primer heredoc de probes 1–6 y contraste, salida 0; F1 expuesto por datos impresos, no una prueba artificialmente declarada aprobada.
- `cd preprocesamiento && UV_NO_SYNC=true uv run --group notebook python - <<'PY' … PY` → segundo heredoc: F1 por interfaz completa, tres Parquet en scratch, lectura de fuente celda 26, extracción de imagen para inspección y hashes; salida 0. La reproducción causal está arriba.

No se ejecutaron pytest/Ruff/ty otra vez; sus resultados previos están atribuidos al reporte, no se presentan como comprobación independiente nueva. No hubo descarga de Kaggle ni kernel remoto. El coordinador informó advertencia TCP local sin cifrado, no error de ejecución.

## Límites y retiro de scratch

- Histórico de lecturas del autor, RED histórico y descarga remota: no verificados personalmente. No constituyen nuevos requisitos ni se exige red.
- Scratch del revisor retenido para el coordinador, sin borrado manual: `/tmp/gphxzMQG/gacQE25e/hotel-notebook-independent-elffff64` (fixtures, Parquet y PNG de inspección).
- Scratch funcional comunicado por el coordinador: `/tmp/gphxzMQG/hotel-parent-notebook-55lbc0ge`.
- Las salidas del notebook no contienen secretos ni rutas personales detectadas. Las rutas temporales de este informe identifican evidencia técnica solicitada, no rutas del autor dentro del notebook.
- La única escritura de entrega de esta auditoría es este `review.md`; los hallazgos no autorizan modificar el artefacto auditado.

## Cierre F1 — recheck limitado de la primera corrección

**Estado actual: F1 cerrado; S6 pasa de unmet a met.** No quedan hallazgos abiertos dentro de F1 ni de sus regresiones directas comprobadas. No se hizo una nueva auditoría completa, no se buscaron problemas ajenos y no se modificó notebook/tests/código/configuración/README/resultados. Los estados históricos de otros requisitos no se reevalúan aquí.

### Snapshot y cambio observado

Hashes calculados por el revisor, coincidentes con el snapshot del autor y la ejecución final del coordinador:

- Notebook SHA-256: `abc95a5d847a66fc02171278debce92bc83c3a2eb8f12c569eb18be6e681ef8b`.
- Código SHA-256 (misma definición estable anterior): `b588314455300d35837758a62936f49f9fc8837be439d6f08e9f1e5ba4791269`.

Se inspeccionaron únicamente las celdas 16–17 para F1 y la cobertura pertinente de pruebas. La celda 17 exige población con clases `{0,1}`, al menos tres grupos originales por clase y, tras la asignación, ambas clases en cada split. Las guardas están condicionadas a clasificación; `run_task` obtiene los splits antes de cualquier fit/exportación. La celda 16 explica en español que tres grupos es una condición necesaria, no suficiente, y que rechazar un resultado del heurístico no demuestra imposibilidad general. La docstring de partición sigue en español.

### Probes propios realmente ejecutados

JSON/AST e imports/definiciones confiables del snapshot actual; sin importar pruebas/fixtures del autor ni `src`. Para cada rechazo se instrumentó `Pipeline.fit` y se comprobó `frame.equals(copia_previa)`, cero llamadas a fit y ausencia del directorio de destino. No se hizo sync, instalación ni descarga.

| Probe | Resultado observado |
|---|---|
| F1 exacto: 600 filas, 60 positivas idénticas + 540 negativas distintas | `ValueError`: `Cannot stratify three splits with intact original groups: class 1 has 1 original groups; at least 3 required`. **fit=0, destino ausente, entrada intacta.** |
| Dos grupos positivos de pesos 37/23 + 540 negativas distintas | Mismo rechazo informativo con `class 1 has 2 original groups`. **fit=0, destino ausente, entrada intacta.** |
| 600 filas distintas, solo clase 0 | `Classification requires both target classes [0, 1]; observed [0]`. **fit=0, destino ausente, entrada intacta.** |
| 600 filas distintas, solo clase 1 | Mismo mensaje con `observed [1]`. **fit=0, destino ausente, entrada intacta.** |
| Tres grupos positivos **desiguales** 22/19/19 + 540 negativas distintas | Supera la precondición de grupos, pero la postcondición rechaza: `Weighted greedy assignment did not preserve all target classes in every split: expected [0, 1], observed {'train': [0, 1], 'validation': [0], 'test': [0]}; heuristic result rejected, not an exhaustive feasibility verdict`. **fit=0, destino ausente, entrada intacta.** |
| Observación causal pre/postguarda del caso 22/19/19 | En una copia AST **solo en memoria**, retirar únicamente el último `if classification` deja 465/68/67 filas, clases `[0,1]` / `[0]` / `[0]`. La función original rechaza ese resultado. No se alteró el archivo ni se entrenó/exportó esta copia. Demuestra que la postcondición es activa, no solo documental. |
| H2 válido propio: 360 grupos, pesos 7/3/1, 560 filas | Clasificación y regresión: **420/70/70**, sin cruzar huellas, repetidos en cada split; clasificación 0.5 en cada destino. |
| H1 válido propio: 120 reservas, niños ausentes en dos filas con adultos 1/2 | Se conservan adultos y `total_guests=NULL`; tarea completa de clasificación pasa con **90/16/14**, dos fits y tres Parquet en scratch. No se exige exactitud 75/12.5/12.5 con grupos/clases discretos. |
| Regresión con 600 reservas, **todas etiquetas 0** | `run_task(..., 'regression', ...)` pasa: **450/75/75**, dos fits, tres Parquet, población completa. Las guardas de clase no afectan esta tarea. |

**Archivos no producidos:** no existen los destinos de los cinco casos rechazados (`exact-F1`, `two-groups`, `only-zero`, `only-one`, `three-unequal`). Únicamente los dos casos válidos generaron sus seis Parquet, todos en scratch externo. No hubo exportaciones canónicas.

### Comandos y disciplina de evidencia

- `git status --short; cd preprocesamiento && UV_NO_SYNC=true uv run --group notebook python - <<'PY' … PY` → salida 0; inspección acotada de celdas 16–17 y hashes actuales.
- Primer heredoc de fixtures con `cd preprocesamiento && UV_NO_SYNC=true uv run --group notebook python - <<'PY' … PY` → **salida 1, `IndentationError: unexpected indent`**, por una línea accidental del script del revisor. Python no llegó a ejecutar el script ni creó scratch. No fue un fallo del notebook ni evidencia RED de F1.
- Segundo heredoc corregido, mismo comando → **salida 0**, todos los probes de la tabla observados; una corrección del script, sin cambios al artefacto auditado. `Pipeline.fit` restaurado en `finally` dentro del proceso.

No se fabricó un nuevo RED: la falla original ya está documentada en la auditoría histórica. El autor comunicó cinco casos RED→GREEN; eso no se presenta como observado personalmente. El GREEN independiente aquí son los cinco rechazos correctos y las regresiones válidas de la tabla.

### Evidencia funcional final atribuida al coordinador

Se leyó la actualización de `outputs/notebook_phase2/coordinator_verification.json`, también comunicada por el coordinador: Python **3.14.6**, **29/17 celdas**, counts 1..17, **cero errores, 8.511 s** en aislamiento con solo notebook+CSV, sin src/informe/pyproject/outputs previos. Tareas JSON completas y seis Parquet exactos a canónicos, **30 hashes intactos**, 13 funciones con docstrings. Ambos hashes coinciden con este recheck. No se duplicó el kernel.

La ejecución propia de calidad de 36 pruebas del coordinador estaba pendiente al recibir su última comunicación; este recheck no inventa su resultado ni repite la suite. No se reabrieron descarga remota, fase 1 ni otros requisitos.

### Scratch del recheck

Retenido para retiro del coordinador, sin borrado manual: `/tmp/gphxzMQG/gaPgGqII/hotel-review-F1-recheck-3hyg3m7v` (seis Parquet de casos válidos; ningún destino de casos rechazados). El scratch funcional final del coordinador es `/tmp/gphxzMQG/hotel-parent-notebook-final-pm81dd65`. Los scratch históricos anteriores siguen documentados arriba, sin asumir su retiro.
