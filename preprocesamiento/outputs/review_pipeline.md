# Revisión independiente T2 — S2–S8/S12

Revisión técnica genérica solicitada por el usuario, no veredicto native gentle-review. Sin correcciones de implementación, instalación, red ni descarga. Leídos ODD hasta `## Log`, `outputs/specification.md`, guía completa extraída (especialmente 3.2), cinco módulos y tests completos, configuración, lock parseado íntegramente, requirements y CI. No se audita Typst/PDF (T3). Rutas siguientes relativas a `preprocesamiento/`, salvo indicación.

## Resultado por requisito

**Estado vigente:** H1/H2 cerrados mediante el recheck limitado al final de este documento. Las secciones de hallazgos, comandos y hashes iniciales se conservan como auditoría histórica; no describen defectos pendientes del candidato nuevo.

| Spec | Estado | Evidencia / límites |
|---|---|---|
| S2 | **met** | Python 3.14.6; único proyecto, Polars/NumPy/SciPy/sklearn/matplotlib/kagglehub; CLI cuatro comandos y rutas configurables (`src/main.py:14–48`). Todos los pins de requirements coinciden con lock; no pandas en código ni dependencias bloqueadas. No se certifica históricamente cómo se generó el lock. |
| S3 | **met** local; descarga **unverified** | CSV real 119390×32, hash/esquema coincidentes con source; agent/company String. Original intacto durante all. `src/data.py:77–94,97–160`: valida/reutiliza original, fallo explícito probado mediante mock. source declara honestamente `local_cached` y fecha original desconocida (`outputs/source.json:3–8`); no acredita descarga ni versión remota por sí solo. |
| S4 | **met** | Recomputados todos los resúmenes cuantitativos (NumPy/SciPy), modos, frecuencias categóricas y siete χ²/esperados/V; coinciden. Convenciones y limitaciones explícitas (`src/eda.py:22–108,233,288`; `outputs/eda.json:7592,8659,8930`). Correlaciones excluyen IDs/calendario; dispersión, ADR/hotel, cancelación, duración/anticipación y siete SVG. χ² informa celdas pequeñas, dependencia de grupos y multiplicidad: no interpretar p como causalidad. |
| S5 | **met** | H1 cerrado: adults permanece en X, total_guests se conserva como derivado trazable pero no se imputa ni se usa en X. Real40679 y fixtures3/5 adultos PASS; ver cierre. Evidencia histórica de limpieza/826 descartes conservada. |
| S6 | **met** | Regresión 74388 no canceladas; clasificación retrospectiva 118564; Dummy y modelos básicos. Exclusiones conservadoras y sesgo temporal/selección expresos (`src/main.py:69–84`, `src/modeling.py:27–47,141–193`). Todas las métricas validation/test recalculadas mediante matriz independiente y fórmulas propias. No selección/fit sobre test en código; no es posible probar desconocimiento histórico de sus resultados. |
| S7 | **met** | H2 cerrado: ponderación por filas/clase y déficit relativo; valida70–80% train y holdouts no vacíos. Clasificación88923/14821/14820; regresión55791/9299/9298: train75% exacto. Fixtures600 factible/4400 imposible PASS, multiplicidad holdouts preservada; ver cierre y limitación heurística. |
| S8 | **met** cobertura; **unverified** RED histórico/CI remota | Nuevas regresiones H1/H2/multiplicidad leídas (`tests/test_pipeline.py:75–167`); probes independientes del recheck PASS. Coordinador comunica20tests/Ruff/format/ty PASS, no repetidos en este recheck. Checks propios históricos14tests conservados abajo. CI remota e historia RED siguen sin certificarse. |
| S12 | **met** | Solo Dummy, LinearRegression y LogisticRegression; sin pandas, modelos avanzados, tuning ni segundo proyecto. |

## Hallazgos históricos priorizados (cerrados en el recheck final, sin cambios de código del revisor)

### H1 — P2: la imputación de huéspedes pierde información conocida

`src/cleaning.py:17–18` calcula `adults + children + babies` antes de imputar: children nulo propaga nulo a total_guests. `src/modeling.py:27–47,59` omite adults de X y mediana-imputa children y total_guests **independientemente**. Omitir adults evita colinealidad cuando la suma está completa, pero no conserva esa información cuando falta children.

**Caso real:** row_id **40679**, adults=3, children=null, babies=0; permanece en classification_train. Medianas aprendidas children=0 y total_guests=2 (`outputs/phase2.json`, transforms). El modelo recibe total_guests=2 pese a conocer al menos tres adultos. No hay fuga, pero sí pérdida de un dato conocido y ruptura del significado de la feature. Las otras tres filas con children faltante tienen adults=2.

**Test propuesto:** entrenar con children observado, luego transformar reserva con adults=3/children=null/babies=0; verificar que huéspedes transformados antes de escalar respetan adultos+bebés+children imputado usando únicamente estadísticas train. Incluir adults distintos y ausencia en validation/test; conservar adults fuera de X final. El probe sintético independiente también observó adults=5, children imputado=1 y total_guests imputado=5, en vez de6. Resolver la política en ingeniería/imputación, no reintroducir redundancia sin justificarla. Impacto real acotado; no invalida las métricas recalculadas, que reproducen el comportamiento actual.

### H2 — P2: 75% de grupos no impone 70–80% de filas

`src/modeling.py:92–108` divide índices de grupos, sin restricción/validación de sus pesos. `tests/test_pipeline.py:74–86` solo prueba una distribución de tamaños favorable. El CSV actual cumple; no afirmar que esta función garantiza la regla por filas.

**Reproducción determinista breve** (desde `preprocesamiento`, sin exportar):

```python
import polars as pl
from src.cleaning import clean_bookings
from src.modeling import split_bookings
raw = pl.read_csv('data/raw/hotel_bookings.csv', n_rows=1,
    null_values=['NULL', 'NA', ''],
    schema_overrides={'agent': pl.String, 'company': pl.String,
                      'children': pl.Float64})
base = raw.to_dicts()[0]
rows = [dict(base, lead_time=i, is_canceled=i % 2, children=0.0,
             adults=2, babies=0, stays_in_week_nights=2)
        for i in range(400)]
f, _ = clean_bookings(pl.DataFrame(rows, schema=raw.schema))
weighted = pl.concat([f, *[f.head(1)] * 200])
parts = split_bookings(weighted, classification=True)
print({k: v.height for k, v in parts.items()})
```

**Observado:** train500/validation50/test50 sobre600 = **83.333/8.333/8.333%**, sin excepción. Una asignación agrupada450/75/75 sí es factible (grupo201 +249 singleton train). **Test propuesto:** exigir intervalo por filas y preservar grupos, reproducibilidad y distribución de clases con tolerancia explícita; cuando restricciones sean incompatibles, rechazo/diagnóstico explícito, no exportación silenciosa. Otro probe con grupo4001 sobre4400 produjo97.727% train: ahí respetar simultáneamente grupos y rango train es imposible y debe diagnosticarse.

## Evidencia independiente adicional

- Refit independiente **en memoria**, sin helpers de modelado del autor: medianas NumPy train, estandarización poblacional, OHE construido desde categorías train; LinearRegression/LogisticRegression básicos nuevos. MAE/RMSE/R² calculados con fórmulas; accuracy/precision/recall/F1/confusión y AUC por rangos (SciPy). Todos los valores de los cuatro modelos en validation/test coinciden (rtol1e-7/atol1e-8).
- IDs completos/disjuntos; originales32 preservados en cleaned por row_id; comparación de **tuplas originales completas**, no solo fingerprints, confirma overlap0 dentro de cada tarea. agent/company no se convierten a magnitudes. No se exige separar la población común entre tareas distintas.
- Perfiles X repetidos se diagnostican, **no son duplicados confirmados**: clasificación2955/2933/843, regresión2040/2084/603 (train-val/train-test/val-test). Agrupar filas originales no demuestra independencia de cliente/grupo real; limitación reconocida.
- Fixture propia de400 reservas (sin fixture del autor),95/5 clases, children faltante y hotel faltante. `run_task` antes/después de sustituir **solo holdouts** por lead_time=1e12 y hotel=UNSEEN: medianas, medias scaler y vocabulario OHE **idénticos**; desconocidas procesadas; severe activa `class_weight=balanced`. Directorios del fixture eliminados por TemporaryDirectory. Sin ajuste de umbrales outlier sobre holdouts: se retienen extremos finitos por política, no hay winsorización aprendida que auditar.
- Limpieza: negativo ADR1, cero huéspedes180, cero noches715, negativos conteos0, target faltante0; unión826 por solapamientos. ADR cero y extremos positivos retenidos con justificación, no confundidos automáticamente con errores. Ausencia se estudia por hotel/resultado/segmento y no se declara demostrado MCAR/MAR/MNAR. La guía3.2 pide tratamiento razonado: aquí predomina retención explícita de colas y eliminación de inválidos, no transformación/winsorización de la cola positiva; conviene explicitar esta elección al evaluar el entregable académico.

## Comandos ejecutados y resultados observados

Desde raíz salvo `cd` explícito; todos en foreground:

- `cd preprocesamiento && uv run pytest` → EXIT0,14 passed, Python3.14.6 (3.81s).
- `cd preprocesamiento && uv run ruff check .` → EXIT0, All checks passed.
- `cd preprocesamiento && uv run ruff format --check .` → EXIT0,6 files already formatted.
- `cd preprocesamiento && uv run ty check` → EXIT0, All checks passed.
- `cd preprocesamiento && uv run python -m src.main --help` → EXIT0, cuatro comandos y cuatro rutas. Checks anteriores se hicieron antes del aviso posterior de cambio tipográfico; no se repitieron innecesariamente.
- `scratch=$(mktemp -d /tmp/hotel-review-all-XXXXXX) && printf 'scratch=%s\n' "$scratch" && cd preprocesamiento && test -f data/raw/hotel_bookings.csv && sha256sum data/raw/hotel_bookings.csv && uv run python -m src.main --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures" all && sha256sum data/raw/hotel_bookings.csv` → **EXIT0**, scratch **/tmp/hotel-review-all-9yxHNo**. stdout: `Validated original dataset: 119390 rows, 32 columns`; `EDA written to /tmp/hotel-review-all-9yxHNo/outputs/eda.json; SVG figures: /tmp/hotel-review-all-9yxHNo/figures`; `Phase 2: 118564 retained rows; results: /tmp/hotel-review-all-9yxHNo/outputs/phase2.json`. Hash raw antes/después idéntico (abajo). No descarga: raw existente, rama de adquisición omitida.
- `cd preprocesamiento && uv run python - < /tmp/hotel-review-all-9yxHNo/probes.py` → corrida final **EXIT0**. Compara todo JSON, recomputa estadísticas/χ², IDs/descartes, matrices/modelos/métricas y fixtures adversariales descritos. Script conservado con SHA256 `78b797e4ca27034849f56143f031a7a7d56eb022925a82a26c347ba755a385bf` para recheck. Warning inocuo al transformar NumPy con imputer ajustado con nombres, no en all.
- Heredocs `cd preprocesamiento && uv run python -` de inspección: lock/pins/hashes; comparación recursiva JSON; fila40679/líneas/hashes finales; reproducción600 filas H2 → EXIT0. Primer recomputador terminó EXIT1 por igualdad exacta de frecuencias relativas flotantes; primer probes.py terminó EXIT1 por igualdad estructural exacta EDA. Diagnóstico: solo redondeo de sumas; corregidas las comparaciones del **probe**, no código/outputs. Recomputador final tolerante pasa: phase2/cleaning exactamente iguales; EDA seis adr_mean por bins difieren≤4.27e-14 (tolerancia1e-12). No se presenta aquello como defecto del pipeline.
- `git status --short` e inspecciones acotadas → árbol previamente sucio preservado. Una inspección inicial fue rechazada por mencionar filtro de archivo sensible; no se leyó contenido sensible. No se ejecutó sync en esta revisión (el coordinador comunica PASS), ni CI remota.

## Candidato y recheck limitado

SHA256 de archivos de lógica/tests al cierre (prefijo `preprocesamiento/`):

```text
588c2bd0ad21cd474c6bc53db780cb792f4913d40bf19748f38f5ba989c6491f src/cleaning.py
10893b41fa5deb25ff54afdc2a5d7041c79e52985b605fc58e9d77404289a7bb src/data.py
cda8897beb840deaf4a93fbb5c70573555f4f5e9736e3b4f1625531d9d0fcb43 src/eda.py
cd4d1acad05c8e2f5d5df328d076c7a1bba9e836c4d1dcd0d9dc4467e2190047 src/main.py
b56cb6ce1321a4f96358b6ac857e6022f0d02cefa20c7fc0b40490e2d54821dd src/modeling.py
4f92484a1ddc3cdb67fe6afa92876b6f8477c72162270d113659e6973dfdb1c6 tests/test_pipeline.py
335a147c0842cd918149af50d40b453e3a038d9e822efa6875b0108ef899d161 pyproject.toml
c6fe108e58c91a32933a75a11e65d3696ac71825046d1bede0887b0d18b4db0b uv.lock
ff5789edcc3ed542f975d33f80f53ff62aad48e758c9510645772a222db6461a requirements.txt
84647c8d9555330477bd2a49de963bafaac30e4516994babbf5baec341b931b9 ../.github/workflows/ci.yml
```

Artefactos congelados, hashes iniciales/finales idénticos:

```text
a97ff27e670c7b21c24af0f571e9ec8b1f7a3f69104a4f27f0bced68ce7287af outputs/eda.json
40a16e7905c21a21d9a81e38e4d052ccaddf81d8949c57826b57099fbddc1eda outputs/phase2.json
1eedc3f00244e52436b4c132a96b0f53cd0fb5223da48bf92469be2d7593aea2 outputs/source.json
fc8db58f0bf0158050cc858ecb251faeee1da0af5bfbdf9cf70eba8f4f2c2bae outputs/cleaning.json
7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06 data/raw/hotel_bookings.csv
```

Cambio concurrente autorizado solo plotting: eda.py comenzó con hash `744c12a21322eb551bc47fc4834ea1cb52ad03e76d7dea7093f0dd635c797657`; hash final arriba. Recheck de esa diferencia limitado a fuente/tamaños/layout; SVG pueden diferir y aquí no se certifica revisión visual final. No repetir all para tipografía. Para H1/H2, recheck posterior de política y tests requiere implementación autorizada por coordinador; esta revisión no la realiza.

Scratch retenido: **/tmp/hotel-review-all-9yxHNo** (salidas all y probes.py); coordinador puede retirarlo después. Fixture temporal `/tmp/gphxzMQG/gajOtEJs/hotel-review-fixtures-qz8tcj34` eliminado automáticamente. No scratch/caché nuevo visible en git status; caches normales pytest/ruff autorizados. Único archivo entregado por esta revisión: este reporte.

## Cierre limitado H1/H2 y regresión directa de multiplicidad

**Conclusión vigente: H1 y H2 cerrados; sin bloqueador reproducido en el alcance autorizado.** No se realizó otra auditoría completa ni se modificaron código, JSON, informe o dependencias. S5/S7 pasan a **met** por la evidencia siguiente; el resto de la auditoría histórica conserva su alcance y limitaciones. Revisión genérica independiente, no veredicto native gentle-review.

### Candidato verificado

Hashes al inicio y final idénticos, coincidentes con los suministrados:

```text
c8ea410c04500face5325cb26bb705386e9651a646975878b5963b66563c8169 src/modeling.py
99b5e3ffafbeb2540a4e6d245fc43ab33d31130cf146cbf2de1d94c40dbb0a6c outputs/phase2.json
4fe14405d91f95a869e96cae01dee1ff34734a20cd7284bfad881057474e2c58 outputs/source.json
f8f838c4a8c5081a744578b0b7ea6c3fdb663ee7bc7015c884b78c27c14933d0 tests/test_pipeline.py
7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06 data/raw/hotel_bookings.csv
```

### Pruebas y evidencia observadas

- **H1 PASS:** `src/modeling.py:26–46` conserva adults/children/babies en X y excluye total_guests; `:253–266` conserva total_guests en Parquet, sin imputarlo como predictor. Fila real40679 conserva adults=3, children=null, total_guests=null; está en población de clasificación, no de regresión (cancelada). Contrato de predictores comprobado directamente en JSON y seis Parquet: sin targets/leakage ni suma redundante en X, pero ambos derivados guardados. Fixture independiente en ambas tareas: train children=[0,1,1,2] → mediana1; holdouts con adults3/5, children=null y hotel desconocido conservan3/5 adultos e imputan1 niño. Transformación finita; lead_time=10¹² y categoría nueva en holdout no cambian medianas, media/escala ni vocabulario OHE entrenados. Nueva regresión del autor leída en `tests/test_pipeline.py:75–113`; el revisor no añadió tests de producción ni afirma RED histórico.
- **H2 PASS:** algoritmo ponderado por filas/clase, déficit normalizado y validación antes de fit/export (`src/modeling.py:87–153`). Fixture original600, ahora con **row_id únicos**, produce450/75/75 y prevalencia1/3 en cada partición. Misma semilla → mismos IDs/asignación; unión completa de IDs, sin solapamiento de IDs/huellas. Fixture4400 con grupo4001 rechaza tanto split como run_task: `Cannot satisfy train row fraction 70%-80% with intact original groups: largest group 4001, total rows 4400`; destino de exportación no creado. Regresiones leídas en `tests/test_pipeline.py:116–131,154–167`.
- **Multiplicidad PASS:** fixture independiente1080 filas/800 grupos (tamaños1/2/3/4), no fixture del autor. Regresión810/135/135, proporción de filas en grupos repetidos0.407407 en las tres. Clasificación810/136/134:0.404938/0.411765/0.417910, prevalencias a menos0.02 de0.5. Todos los holdouts contienen grupos repetidos completos, IDs preservados,0overlap y reproducción exacta. Regresión específica del autor leída en `tests/test_pipeline.py:135–151`.
- **Artefactos reales PASS:** clasificación88923/14821/14820, regresión55791/9299/9298; train0.75 exacto, validation/test≈0.125. Prevalencias clasificación0.372592018/0.372579448/0.372604588. Se recalcularon **todos** los histogramas de tamaños de grupo y campos asociados de JSON desde Parquet, sin usar group_size_summary; coinciden. Fracción de filas en grupos repetidos CL0.337854098/0.337966399/0.337921727; REG0.223458981/0.223464889/0.223488922. No concentración artificial de repetidos únicamente en train. Poblaciones completas, IDs disjuntos,0huellas cruzadas y splits recalculados idénticos a almacenados.
- **Cuatro modelos PASS:** refit independiente en memoria con medianas/estandarización NumPy y OHE construido desde train; modelos básicos nuevos; métricas calculadas por fórmulas propias, AUC por rangos. Coinciden **todas** las métricas validation/test de DummyClassifier, LogisticRegression, DummyRegressor y LinearRegression (rtol1e-7/atol1e-8). Nuevas cifras test: logística accuracy0.7442645074/F10.5697093551/AUC0.7746932546; dummy accuracy0.6273954116/AUC0.5. Lineal MAE22.5296267019/RMSE31.3703220067/R²0.5919468302; dummy MAE37.0703732366/RMSE49.1095407220/R²−0.0000242873. Son las métricas del nuevo split, no las históricas. Sin selección por rendimiento en el código revisado.

### Comandos de este recheck

1. `scratch=$(mktemp -d /tmp/hotel-review-recheck-XXXXXX) && printf 'scratch=%s\n' "$scratch" && cd preprocesamiento && test -f data/raw/hotel_bookings.csv && sha256sum src/modeling.py outputs/phase2.json outputs/source.json tests/test_pipeline.py data/raw/hotel_bookings.csv && uv run python -m src.main --raw-dir data/raw --processed-dir "$scratch/processed" --output-dir "$scratch/outputs" --figures-dir "$scratch/figures" phase2 && sha256sum data/raw/hotel_bookings.csv` → **EXIT0**. Scratch `/tmp/hotel-review-recheck-Vkafni`. stdout: `Validated original dataset: 119390 rows, 32 columns`; `Phase 2: 118564 retained rows; results: /tmp/hotel-review-recheck-Vkafni/outputs/phase2.json`. Raw existente; sin descarga/red. Hash raw antes/después idéntico.
2. `cd preprocesamiento && uv run python - < /tmp/hotel-review-recheck-Vkafni/recheck.py` → **EXIT0 al primer intento**, todos los probes anteriores PASS. `phase2.json` aislado coincide **estructuralmente exacto** con el congelado, incluyendo splits, diagnósticos y métricas. Script SHA256 `c18814d988954d2e4bbe313b4a97477a6370b65e565bb66710741e6062f5cf30`.
3. `cd preprocesamiento && uv run python -` (heredoc de lectura de fracciones/prevalencias y hash de recheck.py) → EXIT0; cifras arriba.
4. `git status --short` → árbol previamente sucio preservado; solo este reporte fue editado por el revisor. Sin scratch/caché nuevo visible.

No se repitieron all/EDA, Ruff/format/ty ni la suite completa: el coordinador ya comunica PASS20tests4.28s para este candidato; se ejecutaron probes independientes específicos y phase2 aislado. **Limitación conservada:** asignación greedy no es solver exhaustivo de factibilidad; puede rechazar otras distribuciones, pero valida70–80% y holdouts no vacíos en lugar de exportar splits inválidos. No se abre otra auditoría ni ciclo de correcciones por esa limitación explícita.

Scratch nuevo retenido para recheck del coordinador: **/tmp/hotel-review-recheck-Vkafni** (phase2 aislado y recheck.py). No se creó destino must_not_exist. Scratch histórico `/tmp/hotel-review-all-9yxHNo` continúa documentado arriba. Sin cambios ni commits fuera del reporte.
