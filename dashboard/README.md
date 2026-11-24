# Dashboard interactivo · Demanda hotelera (Fase 4)

Aplicación Streamlit que integra los resultados de las fases 1 a 3 sobre *Hotel Booking Demand*:
indicadores de negocio filtrables, predicción en vivo de cancelación y tarifa (ADR), segmentación
K-Means sobre un mapa PCA y la evaluación comparativa de los modelos.

## Páginas

| Página | Contenido |
|---|---|
| **Resumen** | Filtros dinámicos (fecha de llegada, hotel, segmento, tipo de cliente, depósito, canal, anticipación, ADR, noches), KPI de negocio y de modelos, lectura rápida y gráficos de demanda, cancelación y tarifas. |
| **Predicción** | Formulario de una reserva nueva → probabilidad de cancelación (indicador con umbral 0,50), ADR estimada con rango entre árboles y segmento asignado. |
| **Segmentación** | Mapa PCA 2D/3D coloreado por segmento, tamaños, perfil medio (mapa de calor y tabla) y posición de la reserva simulada. |
| **Modelos** | Métricas recalculadas frente a referencias y a la Fase 3, importancia por permutación, observada vs. predicha, residuos, matriz de confusión con umbral ajustable, ROC, codo/silueta y piloto DBSCAN. |

## Estructura

```
dashboard/
├── app.py                  # entrada: navegación y filtros compartidos
├── pages/                  # overview, prediction, segmentation, models
├── hotel_dashboard/        # config, data (carga y filtros), filters (widgets),
│                           # kpis, charts (Plotly), modeling (inferencia), theme, ui
├── training/train.py       # reentrenamiento reproducible → artifacts/ y data/
├── artifacts/              # modelos joblib, metrics.json, predicciones de validación
├── data/bookings.parquet   # conjunto compacto para la app (118.564 reservas, ~1,2 MB)
├── tests/                  # pytest
├── .streamlit/config.toml  # tema claro
├── pyproject.toml, uv.lock
└── requirements.txt        # versiones exactas (uv export)
```

## Ejecutar en local

Con [uv](https://docs.astral.sh/uv/) (Python 3.13):

```bash
cd dashboard
uv sync
uv run streamlit run app.py
```

Con pip:

```bash
cd dashboard
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación solo necesita `data/` y `artifacts/`, que se versionan; no requiere el CSV original.

## Reentrenar

Requiere los Parquet de la Fase 2 (`preprocesamiento/data/processed/`, generados con
`preprocesamiento/src/main.py`; no se versionan):

```bash
cd dashboard
uv run python -m training.train --processed-dir ../preprocesamiento/data/processed
```

El script ajusta referencia, modelo lineal/logístico y Random Forest para ADR (solo reservas
mantenidas) y cancelación, K-Means (K = 6) con PCA y el piloto DBSCAN; escribe los modelos
comprimidos, `metrics.json`, las predicciones de validación y `data/bookings.parquet`.
La partición de prueba no se usa durante el entrenamiento.

Todas las configuraciones replican el cuaderno ejecutado de la Fase 3
([`fase_3/Fase_3_Hotel_Booking.ipynb`](../fase_3/Fase_3_Hotel_Booking.ipynb), copia local del
Colab):

| Modelo | Configuración (ganadora de `RandomizedSearchCV`, 3 candidatos × 3 pliegues por huella) |
|---|---|
| Random Forest ADR | 58 árboles, `max_depth=18`, `max_features=0.5`, `min_samples_leaf=6`, `min_samples_split=3` (CV `GroupKFold`, criterio RMSE) |
| Random Forest cancelación | 53 árboles, `max_depth=18`, `max_features="sqrt"`, `min_samples_leaf=8`, `min_samples_split=8`, sin `class_weight` (CV `StratifiedGroupKFold`, criterio AUC) |
| K-Means | K = 6, `k-means++`, `n_init=10`, `max_iter=500`, semillas 42/1042/2042 para la estabilidad (ARI), entrada `float32`, silueta en 2.500 filas |
| DBSCAN (piloto) | 5.000 filas de entrenamiento (`default_rng(5042)`), ε candidatos = cuantiles 0,85–0,975 de la distancia al 10.º vecino, `min_samples` ∈ {5, 10, 20}; seleccionado ε = 1,438558, `min_samples` = 20 |

El cuaderno no imprime `best_params_`; cada ganador se identificó reajustando los tres candidatos
registrados: solo uno reproduce las métricas de validación del cuaderno (y también es el primero
en la validación cruzada). El preprocesador es el del cuaderno (mediana + estandarización,
moda + _one-hot_ disperso) y la segmentación usa la partición de clustering de la Fase 2 (sin
estratificar), que `train.py` reconstruye a partir de `cleaned.parquet`. Con
**scikit-learn 1.7.2**, la versión del cuaderno y la fijada en este proyecto, el reentrenamiento
reproduce exactamente las métricas de validación, las importancias por permutación, los tamaños
de los seis segmentos y el piloto DBSCAN de la Fase 3 (con scikit-learn 1.9 el bosque de ADR
difiere en la tercera cifra decimal y K-Means converge a otra solución para K = 6).

Una vez fijados los modelos, la evaluación única en prueba carga los artefactos ya ajustados
(sin reentrenar ni cambiar el umbral) y escribe `artifacts/test_metrics.json`:

```bash
uv run python -m training.evaluate_test --processed-dir ../preprocesamiento/data/processed
```

## Pruebas y estilo

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Despliegue en Streamlit Community Cloud

1. Publicar el repositorio en GitHub (incluidos `dashboard/data` y `dashboard/artifacts`, ~7 MB).
2. En <https://share.streamlit.io> → *Create app* → elegir repositorio y rama, y como
   *Main file path* `dashboard/app.py`.
3. En *Advanced settings* seleccionar **Python 3.13** (las versiones fijadas en
   `requirements.txt` se resolvieron para 3.13).
4. *Deploy*. Community Cloud instala las dependencias del directorio de la app y aplica el tema de
   `.streamlit/config.toml`.

## Datos y licencia

N. Antonio, A. de Almeida y L. Nunes, «Hotel booking demand datasets», *Data in Brief*, vol. 22,
pp. 41–49, 2019. doi:[10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).
Versión de Kaggle `jessemostipak/hotel-booking-demand`, licencia
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `data/bookings.parquet` es una
versión derivada (limpieza de la Fase 2, columnas seleccionadas, segmento y coordenadas PCA
añadidos). La ADR se expresa en las unidades monetarias de la fuente (u.m.).
