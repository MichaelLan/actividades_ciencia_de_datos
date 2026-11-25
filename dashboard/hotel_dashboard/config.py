"""Paths, feature definitions and Spanish display labels shared by training and app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bookings.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
ADR_MODEL_PATH = ARTIFACTS_DIR / "adr_model.joblib"
CANCEL_MODEL_PATH = ARTIFACTS_DIR / "cancel_model.joblib"
SEGMENTATION_PATH = ARTIFACTS_DIR / "segmentation.joblib"
PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.parquet"

SEED = 42
THRESHOLD = 0.5

NUMERIC_FEATURES = [
    "lead_time",
    "adults",
    "children",
    "babies",
    "total_nights",
    "previous_cancellations",
    "previous_bookings_not_canceled",
]
CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "customer_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLUSTER_FEATURES = NUMERIC_FEATURES

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_ES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]

CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "hotel": {
        "City Hotel": "Hotel urbano (Lisboa)",
        "Resort Hotel": "Hotel vacacional (Algarve)",
    },
    "arrival_date_month": dict(zip(MONTHS, MONTHS_ES, strict=True)),
    "market_segment": {
        "Online TA": "Agencia online",
        "Offline TA/TO": "Agencia o turoperador offline",
        "Groups": "Grupos",
        "Direct": "Directo",
        "Corporate": "Corporativo",
        "Complementary": "Cortesía",
        "Aviation": "Aviación",
        "Undefined": "Sin definir",
    },
    "distribution_channel": {
        "TA/TO": "Agencia o turoperador",
        "Direct": "Directo",
        "Corporate": "Corporativo",
        "GDS": "GDS",
        "Undefined": "Sin definir",
    },
    "customer_type": {
        "Transient": "Individual",
        "Transient-Party": "Individual vinculado",
        "Contract": "Contrato",
        "Group": "Grupo",
    },
    "deposit_type": {
        "No Deposit": "Sin depósito",
        "Non Refund": "No reembolsable",
        "Refundable": "Reembolsable",
    },
    "meal": {
        "BB": "Alojamiento y desayuno",
        "HB": "Media pensión",
        "FB": "Pensión completa",
        "SC": "Solo alojamiento",
        "Undefined": "Sin definir",
    },
    "is_repeated_guest": {"0": "No", "1": "Sí"},
    "reserved_room_type": {code: f"Tipo {code}" for code in ["A", "B", "C", "D", "E", "F", "G", "H", "L"]},
}

FEATURE_LABELS = {
    "lead_time": "Anticipación (días)",
    "adults": "Adultos",
    "children": "Niños",
    "babies": "Bebés",
    "total_nights": "Noches",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas mantenidas",
    "hotel": "Hotel",
    "arrival_date_month": "Mes de llegada",
    "meal": "Régimen",
    "market_segment": "Segmento de mercado",
    "distribution_channel": "Canal de distribución",
    "is_repeated_guest": "Cliente repetido",
    "reserved_room_type": "Habitación reservada",
    "customer_type": "Tipo de cliente",
    "deposit_type": "Tipo de depósito",
    "adr": "ADR",
}

SEGMENTS: dict[str, dict[str, str]] = {
    "short_lead": {
        "name": "Planificación a corto plazo",
        "tagline": "Parejas que reservan un mes antes",
        "description": "Parejas que reservan cerca de un mes antes y se quedan dos o tres noches. Es el núcleo del volumen.",
    },
    "long_lead": {
        "name": "Planificación anticipada",
        "tagline": "Reservan unos siete meses antes",
        "description": "Reservas hechas con unos siete meses de antelación y estancias más largas; concentran más cancelaciones.",
    },
    "children": {
        "name": "Familias con niños",
        "tagline": "Dos adultos con niños",
        "description": "Dos adultos con al menos un niño; estancias algo más largas y tarifas más altas por habitación.",
    },
    "baby": {
        "name": "Familias con bebé",
        "tagline": "Viajan con bebé",
        "description": "Reservas con bebé; grupo pequeño con estancias de unas cuatro noches.",
    },
    "loyal": {
        "name": "Clientes fieles",
        "tagline": "Repiten y casi no cancelan",
        "description": "Huéspedes con muchas reservas previas mantenidas; reservan con poca antelación y casi no cancelan.",
    },
    "serial_cancellers": {
        "name": "Historial de cancelaciones",
        "tagline": "Nicho de riesgo extremo",
        "description": "Reservas con decenas de cancelaciones previas; grupo muy pequeño pero de riesgo extremo.",
    },
}
SEGMENT_ORDER = list(SEGMENTS)

DATASET_CITATION = (
    "N. Antonio, A. de Almeida y L. Nunes, «Hotel booking demand datasets», "
    "Data in Brief, vol. 22, pp. 41–49, 2019. doi:10.1016/j.dib.2018.11.126. "
    "Distribuido en Kaggle (jessemostipak/hotel-booking-demand) bajo licencia CC BY 4.0."
)


def label(column: str, value: object) -> str:
    """Return the Spanish display label for a raw category value."""
    text = str(value)
    return CATEGORY_LABELS.get(column, {}).get(text, text)
