"""Entry point: page config, navigation and shared sidebar filters.

Run with ``streamlit run app.py`` from the ``dashboard`` folder.
"""

import streamlit as st

from hotel_dashboard import theme
from hotel_dashboard.data import apply_filters, load_bookings
from hotel_dashboard.filters import render_filters

st.set_page_config(
    page_title="Demanda hotelera",
    page_icon=":material/hotel:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

overview = st.Page(
    "pages/overview.py", title="Resumen", icon=":material/dashboard:", url_path="resumen", default=True
)
prediction = st.Page(
    "pages/prediction.py", title="Predicción", icon=":material/online_prediction:", url_path="prediccion"
)
segmentation = st.Page(
    "pages/segmentation.py", title="Segmentación", icon=":material/bubble_chart:", url_path="segmentacion"
)
models = st.Page("pages/models.py", title="Modelos", icon=":material/insights:", url_path="modelos")

page = st.navigation([overview, prediction, segmentation, models], position="top")

if page.title in {overview.title, segmentation.title}:
    bookings = load_bookings()
    spec = render_filters(bookings)
    st.session_state["selection"] = apply_filters(bookings, spec)
    st.session_state["filter_spec"] = spec
    with st.sidebar:
        st.divider()
        st.caption(
            "Datos: Hotel Booking Demand (Antonio, de Almeida y Nunes, 2019), CC BY 4.0. "
            "ADR en unidades monetarias de la fuente (u.m.)."
        )

page.run()
