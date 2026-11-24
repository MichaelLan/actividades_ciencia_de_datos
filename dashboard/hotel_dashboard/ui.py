"""Small Streamlit helpers shared by the pages."""

import plotly.graph_objects as go
import streamlit as st

from hotel_dashboard import modeling, theme


@st.cache_resource(show_spinner="Cargando modelos…")
def get_models() -> modeling.Models:
    """Load the trained pipelines once per server process."""
    return modeling.load_models()


def header(eyebrow: str, title: str, lead: str) -> None:
    """Page title block with a short narrative lead."""
    st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(lead)


def plot(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the shared template and toolbar config."""
    st.plotly_chart(fig, width="stretch", theme=None, config=theme.PLOTLY_CONFIG, key=key)


def segment_chip(color: str) -> str:
    """Inline HTML colour swatch for a segment name."""
    return f'<span class="segment-chip" style="background:{color}"></span>'
