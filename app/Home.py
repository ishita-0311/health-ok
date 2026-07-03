import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from style import hero, inject  # noqa: E402
from utils import DISEASES  # noqa: E402

st.set_page_config(page_title="Health-OK", page_icon="⚕", layout="wide")
inject()

hero(
    "Health-OK",
    "A machine learning and deep learning powered risk-assessment system covering "
    "5 diseases. Explore the data on the <b>Dashboard</b>, run your own numbers on the "
    "<b>Prediction</b> page, or learn the clinical background on <b>About</b>.",
    stats=["5 diseases", "12 trained models", "ML + Deep Learning", "GPU-accelerated training"],
    eyebrow="Multi-disease prediction system",
)

st.warning(
    "This tool is for educational and demonstration purposes only and does **not** "
    "provide a medical diagnosis. Always consult a qualified healthcare professional."
)

st.header("Supported Diseases")
ASSETS_ROOT = os.path.dirname(__file__)
cols = st.columns(len(DISEASES))
for col, (key, info) in zip(cols, DISEASES.items()):
    with col:
        svg_path = os.path.join(ASSETS_ROOT, "..", "assets", "images", f"{key}.svg")
        svg = ""
        if os.path.exists(svg_path):
            with open(svg_path, encoding="utf-8") as fh:
                svg = fh.read()

        st.markdown(
            f"""
            <div class="hok-card">
                <div class="hok-card-icon">{svg}</div>
                <h3>{info['title']}</h3>
                <p class="hok-desc">{info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
nav_cols = st.columns(3)
nav_items = [
    ("D", "Dashboard", "Explore the underlying datasets: summary stats, class balance, distributions, correlations, and feature importance."),
    ("P", "Prediction", "Pick a disease, enter clinical values, and get a model-driven risk estimate."),
    ("A", "About", "Learn the clinical background, symptoms, risk factors, and prognosis for each condition."),
]
for col, (letter, title, desc) in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f"""
            <div class="hok-card" style="min-height:160px;">
                <div class="hok-nav-icon">{letter}</div>
                <h3>{title}</h3>
                <p class="hok-desc">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
