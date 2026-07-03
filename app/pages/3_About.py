import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import hero, inject  # noqa: E402
from utils import DISEASES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Health-OK | About", page_icon="⚕", layout="wide")
inject()

hero(
    "About the Diseases",
    "Clinical background, symptoms, risk factors, and prognosis for each condition "
    "covered by Health-OK. Educational reference only, not a substitute for "
    "professional medical advice.",
    eyebrow="About",
)

for key, info in DISEASES.items():
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            svg_path = os.path.join(ROOT, "assets", "images", f"{key}.svg")
            png_path = os.path.join(ROOT, "assets", "images", f"{key}.png")
            if os.path.exists(svg_path):
                with open(svg_path, encoding="utf-8") as fh:
                    svg = fh.read()
                st.markdown(f'<div class="hok-card-icon">{svg}</div>', unsafe_allow_html=True)
            elif os.path.exists(png_path):
                st.image(png_path, use_container_width=True)
        with col2:
            st.subheader(info["title"])
            st.write(info["description"])
            if "prevalence" in info:
                st.info(f"**Global prevalence:** {info['prevalence']}")
            symptom_col, risk_col = st.columns(2)
            with symptom_col:
                with st.expander("Common symptoms"):
                    for s in info["symptoms"]:
                        st.markdown(f"- {s}")
            with risk_col:
                with st.expander("Risk factors"):
                    for r in info["risk_factors"]:
                        st.markdown(f"- {r}")
            if "prevention" in info:
                with st.expander("Prevention & prognosis"):
                    st.markdown(info["prevention"])

st.divider()
st.caption(
    "Clinical background compiled from AHA, ADA, KDIGO, IDF, and peer-reviewed "
    "literature. Full citations available in `docs/DISEASES.md` in the project repo."
)
