import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import PLOT_ACCENT, PLOT_ACCENT2, apply_mpl_theme, hero, inject  # noqa: E402
from utils import DATA_DIR, DISEASES, load_model_and_scaler, load_schema  # noqa: E402

st.set_page_config(page_title="Health-OK | Dashboard", page_icon="⚕", layout="wide")
inject()
apply_mpl_theme()

hero(
    "Exploratory Data Analysis",
    "Summary statistics, class balance, feature distributions, correlations, and "
    "feature importance for every dataset behind Health-OK.",
    eyebrow="Dashboard",
)

disease_key = st.selectbox(
    "Choose a disease dataset",
    options=list(DISEASES.keys()),
    format_func=lambda k: DISEASES[k]["title"],
)
info = DISEASES[disease_key]
csv_path = os.path.join(DATA_DIR, info["csv"])

if not os.path.exists(csv_path):
    st.error(f"Dataset not found at {csv_path}. Run `python train/train_all.py` first.")
    st.stop()

df = pd.read_csv(csv_path)
target_col = info["target"]

m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])
if target_col in df.columns:
    positive_rate = pd.to_numeric(df[target_col], errors="coerce").mean()
    m3.metric("Positive class rate", f"{positive_rate * 100:.1f}%" if pd.notna(positive_rate) else "n/a")

with st.expander("Raw sample", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("Summary statistics")
st.dataframe(df.describe(), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class balance")
    if target_col in df.columns:
        fig, ax = plt.subplots()
        df[target_col].value_counts().sort_index().plot(kind="bar", ax=ax, color=[PLOT_ACCENT, PLOT_ACCENT2])
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No target column found for this dataset.")

with col2:
    st.subheader("Feature distribution")
    numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    feat = st.selectbox("Select feature", numeric_cols, key=f"{disease_key}_feat")
    fig, ax = plt.subplots()
    sns.histplot(df[feat], kde=True, ax=ax, color=PLOT_ACCENT)
    st.pyplot(fig)

st.subheader("Correlation heatmap")
numeric_df = df.select_dtypes(include="number")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax, annot=numeric_df.shape[1] <= 15, fmt=".2f")
st.pyplot(fig)

st.subheader("Feature importance (best model)")
try:
    schema = load_schema(disease_key)
    model, _ = load_model_and_scaler(disease_key, schema)
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        imp_df = pd.DataFrame({
            "feature": schema["feature_order"],
            "importance": importances,
        }).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(imp_df))))
        ax.barh(imp_df["feature"], imp_df["importance"], color=PLOT_ACCENT)
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.info(f"The winning model for {info['title']} ({schema['winner']}) "
                 "doesn't expose feature importances (e.g. it's a neural net or "
                 "linear/kernel model).")
except FileNotFoundError:
    st.info("Model not trained yet. Run `python train/train_all.py` first.")
