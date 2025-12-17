import os
import streamlit as st

st.write("📂 Current working directory:", os.getcwd())
st.write("📁 Files in root:", os.listdir("."))

if os.path.exists("XGBoost Model"):
    st.write("📁 XGBoost Model contents:", os.listdir("XGBoost Model"))
    if os.path.exists("XGBoost Model/artifacts"):
        st.write(
            "📁 artifacts contents:",
            os.listdir("XGBoost Model/artifacts")
        )



# ============================================================
# Streamlit LBW Risk App – Stable XGBoost (Python 3.13 Safe)
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# App config
# ---------------------------
st.set_page_config(
    page_title="LBW Risk Prediction",
    layout="wide"
)

ARTIFACT_DIR = Path("XGBoost Model/artifacts")
MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
FEATURES_PATH = ARTIFACT_DIR / "features.json"
BACKGROUND_PATH = ARTIFACT_DIR / "background.csv"

# ---------------------------
# Load artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None, None, "xgb_model.pkl not found"

    if not FEATURES_PATH.exists():
        return None, None, None, "features.json not found"

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    background = None
    if BACKGROUND_PATH.exists():
        background = pd.read_csv(BACKGROUND_PATH)

    return model, features, background, None


model, FEATURES, BACKGROUND, load_err = load_artifacts()

if load_err:
    st.error(load_err)
    st.stop()

# ---------------------------
# Helper: clean input
# ---------------------------
def clean_value(v):
    if v in ("", None):
        return np.nan
    return v


# ---------------------------
# Sidebar: beneficiary info
# ---------------------------
st.sidebar.header("Beneficiary details")

beneficiary_name = st.sidebar.text_input(
    "Beneficiary name (not used in model)",
    value="",
    key="beneficiary_name"
)

st.sidebar.markdown("---")

# ---------------------------
# Main form
# ---------------------------
st.title("Low Birth Weight (LBW) Risk Prediction")
st.markdown("Enter beneficiary details to estimate LBW risk.")

with st.form("lbw_form", clear_on_submit=True):

    user_inputs = {}

    # --- Split UI into 3 columns ---
    col1, col2, col3 = st.columns(3)

    for i, feat in enumerate(FEATURES):

        # distribute fields
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        # Binary yes/no
        if feat.lower().startswith(("consume", "registered", "service received")):
            val = col.selectbox(
                feat,
                ["Yes", "No"],
                index=0
            )
            user_inputs[feat] = 1 if val == "Yes" else 0

        # Categorical with known buckets
        elif feat in ["Food_Groups_Category", "PMMVY-Number of installment received", "JSY-Number of installment received"]:
            user_inputs[feat] = col.selectbox(feat, [0, 1, 2, 3, 4, 5])

        # Generic numeric
        else:
            user_inputs[feat] = col.number_input(
                feat,
                value=0.0,
                step=1.0
            )

    submitted = st.form_submit_button("Predict LBW Risk")

# ---------------------------
# Prediction
# ---------------------------
if submitted:

    # Build DataFrame in correct order
    df = pd.DataFrame([{f: clean_value(user_inputs.get(f)) for f in FEATURES}])

    # Force numeric (critical)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---------------------------
    # 🚨 CRITICAL FIX: DMatrix prediction
    # ---------------------------
    dmatrix = xgb.DMatrix(df, feature_names=FEATURES)

    prob = float(model.get_booster().predict(dmatrix)[0])

    st.success(f"### Predicted LBW risk: **{prob:.2%}**")

    # ---------------------------
    # SHAP explanation
    # ---------------------------
    st.markdown("### Factors influencing this prediction")

    try:
        explainer = shap.TreeExplainer(model.get_booster())

        shap_vals = explainer.shap_values(dmatrix)

        shap_df = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP value": shap_vals[0]
        }).sort_values(
            "SHAP value",
            key=lambda x: np.abs(x),
            ascending=False
        )

        st.dataframe(shap_df.head(15), use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        shap_df.head(10).iloc[::-1].plot(
            kind="barh",
            x="Feature",
            y="SHAP value",
            ax=ax,
            legend=False
        )
        plt.title("Top SHAP drivers")
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation not available")
        st.text(str(e))

