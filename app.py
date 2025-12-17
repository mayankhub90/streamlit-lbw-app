import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="LBW Risk Prediction (XGBoost)",
    layout="wide"
)

st.title("🤰 Low Birth Weight (LBW) Risk Predictor")
st.caption("XGBoost-based clinical risk scoring tool")

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.csv")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ model/xgb_model.pkl not found")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("❌ model/features.json not found")

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    bg = None
    if os.path.exists(BACKGROUND_PATH):
        bg = pd.read_csv(BACKGROUND_PATH)

    return model, features, bg


try:
    model, FEATURES, BACKGROUND = load_artifacts()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# -----------------------------
# Reset state on reload (new user)
# -----------------------------
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# -----------------------------
# Helper: safe numeric cast
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# -----------------------------
# UI: Beneficiary details
# -----------------------------
st.subheader("🧍 Beneficiary Information")

beneficiary_name = st.text_input(
    "Beneficiary Name (for record only – not used in model)"
)

st.divider()

# -----------------------------
# Feature groups (ONLY model features)
# -----------------------------
INPUTS = {}

with st.expander("🩺 Background & Physiological"):
    for f in FEATURES:
        if any(k in f.lower() for k in ["age", "bmi", "child", "month"]):
            INPUTS[f] = st.number_input(f, value=0.0)

with st.expander("🚭 Tobacco & Alcohol"):
    for f in FEATURES:
        if "tobacco" in f.lower() or "alcohol" in f.lower():
            INPUTS[f] = st.selectbox(f, ["Yes", "No"])

with st.expander("🥗 Nutrition & Supplements"):
    for f in FEATURES:
        if "ifa" in f.lower() or "calcium" in f.lower() or "food" in f.lower():
            INPUTS[f] = st.number_input(f, value=0.0)

with st.expander("🚽 Household & SES"):
    for f in FEATURES:
        if "toilet" in f.lower() or "water" in f.lower() or "education" in f.lower():
            INPUTS[f] = st.text_input(f)

with st.expander("🏥 ANC & Program"):
    for f in FEATURES:
        if "anc" in f.lower() or "jsy" in f.lower() or "pmmvy" in f.lower():
            INPUTS[f] = st.number_input(f, value=0.0)

# -----------------------------
# Prepare dataframe
# -----------------------------
def build_input_df(inputs, feature_list):
    row = {}
    for f in feature_list:
        val = inputs.get(f, np.nan)

        if isinstance(val, str):
            if val.lower() == "yes":
                row[f] = 1
            elif val.lower() == "no":
                row[f] = 0
            else:
                row[f] = val
        else:
            row[f] = safe_float(val)

    return pd.DataFrame([row])


# -----------------------------
# Predict
# -----------------------------
st.divider()

if st.button("🔍 Predict LBW Risk"):
    st.session_state.submitted = True

    df = build_input_df(INPUTS, FEATURES)

    # Check missing columns
    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing columns expected by model: {missing}")
        st.stop()

    try:
        proba = model.predict_proba(df)[0, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.success("✅ Prediction successful")

    st.metric(
        label="Predicted LBW Risk Probability",
        value=f"{proba:.2%}"
    )

    if proba >= 0.5:
        st.error("⚠️ High risk of Low Birth Weight")
    else:
        st.success("🟢 Lower predicted risk")

# -----------------------------
# Footer
# -----------------------------
st.caption("Model: XGBoost | Deployment: Streamlit")
