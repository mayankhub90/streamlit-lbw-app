import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="LBW Risk Prediction (XGBoost)",
    layout="wide"
)

st.title("🤰 Low Birth Weight (LBW) Risk Predictor")

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    return model, features

try:
    model, FEATURES = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model artifacts: {e}")
    st.stop()

# -----------------------------
# Reset state on reload
# -----------------------------
st.session_state.clear()

# -----------------------------
# Feature grouping (ONCE)
# -----------------------------
GROUPS = {
    "🧍 Background": [],
    "🚭 Tobacco & Alcohol": [],
    "🥗 Nutrition & Supplements": [],
    "🚽 Household & SES": [],
    "🏥 ANC & Program": [],
}

for f in FEATURES:
    fl = f.lower()
    if any(k in fl for k in ["age", "bmi", "child", "month"]):
        GROUPS["🧍 Background & Physiological"].append(f)
    elif any(k in fl for k in ["tobacco", "alcohol"]):
        GROUPS["🚭 Tobacco & Alcohol"].append(f)
    elif any(k in fl for k in ["ifa", "calcium", "food"]):
        GROUPS["🥗 Nutrition & Supplements"].append(f)
    elif any(k in fl for k in ["toilet", "water", "education"]):
        GROUPS["🚽 Household & SES"].append(f)
    else:
        GROUPS["🏥 ANC & Program"].append(f)

# -----------------------------
# Inputs
# -----------------------------
INPUTS = {}

st.subheader("Beneficiary details")
st.text_input(
    "Beneficiary Name (for record only, not used in model)",
    key="beneficiary_name"
)

st.divider()

# -----------------------------
# Render widgets (ONCE)
# -----------------------------
for group_name, group_feats in GROUPS.items():
    with st.expander(group_name, expanded=False):
        for f in group_feats:
            fl = f.lower()
            key = f"input_{f}"  # ✅ unique key

            if any(k in fl for k in ["yes", "no", "tobacco", "alcohol"]):
                INPUTS[f] = st.selectbox(
                    f,
                    ["No", "Yes"],
                    key=key
                )
            elif any(k in fl for k in ["category", "bucket"]):
                INPUTS[f] = st.number_input(
                    f,
                    min_value=0,
                    step=1,
                    key=key
                )
            else:
                INPUTS[f] = st.number_input(
                    f,
                    value=0.0,
                    key=key
                )

# -----------------------------
# Prepare dataframe
# -----------------------------
def build_input_df(inputs, feature_list):
    row = {}
    for f in feature_list:
        v = inputs.get(f)

        if isinstance(v, str):
            row[f] = 1 if v.lower() == "yes" else 0
        else:
            row[f] = float(v) if v is not None else np.nan

    return pd.DataFrame([row])

# -----------------------------
# Predict
# -----------------------------
st.divider()

if st.button("🔍 Predict LBW Risk", use_container_width=True):
    df = build_input_df(INPUTS, FEATURES)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing columns expected by model: {missing}")
        st.stop()

    try:
        prob = model.predict_proba(df)[0, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("✅ Prediction successful")
    st.metric("Predicted LBW Risk Probability", f"{prob:.2%}")

    if prob >= 0.5:
        st.error("⚠️ High risk of Low Birth Weight")
    else:
        st.success("🟢 Lower predicted risk")

st.caption("Model: XGBoost | Deployment: Streamlit")

