# ===============================
# LBW Risk Prediction – Streamlit
# ===============================

import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="LBW Risk Prediction (XGBoost)",
    layout="wide"
)

st.title("Low Birth Weight (LBW) Risk Prediction")
st.caption("XGBoost model – full feature capture")

# -------------------------------
# Paths
# -------------------------------
MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/xgb_model.pkl"
FEATURES_PATH = f"{MODEL_DIR}/features.json"

# -------------------------------
# Load artifacts (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    if not st.secrets:
        pass  # avoids cloud warnings

    if not st.runtime.exists(MODEL_PATH):
        raise FileNotFoundError("❌ model/xgb_model.pkl not found")

    if not st.runtime.exists(FEATURES_PATH):
        raise FileNotFoundError("❌ model/features.json not found")

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    return model, features


try:
    model, FEATURES = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Model loaded successfully with {len(FEATURES)} features")

# -------------------------------
# Helper: type inference
# -------------------------------
def is_numeric_feature(name: str) -> bool:
    name_l = name.lower()
    keywords = [
        "age", "bmi", "count", "number", "days",
        "log", "pw", "inst", "lmp", "score", "gain"
    ]
    return any(k in name_l for k in keywords)

# -------------------------------
# Build UI (ALL features)
# -------------------------------
st.header("Beneficiary Information")

input_data = {}

with st.form("lbw_form"):
    for feature in FEATURES:

        if is_numeric_feature(feature):
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                step=1.0,
                format="%.2f"
            )
        else:
            input_data[feature] = st.selectbox(
                label=feature,
                options=["Yes", "No", "Unknown"]
            )

    submitted = st.form_submit_button("🔮 Predict LBW Risk")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    try:
        # Build DataFrame EXACTLY as model expects
        df = pd.DataFrame([input_data])

        # Convert Yes/No to numeric where needed
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map(
                    {"Yes": 1, "No": 0, "Unknown": np.nan}
                )

        # Enforce column order
        df = df[FEATURES]

        # XGBoost prediction
        prob = model.predict_proba(df)[0, 1]

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted LBW Risk Probability",
            value=f"{prob:.3f}"
        )

        if prob >= 0.5:
            st.error("⚠️ High risk of Low Birth Weight")
        else:
            st.success("✅ Lower predicted risk of Low Birth Weight")

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
