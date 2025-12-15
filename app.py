import streamlit as st
import pandas as pd
import numpy as np
import json, os, joblib
import xgboost as xgb
import shap
from datetime import datetime

# ---------------- MODEL FILE PATHS ----------------
MODEL_DIR = "model"
PREPROC_CLOUD = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
BOOSTER_JSON = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
FEATURES_T_PATH = os.path.join(MODEL_DIR, "features_transformed.json")
BACKGROUND_CSV = os.path.join(MODEL_DIR, "background.csv")


# ---------------- ARTIFACT LOADER ----------------
@st.cache_resource
def load_artifacts():
    """Load preprocessor, model, features, background dataset.
       ALWAYS RETURNS 6 VALUES: (preproc, booster, features, features_t, background, err)"""

    # --- Load preprocessor ---
    try:
        import cloudpickle
        with open(PREPROC_CLOUD, "rb") as f:
            preproc = cloudpickle.load(f)
    except Exception as e:
        return None, None, None, None, None, f"Failed loading preprocessor: {e}"

    # --- Load booster ---
    booster = None
    try:
        if os.path.exists(BOOSTER_JSON):
            booster = xgb.Booster()
            booster.load_model(BOOSTER_JSON)
    except Exception:
        booster = None

    # --- Load features.json ---
    try:
        with open(FEATURES_PATH, "r") as f:
            FEATURES = json.load(f)
    except Exception as e:
        return None, None, None, None, None, f"Failed loading features.json: {e}"

    # --- Load transformed feature names ---
    try:
        with open(FEATURES_T_PATH, "r") as f:
            FEATURES_T = json.load(f)
    except:
        FEATURES_T = None

    # --- Load background dataset ---
    BACKGROUND = None
    try:
        if os.path.exists(BACKGROUND_CSV):
            BACKGROUND = pd.read_csv(BACKGROUND_CSV)
    except:
        BACKGROUND = None

    return preproc, booster, FEATURES, FEATURES_T, BACKGROUND, None


# Load artifacts
preproc, booster, FEATURES, FEATURES_T, BACKGROUND, load_err = load_artifacts()
if load_err:
    st.error(load_err)
    st.stop()

# --------------- Prediction Function ---------------
def predict_proba(df_raw):
    X = preproc.transform(df_raw)
    X_float = X.astype(np.float32)

    if booster is not None:
        dmat = xgb.DMatrix(X_float)
        proba = booster.predict(dmat)
        if proba.ndim > 1:
            return float(proba[:, 1][0])
        return float(proba[0])
    else:
        return float(model.predict_proba(X_float)[:, 1][0])


# --------------- SHAP Function ----------------
def compute_shap(df_raw):
    X = preproc.transform(df_raw)

    if booster is not None:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 for binary
        return shap_vals[0]  # row 0

    # fallback – slow kernel SHAP
    if BACKGROUND is None:
        return None
    return None


# --------------- UI ---------------
st.title("LBW Risk Prediction App")
st.write("Fill the details below to estimate the LBW risk and see contributing factors.")

# ---------------- Beneficiary Name ----------------
benef_name = st.text_input("Beneficiary Name (optional)", "")

# ---------------- INPUT FORM ----------------
st.header("Beneficiary Details")

# SECTION: Dates
with st.expander("Registration & LMP Dates"):
    col1, col2 = st.columns(2)
    reg_date = col1.date_input("Registration Date")
    lmp_date = col2.date_input("LMP Date")

# Compute RegistrationBucket
try:
    days = (pd.to_datetime(reg_date) - pd.to_datetime(lmp_date)).days
    if days <= 84:
        reg_bucket = "Early"
    elif days <= 168:
        reg_bucket = "On-time"
    else:
        reg_bucket = "Late"
except:
    reg_bucket = ""

# ---------------- FORM CATEGORIES ----------------

st.subheader("Background Variables")
with st.expander("Background Variables"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Beneficiary age", min_value=10, max_value=60, step=1)
    hb = col2.selectbox("measured_HB_risk_bin", ["Low", "Medium", "High"])

    parity = col1.number_input("Child order/parity", 0, 10, 1)
    living_children = col2.number_input("Number of living children", 0, 10, 0)
    month_conception = col1.number_input("MonthConception", 1, 12, 1)

st.subheader("Household Environment")
with st.expander("Household Environment"):
    col1, col2 = st.columns(2)
    toilet_type = col1.selectbox("toilet_type_clean", ["Yes", "No"])
    water_source = col2.selectbox("water_source_clean", ["Yes", "No"])
    education = col1.selectbox(
        "education_clean",
        [
            "Illiterate",
            "Primary",
            "Upper Primary",
            "Secondary",
            "Senior Secondary",
            "Graduate",
            "Graduate and Above",
        ],
    )

st.subheader("Health Behaviors")
with st.expander("Health Behaviors"):
    col1, col2 = st.columns(2)
    consume_tobacco = col1.selectbox("consume_tobacco", ["Yes", "No"])
    current_chew = col2.selectbox("Status of current chewing of tobacco", ["Yes", "No"])
    consume_alcohol = col1.selectbox("consume_alcohol", ["Yes", "No"])

st.subheader("Nutrition")
with st.expander("Nutrition"):
    col1, col2 = st.columns(2)
    food_group = col1.selectbox("Food_Groups_Category", [1, 2, 3, 4, 5])
    social_media = col2.selectbox("Social_Media_Category", ["Yes", "No"])

st.subheader("Cash Transfer Schemes")
with st.expander("Cash Transfer Schemes"):
    col1, col2 = st.columns(2)
    jsyn = col1.selectbox("Registered for cash transfer scheme: JSY", ["Yes", "No"])
    raj = col2.selectbox("Registered for cash transfer scheme: RAJHSRI", ["Yes", "No"])

    pmmvy_inst = col1.number_input("PMMVY-Number of installment received", 0, 3, 0)
    jsy_inst = col2.number_input("JSY-Number of installment received", 0, 2, 0)


# --------------- Build dataframe for prediction ----------------
def build_input_df():
    mapping = {
        "Yes": 1,
        "No": 0
    }

    df = pd.DataFrame([{
        "Beneficiary age": age,
        "measured_HB_risk_bin": hb,
        "Child order/parity": parity,
        "Number of living child at now": living_children,
        "MonthConception": month_conception,
        "consume_tobacco": mapping.get(consume_tobacco, consume_tobacco),
        "Status of current chewing of tobacco": mapping.get(current_chew, current_chew),
        "consume_alcohol": mapping.get(consume_alcohol, consume_alcohol),
        "RegistrationBucket": reg_bucket,
        "ANCBucket": "",

        "Food_Groups_Category": food_group,
        "Social_Media_Category": mapping.get(social_media, social_media),
        "toilet_type_clean": mapping.get(toilet_type, toilet_type),
        "water_source_clean": mapping.get(water_source, water_source),
        "education_clean": education,
        "Registered for cash transfer scheme: JSY": mapping.get(jsyn, jsyn),
        "Registered for cash transfer scheme: RAJHSRI": mapping.get(raj, raj),
        "PMMVY-Number of installment received": pmmvy_inst,
        "JSY-Number of installment received": jsy_inst,
    }])

    return df


# ---------------- RUN PREDICTION ----------------
st.header("Prediction")

if st.button("Predict LBW Risk"):
    df_input = build_input_df()

    try:
        prob = predict_proba(df_input)
        st.success(f"Predicted LBW Probability: **{prob:.2f}**")

        if prob < 0.33:
            st.info("Risk Category: **Low**")
        elif prob < 0.66:
            st.warning("Risk Category: **Medium**")
        else:
            st.error("Risk Category: **High**")

        # SHAP
        shap_vals = compute_shap(df_input)
        if shap_vals is not None:
            st.subheader("Top SHAP Contributors")

            top_pos = np.argsort(shap_vals)[-5:]
            top_neg = np.argsort(shap_vals)[:5]

            st.write("**Top Risk-Increasing Factors**")
            for idx in reversed(top_pos):
                st.write(f"{FEATURES[idx]}: +{shap_vals[idx]:.4f}")

            st.write("---")
            st.write("**Top Protective Factors**")
            for idx in top_neg:
                st.write(f"{FEATURES[idx]}: {shap_vals[idx]:.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

