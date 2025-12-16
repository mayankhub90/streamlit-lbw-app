import streamlit as st
import pandas as pd
import numpy as np
import json, os
import xgboost as xgb
import cloudpickle
import shap
import matplotlib.pyplot as plt
from datetime import date

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="LBW Risk Prediction", layout="wide")
st.title("Low Birth Weight (LBW) Risk Prediction")
st.caption("XGBoost-based decision support tool")

# ------------------------------------------------
# RESET FORM (NEW USER)
# ------------------------------------------------
if st.button("New Beneficiary / Reset Form"):
    st.session_state.clear()
    st.rerun()

# ------------------------------------------------
# PATHS
# ------------------------------------------------
MODEL_DIR = "model"
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
BOOSTER_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.csv")

# ------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        with open(PREPROC_PATH, "rb") as f:
            preproc = cloudpickle.load(f)
    except Exception as e:
        return None, None, None, None, f"Preprocessor load failed: {e}"

    try:
        booster = xgb.Booster()
        booster.load_model(BOOSTER_PATH)
    except Exception as e:
        return None, None, None, None, f"Model load failed: {e}"

    try:
        with open(FEATURES_PATH, "r") as f:
            FEATURES = json.load(f)
    except Exception as e:
        return None, None, None, None, f"features.json error: {e}"

    background = None
    if os.path.exists(BACKGROUND_PATH):
        background = pd.read_csv(BACKGROUND_PATH)

    return preproc, booster, FEATURES, background, None


preproc, booster, FEATURES, BACKGROUND, load_err = load_artifacts()
if load_err:
    st.error(load_err)
    st.stop()

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def compute_registration_bucket(lmp, reg):
    if lmp is None or reg is None:
        return np.nan
    days = (reg - lmp).days
    if days <= 84:
        return "Early"
    elif days <= 168:
        return "On-time"
    else:
        return "Late"


def compute_hb_risk_bin(tobacco, chew, alcohol):
    if tobacco == "Yes" or chew == "Yes" or alcohol == "Yes":
        return "High"
    return "Low"


def predict(df):
    X = preproc.transform(df)
    dmat = xgb.DMatrix(X.astype(np.float32))
    return float(booster.predict(dmat)[0])


def compute_shap(df):
    X = preproc.transform(df)
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return shap_vals[0]


# ------------------------------------------------
# INPUT UI
# ------------------------------------------------
st.header("Beneficiary Information")

beneficiary_name = st.text_input("Beneficiary Name (optional)")

with st.expander("Pregnancy Dates", expanded=True):
    c1, c2 = st.columns(2)
    lmp_date = c1.date_input("LMP Date", value=None)
    reg_date = c2.date_input("Registration Date", value=None)

registration_bucket = compute_registration_bucket(lmp_date, reg_date)
st.info(f"Registration Bucket: **{registration_bucket}**")

inputs = {}

# ------------------------------------------------
# PHYSIO & DEMOGRAPHIC
# ------------------------------------------------
with st.expander("Physiological & Demographic"):
    c1, c2 = st.columns(2)
    inputs["Beneficiary age"] = c1.number_input("Beneficiary age", 10, 60)
    inputs["Child order/parity"] = c2.number_input("Child order / Parity", 0, 10)
    inputs["Number of living child at now"] = c1.number_input("Number of living children", 0, 10)
    inputs["MonthConception"] = c2.number_input("Month of Conception", 1, 12)

    inputs["BMI_PW1_Prog"] = c1.number_input("BMI PW1")
    inputs["BMI_PW2_Prog"] = c2.number_input("BMI PW2")
    inputs["BMI_PW3_Prog"] = c1.number_input("BMI PW3")
    inputs["BMI_PW4_Prog"] = c2.number_input("BMI PW4")

# ------------------------------------------------
# HEALTH BEHAVIOURS
# ------------------------------------------------
with st.expander("Health Behaviours"):
    c1, c2 = st.columns(2)
    inputs["consume_tobacco"] = c1.selectbox("Consume Tobacco", ["Yes", "No"])
    inputs["Status of current chewing of tobacco"] = c2.selectbox("Chewing Tobacco", ["Yes", "No"])
    inputs["consume_alcohol"] = c1.selectbox("Consume Alcohol", ["Yes", "No"])

# DERIVED HB RISK
inputs["measured_HB_risk_bin"] = compute_hb_risk_bin(
    inputs["consume_tobacco"],
    inputs["Status of current chewing of tobacco"],
    inputs["consume_alcohol"],
)

# ------------------------------------------------
# NUTRITION (RAW → LOG)
# ------------------------------------------------
with st.expander("Nutrition"):
    c1, c2 = st.columns(2)
    inputs["Food_Groups_Category"] = c1.selectbox("Food Groups Category", [1, 2, 3, 4, 5])

    ifa_raw = c2.number_input("IFA tablets received (last month)", min_value=0)
    calcium_raw = c1.number_input("Calcium tablets consumed (last month)", min_value=0)

    inputs["No. of IFA tablets received/procured in last one month_log1p"] = np.log1p(ifa_raw)
    inputs["No. of calcium tablets consumed in last one month_log1p"] = np.log1p(calcium_raw)

# ------------------------------------------------
# HOUSEHOLD & SES (RAW → LOG)
# ------------------------------------------------
with st.expander("Household & Socioeconomic Status"):
    c1, c2 = st.columns(2)
    asset_raw = c1.number_input("Household Asset Score", min_value=0)
    inputs["Household_Assets_Score_log1p"] = np.log1p(asset_raw)

    inputs["toilet_type_clean"] = c2.selectbox("Toilet Available", ["Yes", "No"])
    inputs["water_source_clean"] = c1.selectbox("Safe Water Source", ["Yes", "No"])
    inputs["education_clean"] = c2.selectbox(
        "Education Level",
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

# ------------------------------------------------
# PROGRAM & SERVICES
# ------------------------------------------------
with st.expander("Programme & Health Services"):
    c1, c2 = st.columns(2)
    inputs["No of ANCs completed"] = c1.number_input("ANCs Completed", 0, 10)
    inputs["Service received during last ANC: TT Injection given"] = c2.selectbox("TT Injection Given", ["Yes", "No"])
    inputs["counselling_gap_days"] = c1.number_input("Counselling Gap Days", min_value=0)

    inputs["Registered for cash transfer scheme: JSY"] = c2.selectbox("Registered for JSY", ["Yes", "No"])
    inputs["Registered for cash transfer scheme: RAJHSRI"] = c1.selectbox("Registered for RAJHSRI", ["Yes", "No"])
    inputs["PMMVY-Number of installment received"] = c2.number_input("PMMVY Installments", 0, 3)
    inputs["JSY-Number of installment received"] = c1.number_input("JSY Installments", 0, 2)

# ------------------------------------------------
# DERIVED / ENGINEERED (ALWAYS PRESENT)
# ------------------------------------------------
inputs["RegistrationBucket"] = registration_bucket
inputs["ANCBucket"] = np.nan
inputs["LMPtoINST1"] = np.nan
inputs["LMPtoINST2"] = np.nan
inputs["LMPtoINST3"] = np.nan

# ------------------------------------------------
# BUILD FINAL DATAFRAME (CRITICAL)
# ------------------------------------------------
row = {col: inputs.get(col, np.nan) for col in FEATURES}
df = pd.DataFrame([row])

# ------------------------------------------------
# DEBUG (REMOVE AFTER STABLE)
# ------------------------------------------------
expected = set(preproc.feature_names_in_)
received = set(df.columns)
missing = expected - received

if missing:
    st.error(f"Missing columns expected by model: {missing}")
    st.stop()

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
st.header("Prediction")

if st.button("Predict LBW Risk"):
    prob = predict(df)

    st.subheader(f"Prediction for: {beneficiary_name or 'Beneficiary'}")
    st.metric("LBW Probability", f"{prob:.3f}")

    if prob < 0.3:
        st.success("Low Risk of LBW")
    elif prob < 0.6:
        st.warning("Moderate Risk of LBW")
    else:
        st.error("High Risk of LBW")

    # ---------------- SHAP ----------------
    st.subheader("Why this prediction?")
    shap_vals = compute_shap(df)

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP value": shap_vals
    }).sort_values("SHAP value", key=abs, ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        st.write("Top Risk Increasing Factors")
        st.dataframe(shap_df.head(5))

    with c2:
        st.write("Top Protective Factors")
        st.dataframe(shap_df.tail(5))

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"][:10], shap_df["SHAP value"][:10])
    ax.invert_yaxis()
    st.pyplot(fig)
