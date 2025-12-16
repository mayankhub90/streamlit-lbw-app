# =========================
# LBW Streamlit App
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import os, json
import cloudpickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import date

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="LBW Risk Prediction", layout="wide")
st.title("Low Birth Weight (LBW) Risk Prediction")
st.caption("XGBoost-based clinical decision support tool")

# -------------------------
# RESET FORM
# -------------------------
if st.button("New Beneficiary / Reset Form"):
    st.session_state.clear()
    st.rerun()

# -------------------------
# PATHS
# -------------------------
MODEL_DIR = "model"
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
BOOSTER_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.csv")

# -------------------------
# LOAD ARTIFACTS
# -------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts(_version="v3"):  # cache buster
    with open(PREPROC_PATH, "rb") as f:
        preproc = cloudpickle.load(f)

    booster = xgb.Booster()
    booster.load_model(BOOSTER_PATH)

    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)

    bg = None
    if os.path.exists(BACKGROUND_PATH):
        bg = pd.read_csv(BACKGROUND_PATH)

    return preproc, booster, FEATURES, bg


preproc, booster, FEATURES, BACKGROUND = load_artifacts()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
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


# -------------------------
# INPUT UI
# -------------------------
inputs = {}

st.header("Beneficiary Details")
beneficiary_name = st.text_input("Beneficiary Name (optional)")

with st.expander("Pregnancy Dates", expanded=True):
    c1, c2 = st.columns(2)
    lmp_date = c1.date_input("LMP Date", value=None)
    reg_date = c2.date_input("Registration Date", value=None)

inputs["RegistrationBucket"] = compute_registration_bucket(lmp_date, reg_date)

with st.expander("Physiological & Demographic"):
    c1, c2 = st.columns(2)
    inputs["Beneficiary age"] = c1.number_input("Beneficiary age", 10, 60)
    inputs["Child order/parity"] = c2.number_input("Parity", 0, 10)
    inputs["Number of living child at now"] = c1.number_input("Living children", 0, 10)
    inputs["MonthConception"] = c2.number_input("Month of Conception", 1, 12)

    inputs["BMI_PW1_Prog"] = c1.number_input("BMI PW1")
    inputs["BMI_PW2_Prog"] = c2.number_input("BMI PW2")
    inputs["BMI_PW3_Prog"] = c1.number_input("BMI PW3")
    inputs["BMI_PW4_Prog"] = c2.number_input("BMI PW4")

with st.expander("Health Behaviours"):
    c1, c2 = st.columns(2)
    inputs["consume_tobacco"] = c1.selectbox("Consume Tobacco", ["Yes", "No"])
    inputs["Status of current chewing of tobacco"] = c2.selectbox("Chewing Tobacco", ["Yes", "No"])
    inputs["consume_alcohol"] = c1.selectbox("Consume Alcohol", ["Yes", "No"])

inputs["measured_HB_risk_bin"] = compute_hb_risk_bin(
    inputs["consume_tobacco"],
    inputs["Status of current chewing of tobacco"],
    inputs["consume_alcohol"]
)

with st.expander("Nutrition"):
    c1, c2 = st.columns(2)
    inputs["Food_Groups_Category"] = c1.selectbox("Food Groups Category", [1, 2, 3, 4, 5])

    ifa = c2.number_input("IFA tablets (last month)", min_value=0)
    calcium = c1.number_input("Calcium tablets (last month)", min_value=0)

    inputs["No. of IFA tablets received/procured in last one month_log1p"] = np.log1p(ifa)
    inputs["No. of calcium tablets consumed in last one month_log1p"] = np.log1p(calcium)

with st.expander("Household & SES"):
    c1, c2 = st.columns(2)
    assets = c1.number_input("Household Asset Score", min_value=0)
    inputs["Household_Assets_Score_log1p"] = np.log1p(assets)

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

with st.expander("Program & Services"):
    c1, c2 = st.columns(2)
    inputs["No of ANCs completed"] = c1.number_input("ANCs Completed", 0, 10)
    inputs["Service received during last ANC: TT Injection given"] = c2.selectbox("TT Injection", ["Yes", "No"])
    inputs["Registered for cash transfer scheme: JSY"] = c1.selectbox("Registered for JSY", ["Yes", "No"])
    inputs["Registered for cash transfer scheme: RAJHSRI"] = c2.selectbox("Registered for RAJHSRI", ["Yes", "No"])
    inputs["PMMVY-Number of installment received"] = c1.number_input("PMMVY installments", 0, 3)
    inputs["JSY-Number of installment received"] = c2.number_input("JSY installments", 0, 2)
    inputs["counselling_gap_days"] = c1.number_input("Counselling gap days", min_value=0)

# Always present engineered vars
inputs["ANCBucket"] = np.nan
inputs["LMPtoINST1"] = np.nan
inputs["LMPtoINST2"] = np.nan
inputs["LMPtoINST3"] = np.nan

# -------------------------
# BUILD DATAFRAME
# -------------------------
row = {col: inputs.get(col, np.nan) for col in FEATURES}
df = pd.DataFrame([row])

# -------------------------
# PREDICTION
# -------------------------
st.header("Prediction")

if st.button("Predict LBW Risk"):
    prob = predict(df)

    st.metric("LBW Probability", f"{prob:.3f}")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.6:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    # -------------------------
    # SHAP (RAW FEATURE LEVEL)
    # -------------------------
    X_trans = preproc.transform(df)
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_trans)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = shap_vals[0]

    try:
        trans_names = preproc.get_feature_names_out()
    except:
        trans_names = [f"f{i}" for i in range(len(shap_vals))]

    shap_long = pd.DataFrame({
        "feature_trans": trans_names,
        "shap_value": shap_vals
    })

    def map_raw(col):
        for raw in FEATURES:
            if col.startswith(raw):
                return raw
        return None

    shap_long["raw_feature"] = shap_long["feature_trans"].apply(map_raw)
    shap_agg = (
        shap_long.dropna()
        .groupby("raw_feature", as_index=False)["shap_value"]
        .sum()
        .sort_values("shap_value", key=abs, ascending=False)
    )

    st.subheader("Key factors influencing this prediction")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Factors increasing LBW risk")
        st.dataframe(shap_agg[shap_agg["shap_value"] > 0].head(5))

    with c2:
        st.write("Protective factors")
        st.dataframe(shap_agg[shap_agg["shap_value"] < 0].tail(5))

    fig, ax = plt.subplots()
    top = shap_agg.head(10)
    ax.barh(top["raw_feature"], top["shap_value"])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP contribution")
    st.pyplot(fig)
