import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import shap
from datetime import date

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="LBW Risk Assessment", layout="wide")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/xgb_model.pkl")
    features = json.load(open("model/features.json"))
    background = pd.read_csv("model/background.csv")
    return model, features, background

model, FEATURES, BACKGROUND = load_artifacts()

# -----------------------------
# Categorical features used in training
# -----------------------------
CATEGORICAL_FEATURES = [
    "Child order/parity",
    "Food_Groups_Category",
    "Social_Media_Category",
    "RegistrationBucket",
    "ANCBucket",
]

# -----------------------------
# Helper functions
# -----------------------------
def bmi(weight, height_cm):
    if height_cm <= 0 or weight <= 0:
        return np.nan
    return weight / ((height_cm / 100) ** 2)

def hb_risk_score(hb):
    if hb < 7: return 3
    if hb < 8: return 2
    if hb < 11: return 1
    return 0

def age_band(age):
    if age < 18: return 3
    if age < 25: return 2
    if age < 30: return 1
    if age < 35: return 0
    return 1

def log1p_safe(x):
    return np.log1p(max(0, x))

def days_between(d1, d2):
    if d1 is None or d2 is None:
        return np.nan
    return (d2 - d1).days

# -----------------------------
# UI
# -----------------------------
st.title("📊 Low Birth Weight (LBW) Risk Assessment")

with st.form("lbw_form"):

    st.subheader("👩 Beneficiary Details")
    age = st.number_input("Age (years)", 15, 49, 22)
    height_cm = st.number_input("Height (cm)", 130, 200, 155)

    st.subheader("🧬 Clinical")
    hb = st.number_input("Hemoglobin (g/dL)", 4.0, 16.0, 10.5)
    w1 = st.number_input("Weight PW1 (kg)", 30.0, 120.0, 45.0)
    w2 = st.number_input("Weight PW2 (kg)", 30.0, 120.0, 48.0)
    w3 = st.number_input("Weight PW3 (kg)", 30.0, 120.0, 50.0)
    w4 = st.number_input("Weight PW4 (kg)", 30.0, 120.0, 52.0)

    st.subheader("🤰 Pregnancy & ANC")
    parity = st.number_input("Child order / parity", 0, 6, 1)
    anc_count = st.number_input("Number of ANCs completed", 0, 6, 3)
    pc_visits = st.number_input("PC visits", 0, 10, 3)
    tt_given = st.selectbox("TT Injection given?", ["Yes", "No"])

    st.subheader("📅 Dates")
    lmp = st.date_input("LMP Date", value=None)
    inst1 = st.date_input("INST1 Date", value=None)

    st.subheader("🍎 Nutrition")
    food_group = st.selectbox("Food Groups Category", [1, 2, 3, 4, 5])
    ifa = st.number_input("IFA tablets last month", 0, 120, 30)
    calcium = st.number_input("Calcium tablets last month", 0, 120, 30)

    st.subheader("💰 Schemes")
    jsy = st.selectbox("Registered for JSY?", ["Yes", "No"])
    raj = st.selectbox("Registered for RAJHSRI?", ["Yes", "No"])
    jsy_inst = st.number_input("JSY installments", 0, 2, 0)
    pmmvy_inst = st.number_input("PMMVY installments", 0, 3, 0)

    submit = st.form_submit_button("🔮 Predict LBW Risk")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    row = {
        "BMI_PW1_Prog": bmi(w1, height_cm),
        "BMI_PW2_Prog": bmi(w2, height_cm),
        "BMI_PW3_Prog": bmi(w3, height_cm),
        "BMI_PW4_Prog": bmi(w4, height_cm),
        "Child order/parity": parity,
        "No of ANCs completed": anc_count,
        "No of PC's total visit to PW": pc_visits,
        "Service received during last ANC: TT Injection given": int(tt_given == "Yes"),
        "Food_Groups_Category": food_group,
        "No. of IFA tablets received/procured in last one month_log1p": log1p_safe(ifa),
        "No. of calcium tablets consumed in last one month_log1p": log1p_safe(calcium),
        "Registered for cash transfer scheme: JSY": int(jsy == "Yes"),
        "Registered for cash transfer scheme: RAJHSRI": int(raj == "Yes"),
        "JSY-Number of installment received": jsy_inst,
        "PMMVY-Number of installment received": pmmvy_inst,
        "measured_HB_risk_score_m": hb_risk_score(hb),
        "Beneficiary_age_band_m": age_band(age),
        "LMPtoINST1_log1p_m": log1p_safe(days_between(lmp, inst1))
    }

    X = pd.DataFrame([row])

    # ensure all features exist
    for f in FEATURES:
        if f not in X.columns:
            X[f] = np.nan

    X = X[FEATURES]

    # 🔑 FINAL FIX: encode categoricals as integer codes
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = pd.Categorical(X[c]).codes.astype("float32")

    # prediction
    prob = float(model.predict_proba(X)[0, 1])

    st.success(f"**LBW Risk Probability:** {prob:.2f}")

    # -----------------------------
    # SHAP
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    shap_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "SHAP value": shap_vals[0]
        })
        .assign(abs=lambda d: d["SHAP value"].abs())
        .sort_values("abs", ascending=False)
        .head(10)
    )

    st.subheader("🔍 Top drivers of risk")
    st.dataframe(shap_df[["Feature", "SHAP value"]], use_container_width=True)
