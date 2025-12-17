import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="LBW Risk Assessment",
    layout="wide"
)

# -----------------------------
# Load artifacts (once)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/xgb_model.pkl")
    features = json.load(open("model/features.json"))
    background = pd.read_csv("model/background.csv")
    return model, features, background

model, FEATURES, BACKGROUND = load_artifacts()

# -----------------------------
# Helper functions
# -----------------------------
def bmi(weight, height_cm):
    if height_cm <= 0:
        return np.nan
    return weight / ((height_cm / 100) ** 2)

def hb_risk_score(hb):
    if pd.isna(hb):
        return np.nan
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

def days_between(d1, d2):
    if d1 is None or d2 is None:
        return np.nan
    return (d2 - d1).days

def log1p_safe(x):
    return np.log1p(max(0, x)) if pd.notna(x) else np.nan

# -----------------------------
# UI
# -----------------------------
st.title("📊 Low Birth Weight (LBW) Risk Assessment")

with st.form("lbw_form"):

    st.subheader("👩 Beneficiary Details")
    name = st.text_input("Beneficiary Name (for reference only)")
    age = st.number_input("Age (years)", 15, 49, 22)
    height_cm = st.number_input("Height (cm)", 130, 190, 155)

    st.subheader("🧬 Clinical & Nutrition")
    hb = st.number_input("Hemoglobin (g/dL)", 4.0, 16.0, 10.5)
    w1 = st.number_input("Weight PW1 (kg)", 30.0, 100.0, 45.0)
    w2 = st.number_input("Weight PW2 (kg)", 30.0, 100.0, 48.0)
    w3 = st.number_input("Weight PW3 (kg)", 30.0, 100.0, 50.0)
    w4 = st.number_input("Weight PW4 (kg)", 30.0, 100.0, 52.0)

    st.subheader("🤰 Pregnancy & ANC")
    parity = st.number_input("Child order / parity", 0, 6, 1)
    living_children = st.number_input("Number of living children", 0, 6, 1)
    anc_count = st.number_input("Number of ANCs completed", 0, 6, 3)
    pc_visits = st.number_input("Number of PC visits", 0, 10, 3)
    tt_given = st.selectbox("TT Injection given?", ["Yes", "No"])

    st.subheader("📅 Key Dates")
    lmp = st.date_input("LMP Date")
    inst1 = st.date_input("INST1 Date", value=None)

    st.subheader("🍎 Nutrition & Household")
    food_group = st.selectbox("Food Groups Category", [1, 2, 3, 4, 5])
    ifa = st.number_input("IFA tablets (last month)", 0, 120, 30)
    calcium = st.number_input("Calcium tablets (last month)", 0, 120, 30)

    toilet = st.selectbox("Toilet type improved?", ["Yes", "No"])
    water = st.selectbox("Clean water source?", ["Yes", "No"])
    education = st.selectbox(
        "Education",
        ["Illiterate", "Primary", "Upper Primary", "Secondary",
         "Senior Secondary", "Graduate", "Graduate and above"]
    )

    st.subheader("💰 Schemes")
    jsy = st.selectbox("Registered for JSY?", ["Yes", "No"])
    raj = st.selectbox("Registered for RAJHSRI?", ["Yes", "No"])
    jsy_inst = st.number_input("JSY installments received", 0, 2, 0)
    pmmvy_inst = st.number_input("PMMVY installments received", 0, 3, 0)

    submit = st.form_submit_button("🔮 Predict LBW Risk")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    df = {}

    df["BMI_PW1_Prog"] = bmi(w1, height_cm)
    df["BMI_PW2_Prog"] = bmi(w2, height_cm)
    df["BMI_PW3_Prog"] = bmi(w3, height_cm)
    df["BMI_PW4_Prog"] = bmi(w4, height_cm)

    df["Child order/parity"] = parity
    df["Number of living child at now"] = living_children
    df["No of ANCs completed"] = anc_count
    df["No of PC's total visit to PW"] = pc_visits
    df["Service received during last ANC: TT Injection given"] = int(tt_given == "Yes")

    df["Food_Groups_Category"] = food_group
    df["No. of IFA tablets received/procured in last one month_log1p"] = log1p_safe(ifa)
    df["No. of calcium tablets consumed in last one month_log1p"] = log1p_safe(calcium)

    df["Registered for cash transfer scheme: JSY"] = int(jsy == "Yes")
    df["Registered for cash transfer scheme: RAJHSRI"] = int(raj == "Yes")
    df["JSY-Number of installment received"] = jsy_inst
    df["PMMVY-Number of installment received"] = pmmvy_inst

    df["measured_HB_risk_score_m"] = hb_risk_score(hb)
    df["Beneficiary_age_band_m"] = age_band(age)

    df["LMPtoINST1_log1p_m"] = log1p_safe(days_between(lmp, inst1))

    X = pd.DataFrame([df]).reindex(columns=FEATURES, fill_value=0)

    prob = model.predict_proba(X)[0, 1]

    st.success(f"**LBW Risk Probability:** {prob:.2f}")

    # -----------------------------
    # SHAP
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP": shap_vals[0]
    }).sort_values("SHAP", key=abs, ascending=False).head(10)

    st.subheader("🔍 Top factors influencing this prediction")
    st.dataframe(shap_df)
