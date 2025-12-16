# ==============================
# LBW Risk Prediction App
# Option B: Monotonic XGBoost
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import date

# ------------------------------
# App config
# ------------------------------
st.set_page_config(
    page_title="LBW Risk Prediction",
    layout="wide"
)

# Reset state for new user
if "reset" not in st.session_state:
    st.session_state.clear()
    st.session_state["reset"] = True

MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/xgb_mono.json"
FEATURES_PATH = f"{MODEL_DIR}/features.json"
BG_PATH = f"{MODEL_DIR}/background.csv"

# ------------------------------
# Load artifacts
# ------------------------------
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/xgb_mono.pkl")
    FEATURES = json.load(open("model/features.json"))
    BACKGROUND = pd.read_csv("model/background.csv")
    return model, FEATURES, BACKGROUND

    FEATURES = json.load(open(FEATURES_PATH))
    BACKGROUND = pd.read_csv(BG_PATH)

    return booster, FEATURES, BACKGROUND

booster, FEATURES, BACKGROUND = load_artifacts()

# ------------------------------
# Helper functions
# ------------------------------
def yes_no(val): 
    return 1 if val == "Yes" else 0

def bmi(weight, height_cm):
    if weight <= 0 or height_cm <= 0:
        return np.nan
    return weight / ((height_cm / 100) ** 2)

def hb_risk_score(hb):
    if hb < 7: return 3
    elif hb < 8: return 2
    elif hb < 11: return 1
    else: return 0

def age_band(age):
    if age < 19: return 3
    elif age < 25: return 2
    elif age < 30: return 1
    elif age < 35: return 0
    else: return 1

def reg_bucket(days):
    if days <= 90: return "Early"
    elif days <= 180: return "Mid"
    else: return "Late"

def log1p_safe(x):
    return np.log1p(max(0, x))

def days_between(d1, d2):
    if d1 and d2:
        return (d2 - d1).days
    return np.nan

# ------------------------------
# UI
# ------------------------------
st.title("Low Birth Weight (LBW) Risk Assessment")

with st.form("lbw_form"):

    st.header("Beneficiary Details")
    name = st.text_input("Beneficiary name")

    st.header("Anthropometry")
    age = st.number_input("Age (years)", 15, 49)
    height = st.number_input("Height (cm)", 120, 190)
    hb = st.number_input("Hemoglobin (g/dL)", 4.0, 16.0)

    st.header("Weight (kg)")
    w1 = st.number_input("PW1 weight", 30.0, 120.0)
    w2 = st.number_input("PW2 weight", 30.0, 120.0)
    w3 = st.number_input("PW3 weight", 30.0, 120.0)
    w4 = st.number_input("PW4 weight", 30.0, 120.0)

    st.header("Pregnancy & ANC")
    parity = st.number_input("Child order / parity", 0, 10)
    living_children = st.number_input("Number of living children", 0, 10)

    lmp = st.date_input("LMP date")
    reg_date = st.date_input("Registration date")
    inst1 = st.date_input("INST1 date", value=None)
    inst2 = st.date_input("INST2 date", value=None)
    inst3 = st.date_input("INST3 date", value=None)

    anc_count = st.number_input("ANCs completed", 0, 6)
    pc_visits = st.number_input("PC visits", 0, 10)
    tt = st.selectbox("TT injection given", ["Yes", "No"])

    st.header("Nutrition")
    ifa = st.number_input("IFA tablets (last month)", 0, 120)
    calcium = st.number_input("Calcium tablets (last month)", 0, 120)
    food_grp = st.selectbox("Food groups category", [1,2,3,4,5])
    fruits = st.selectbox("Vegetables/fruits grown at home", ["Yes","No"])

    st.header("Household & SES")
    toilet = st.selectbox("Toilet type", ["Improved","Basic","None"])
    water = st.selectbox("Drinking water source", ["Piped","Groundwater","Surface","Other"])
    education = st.selectbox("Education", [
        "No schooling","Primary","Middle","Secondary","Graduate & above"
    ])
    asset_score = st.number_input("Household asset score", 0.0, 50.0)
    wm = st.selectbox("Washing machine", ["Yes","No"])
    ac = st.selectbox("AC / Cooler", ["Yes","No"])

    st.header("Programme Exposure")
    jsy = st.selectbox("JSY registered", ["Yes","No"])
    raj = st.selectbox("RAJHSRI registered", ["Yes","No"])
    jsy_inst = st.selectbox("JSY installments", [0,1,2])
    pmmvy_inst = st.selectbox("PMMVY installments", [0,1,2,3])
    asha = st.selectbox("ASHA visit", ["Yes","No"])
    aww = st.selectbox("AWW visit", ["Yes","No"])
    social = st.number_input("Social media count", 0, 20)

    submit = st.form_submit_button("Predict LBW Risk")

# ------------------------------
# Prediction
# ------------------------------
if submit:

    df = pd.DataFrame([{
        "Child order/parity": parity,
        "MonthConception": lmp.month,
        "BMI_PW1_Prog": bmi(w1, height),
        "BMI_PW2_Prog": bmi(w2, height),
        "BMI_PW3_Prog": bmi(w3, height),
        "BMI_PW4_Prog": bmi(w4, height),
        "consume_tobacco": 0,
        "consume_alcohol": 0,
        "RegistrationBucket": reg_bucket(days_between(lmp, reg_date)),
        "No of ANCs completed": anc_count,
        "No of PC's total visit to PW": pc_visits,
        "Service received during last ANC: TT Injection given": yes_no(tt),
        "No. of IFA tablets received/procured in last one month_log1p": log1p_safe(ifa),
        "No. of calcium tablets consumed in last one month_log1p": log1p_safe(calcium),
        "Food_Groups_Category": food_grp,
        "Whether  vegetables/fruits are grown by the household for own consumption": yes_no(fruits),
        "Household_Assets_Score_log1p": log1p_safe(asset_score),
        "Household asset: Washing Machine": yes_no(wm),
        "Household asset: Air Conditioner/ cooler": yes_no(ac),
        "toilet_type": toilet,
        "Main Source of drinking water": water,
        "education": education,
        "Social_Media_Count": social,
        "Registered for cash transfer scheme: JSY": yes_no(jsy),
        "Registered for cash transfer scheme: RAJHSRI": yes_no(raj),
        "PMMVY-Number of installment received": pmmvy_inst,
        "JSY-Number of installment received": jsy_inst,
        "Visit of ASHA in household in last three months": yes_no(asha),
        "Visit of AWW in household in last three months": yes_no(aww),
        "measured_HB_risk_score_m": hb_risk_score(hb),
        "Beneficiary_age_band_m": age_band(age),
        "LMPtoEarliestANC_log1p_m": log1p_safe(days_between(lmp, reg_date)),
        "LMPtoINST1_log1p_m": log1p_safe(days_between(lmp, inst1)),
        "LMPtoINST2_log1p_m": log1p_safe(days_between(lmp, inst2)),
        "LMPtoINST3_log1p_m": log1p_safe(days_between(lmp, inst3))
    }])

    df = df[FEATURES]

    dmat = xgb.DMatrix(df)
    prob = model.predict_proba(df)[0, 1]


    st.success(f"Predicted LBW Risk: **{prob:.2%}**")

    # ------------------------------
    # SHAP
    # ------------------------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(df)

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP value": shap_vals[0]
    }).sort_values("SHAP value", key=abs, ascending=False)

    st.subheader("Top drivers of risk")
    st.dataframe(shap_df.head(10))

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"].head(10)[::-1],
            shap_df["SHAP value"].head(10)[::-1])
    ax.set_title("SHAP contributions")
    st.pyplot(fig)

