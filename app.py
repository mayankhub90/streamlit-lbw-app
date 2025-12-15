import streamlit as st
import traceback
import pandas as pd
import numpy as np
import xgboost as xgb
import cloudpickle
import json
import os
import shap
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="LBW Risk Prediction App", layout="wide")
st.title("LBW Risk Prediction Application")

# ──────────────────────────────────────────────────────────────
# SAFE WRAPPER — PREVENT BLANK PAGE ON ERROR
# ──────────────────────────────────────────────────────────────
try:
    # --------------------------------------------------------------------
    # LOAD ML ARTIFACTS
    # --------------------------------------------------------------------
    MODEL_DIR = "model"
    PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
    BOOSTER_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
    FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
    FEATURES_T_PATH = os.path.join(MODEL_DIR, "features_transformed.json")
    BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.csv")

    @st.cache_resource
    def load_artifacts():
        # Load preprocessor
        if not os.path.exists(PREPROC_PATH):
            return None, None, None, None, "ERROR: preprocessor_cloud.pkl missing."
        try:
            with open(PREPROC_PATH, "rb") as f:
                preproc = cloudpickle.load(f)
        except Exception as e:
            return None, None, None, None, f"Failed to load preprocessor: {e}"

        # Load booster
        booster = None
        try:
            booster = xgb.Booster()
            booster.load_model(BOOSTER_PATH)
        except Exception as e:
            booster = None

        # Load features
        if os.path.exists(FEATURES_PATH):
            FEATURES = json.load(open(FEATURES_PATH))
        else:
            return None, None, None, None, "ERROR: features.json missing."

        # Load transformed feature names
        FEATURES_T = []
        if os.path.exists(FEATURES_T_PATH):
            try:
                FEATURES_T = json.load(open(FEATURES_T_PATH))
            except:
                FEATURES_T = []

        # Background for SHAP
        BACKGROUND = None
        if os.path.exists(BACKGROUND_PATH):
            BACKGROUND = pd.read_csv(BACKGROUND_PATH)

        return preproc, booster, FEATURES, FEATURES_T, BACKGROUND, None

    preproc, booster, FEATURES, FEATURES_T, BACKGROUND, load_err = load_artifacts()

    if load_err is not None:
        st.error(load_err)
        st.stop()

    # Numeric fields found automatically
    NUMERIC_COLS = [
        f for f in FEATURES
        if ("age" in f.lower()) or ("bmi" in f.lower()) or ("gap" in f.lower())
        or ("lmp" in f.lower()) or ("ifa" in f.lower()) or ("calcium" in f.lower())
        or ("parity" in f.lower()) or ("month" in f.lower()) 
        or ("number" in f.lower()) or ("score" in f.lower())
    ]

    # --------------------------------------------------------------------
    # CATEGORICAL OPTIONS (per your specifications)
    # --------------------------------------------------------------------
    CAT_OPTIONS = {
        "consume_tobacco": ["Yes", "No"],
        "Status of current chewing of tobacco": ["Yes", "No"],
        "consume_alcohol": ["Yes", "No"],
        "Food_Groups_Category": ["1", "2", "3", "4", "5"],
        "Social_Media_Category": ["Yes", "No"],
        "toilet_type_clean": ["Yes", "No"],
        "water_source_clean": ["Yes", "No"],
        "education_clean": [
            "Illiterate", "Primary", "Upper Primary", "Secondary",
            "Senior Secondary", "Graduate", "Graduate and Above"
        ],
        "Registered for cash transfer scheme: JSY": ["Yes", "No"],
        "Registered for cash transfer scheme: RAJHSRI": ["Yes", "No"],
        "PMMVY-Number of installment received": ["0", "1", "2", "3"],
        "JSY-Number of installment received": ["0", "1", "2"],
        "Service received during last ANC: TT Injection given": ["Yes", "No"],
        "RegistrationBucket": [""],   # AUTO COMPUTED
        "ANCBucket": [""]            # LEFT BLANK FOR NOW
    }

    # --------------------------------------------------------------------
    # INPUT FORM HELPERS
    # --------------------------------------------------------------------
    def clean_inputs(d):
        """Convert blank strings to NaN, numeric inputs to float."""
        cleaned = {}
        for k, v in d.items():
            if k in NUMERIC_COLS:
                if v in ("", None):
                    cleaned[k] = np.nan
                else:
                    try:
                        cleaned[k] = float(v)
                    except:
                        cleaned[k] = np.nan
            else:
                cleaned[k] = v if v not in ("", None) else np.nan
        return cleaned

    def compute_registration_bucket(lmp_date, reg_date):
        """Apply your rule: Early ≤ 84d, On-time ≤ 168d, Late > 168d."""
        if (lmp_date is None) or (reg_date is None):
            return ""
        days = (reg_date - lmp_date).days
        if days <= 84:
            return "Early"
        elif days <= 168:
            return "On-time"
        else:
            return "Late"

    def predict_proba(df_raw):
        X = preproc.transform(df_raw)
        dmat = xgb.DMatrix(X.astype(np.float32))
        pred = booster.predict(dmat)
        if isinstance(pred, np.ndarray):
            return float(pred[0])
        return float(pred)

    def compute_shap(df_raw):
        X = preproc.transform(df_raw)
        expl = shap.TreeExplainer(booster)
        shap_vals = expl.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return shap_vals[0]

    # --------------------------------------------------------------------
    # USER INPUT FORM
    # --------------------------------------------------------------------
    st.header("Enter Beneficiary Details")

    beneficiary_name = st.text_input("Beneficiary Name")

    col_lmp, col_reg = st.columns(2)
    LMP = col_lmp.date_input("LMP Date", value=None)
    REG_DATE = col_reg.date_input("Registration Date", value=None)

    # Auto compute bucket
    REG_BUCKET = compute_registration_bucket(LMP, REG_DATE)

    st.write(f"**Registration Bucket:** {REG_BUCKET}")

    # Store inputs
    form_inputs = {"RegistrationBucket": REG_BUCKET}

    # --------------------------------------------------------------------
    # EXPANDERS + TWO-COLUMN UI
    # --------------------------------------------------------------------
    # BACKGROUND VARIABLES
    with st.expander("Background Variables"):
        c1, c2 = st.columns(2)
        form_inputs["Beneficiary age"] = c1.number_input("Beneficiary age", value=np.nan)
        form_inputs["Child order/parity"] = c2.number_input("Child order/parity", value=np.nan)
        form_inputs["Number of living child at now"] = c1.number_input("Number of living children", value=np.nan)
        form_inputs["MonthConception"] = c2.number_input("Month of Conception", value=np.nan)

    # HOUSEHOLD
    with st.expander("Household Environment"):
        c1, c2 = st.columns(2)
        form_inputs["toilet_type_clean"] = c1.selectbox("Toilet Type", [""] + CAT_OPTIONS["toilet_type_clean"])
        form_inputs["water_source_clean"] = c2.selectbox("Water Source", [""] + CAT_OPTIONS["water_source_clean"])
        form_inputs["education_clean"] = c1.selectbox("Education Level", [""] + CAT_OPTIONS["education_clean"])
        form_inputs["Household_Assets_Score_log1p"] = c2.number_input("Household Asset Score", value=np.nan)

    # HEALTH BEHAVIORS
    with st.expander("Health Behaviors"):
        c1, c2 = st.columns(2)
        form_inputs["consume_tobacco"] = c1.selectbox("Tobacco Consumption", [""] + CAT_OPTIONS["consume_tobacco"])
        form_inputs["Status of current chewing of tobacco"] = c2.selectbox("Chewing Tobacco", [""] + CAT_OPTIONS["Status of current chewing of tobacco"])
        form_inputs["consume_alcohol"] = c1.selectbox("Alcohol Consumption", [""] + CAT_OPTIONS["consume_alcohol"])
        form_inputs["Social_Media_Category"] = c2.selectbox("Social Media Category", [""] + CAT_OPTIONS["Social_Media_Category"])

    # NUTRITION
    with st.expander("Nutrition Indicators"):
        c1, c2 = st.columns(2)
        form_inputs["Food_Groups_Category"] = c1.selectbox("Food Groups Category", [""] + CAT_OPTIONS["Food_Groups_Category"])
        form_inputs["BMI_PW1_Prog"] = c2.number_input("BMI PW1", value=np.nan)
        form_inputs["BMI_PW2_Prog"] = c1.number_input("BMI PW2", value=np.nan)
        form_inputs["BMI_PW3_Prog"] = c2.number_input("BMI PW3", value=np.nan)
        form_inputs["BMI_PW4_Prog"] = c1.number_input("BMI PW4", value=np.nan)

    # ANC & SERVICES
    with st.expander("ANC & Health Services"):
        c1, c2 = st.columns(2)
        form_inputs["counselling_gap_days"] = c1.number_input("Counselling Gap Days", value=np.nan)
        form_inputs["LMPtoINST1"] = c2.number_input("LMP to INST1 (days)", value=np.nan)
        form_inputs["LMPtoINST2"] = c1.number_input("LMP to INST2 (days)", value=np.nan)
        form_inputs["LMPtoINST3"] = c2.number_input("LMP to INST3 (days)", value=np.nan)
        form_inputs["No of ANCs completed"] = c1.number_input("No. of ANCs completed", value=np.nan)
        form_inputs["Service received during last ANC: TT Injection given"] = c2.selectbox(
            "TT Injection Given?", [""] + CAT_OPTIONS["Service received during last ANC: TT Injection given"]
        )
        form_inputs["ANCBucket"] = c1.selectbox("ANC Bucket", [""])  # blank for now

    # SUPPLEMENTATION
    with st.expander("Supplementation"):
        c1, c2 = st.columns(2)
        form_inputs["No. of IFA tablets received/procured in last one month_log1p"] = c1.number_input("IFA Tablets (last month)", value=np.nan)
        form_inputs["No. of calcium tablets consumed in last one month_log1p"] = c2.number_input("Calcium Tablets (last month)", value=np.nan)

    # CASH BENEFITS
    with st.expander("Cash Transfer Schemes"):
        c1, c2 = st.columns(2)
        form_inputs["Registered for cash transfer scheme: JSY"] = c1.selectbox(
            "Registered for JSY?", [""] + CAT_OPTIONS["Registered for cash transfer scheme: JSY"]
        )
        form_inputs["Registered for cash transfer scheme: RAJHSRI"] = c2.selectbox(
            "Registered for RAJSHRI?", [""] + CAT_OPTIONS["Registered for cash transfer scheme: RAJHSRI"]
        )
        form_inputs["PMMVY-Number of installment received"] = c1.selectbox(
            "PMMVY Installments Received", [""] + CAT_OPTIONS["PMMVY-Number of installment received"]
        )
        form_inputs["JSY-Number of installment received"] = c2.selectbox(
            "JSY Installments Received", [""] + CAT_OPTIONS["JSY-Number of installment received"]
        )

    # --------------------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------------------
    if st.button("Predict LBW Risk"):
        st.subheader(f"Prediction for: {beneficiary_name if beneficiary_name else 'Beneficiary'}")

        cleaned = clean_inputs(form_inputs)
        df = pd.DataFrame([cleaned], columns=FEATURES)

        prob = predict_proba(df)

        st.metric("Predicted LBW Probability", f"{prob:.3f}")

        # Interpretation
        if prob < 0.3:
            st.success("Low risk of LBW")
        elif prob < 0.6:
            st.warning("Moderate risk of LBW")
        else:
            st.error("HIGH risk of LBW — beneficiary requires attention")

        # ----------------------------------------------------------------
        # SHAP EXPLANATIONS
        # ----------------------------------------------------------------
        shap_vals = compute_shap(df)

        # Feature names
        if FEATURES_T and len(FEATURES_T) == len(shap_vals):
            fn = FEATURES_T
        else:
            fn = [f"f_{i}" for i in range(len(shap_vals))]

        st.subheader("SHAP Feature Impact")

        # BAR CHART (Top 5 + Top 5)
        sorted_idx = np.argsort(shap_vals)
        neg_idx = sorted_idx[:5]
        pos_idx = sorted_idx[-5:]

        st.write("### Factors Decreasing Risk")
        fig1, ax1 = plt.subplots()
        ax1.barh([fn[i] for i in neg_idx], shap_vals[neg_idx], color="green")
        st.pyplot(fig1)

        st.write("### Factors Increasing Risk")
        fig2, ax2 = plt.subplots()
        ax2.barh([fn[i] for i in pos_idx], shap_vals[pos_idx], color="red")
        st.pyplot(fig2)

        # WATERFALL
        st.write("### Detailed SHAP Waterfall Plot")
        shap.waterfall_plot = shap.plots.waterfall
        fig3 = shap.plots.waterfall(shap_vals, feature_names=fn, show=False)
        st.pyplot(fig3)

# ──────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ──────────────────────────────────────────────────────────────
except Exception as e:
    st.error("The application encountered an error:")
    st.code(traceback.format_exc())
