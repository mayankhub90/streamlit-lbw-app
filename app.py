# app.py (loading helper)
import os, joblib
try:
    import cloudpickle
except Exception:
    cloudpickle = None

MODEL_PATH = "model/pipeline.pkl"

def load_pipeline_safe(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, "Model file not found"
    # try joblib first
    try:
        pl = joblib.load(path)
        return pl, None
    except Exception as e_job:
        # try cloudpickle fallback if available
        if cloudpickle is None:
            try:
                import pip, subprocess
                subprocess.check_call(["pip", "install", "cloudpickle"])
                import cloudpickle
            except Exception:
                return None, f"joblib.load failed: {e_job}\n(no cloudpickle available)"
        try:
            with open(path, "rb") as f:
                pl = cloudpickle.load(f)
            return pl, None
        except Exception as e_cloud:
            return None, f"joblib.load failed: {e_job}\ncloudpickle.load failed: {e_cloud}"

pipeline, background, load_error = None, None, None
pipeline, background, load_error = load_pipeline_and_background()  # adapt to your function naming

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

FEATURES = [
 "Beneficiary age",
 "measured_HB_risk_bin",
 "Child order/parity",
 "Number of living child at now",
 "MonthConception",
 "BMI_PW1_Prog",
 "BMI_PW2_Prog",
 "BMI_PW3_Prog",
 "BMI_PW4_Prog",
 "consume_tobacco",
 "Status of current chewing of tobacco",
 "consume_alcohol",
 "RegistrationBucket",
 "counselling_gap_days",
 "ANCBucket",
 "LMPtoINST1",
 "LMPtoINST2",
 "LMPtoINST3",
 "No of ANCs completed",
 "Service received during last ANC: TT Injection given",
 "No. of IFA tablets received/procured in last one month_log1p",
 "No. of calcium tablets consumed in last one month_log1p",
 "Food_Groups_Category",
 "Household_Assets_Score_log1p",
 "toilet_type_clean",
 "water_source_clean",
 "education_clean",
 "Social_Media_Category",
 "Registered for cash transfer scheme: JSY",
 "Registered for cash transfer scheme: RAJHSRI",
 "PMMVY-Number of installment received",
 "JSY-Number of installment received"
]

MODEL_PATH = os.path.join("model", "pipeline.pkl")
BACKGROUND_PATH = os.path.join("model", "background.csv")

def identify_numeric_cols(features):
    indicators = [
        "age", "number", "no", "month", "lmpto", "counselling_gap_days", "bmi", "score", "installment", "count"
    ]
    numeric = []
    for c in features:
        low = c.lower()
        if any(ind in low for ind in indicators) or "log1p" in c:
            numeric.append(c)
    return numeric

NUMERIC_COLS = identify_numeric_cols(FEATURES)

@st.cache_resource
def load_pipeline_and_background():
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model not found at model/pipeline.pkl. Run train_pipeline.py first."
    pipeline = joblib.load(MODEL_PATH)
    background = None
    if os.path.exists(BACKGROUND_PATH):
        background = pd.read_csv(BACKGROUND_PATH)
        # ensure same column order
        background = background[FEATURES].copy()
    return pipeline, background, None

pipeline, background, load_error = load_pipeline_and_background()

st.set_page_config(page_title="LBW Predictor + SHAP", layout="centered")
st.title("LBW Predictor — Fill the form and click Predict")
st.markdown("Form uses your selected features. After prediction a SHAP-style explanation will show top contributors.")

if load_error:
    st.error(load_error)
    st.stop()

threshold = st.sidebar.slider("Decision threshold (probability)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
cols = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    col = cols[i % 2]
    low = feat.lower()
    if any(x in low for x in ["consume_", "registered", "status", "service received", "registered for"]):
        inputs[feat] = col.selectbox(feat, options=["", "Yes", "No"], index=0)
    elif feat in NUMERIC_COLS:
        inputs[feat] = col.text_input(feat, value="", placeholder="Enter numeric")
    else:
        inputs[feat] = col.text_input(feat, value="", placeholder="Enter category/text")

st.write("---")
predict_btn = st.button("Predict LBW")

def build_input_df(inputs_dict):
    row = {}
    for feat, raw in inputs_dict.items():
        v = str(raw).strip()
        if v == "" or v.lower() in ["nan", "none"]:
            row[feat] = np.nan
            continue
        if feat in NUMERIC_COLS:
            try:
                row[feat] = float(v.replace(",", ""))
            except:
                row[feat] = np.nan
        else:
            row[feat] = v
    df = pd.DataFrame([row], columns=FEATURES)
    return df

if predict_btn:
    st.info("Running model prediction...")
    X_input = build_input_df(inputs)
    st.subheader("Input preview")
    st.dataframe(X_input.T)

    if pipeline is None:
        st.error("Model pipeline not loaded.")
        st.stop()

    try:
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X_input)[:, 1][0]
            label = "LBW" if prob >= threshold else "Not LBW"
            st.metric("Predicted probability (LBW)", f"{prob:.3f}")
            st.write("Prediction:", label)
        else:
            pred = pipeline.predict(X_input)[0]
            st.metric("Prediction (model output)", str(pred))
            prob = None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if background is None:
        st.warning("background.csv not found; SHAP explanation unavailable. Save model/background.csv during training.")
    else:
        st.write("---")
        st.subheader("Feature contributions (SHAP-style)")
        bg_for_shap = background.sample(n=min(50, len(background)), random_state=42)

        def predict_fn(x):
            if isinstance(x, np.ndarray):
                df_x = pd.DataFrame(x, columns=FEATURES)
            else:
                df_x = pd.DataFrame(x, columns=FEATURES)
            return pipeline.predict_proba(df_x)[:, 1]

        try:
            with st.spinner("Computing SHAP values (may take a little time)..."):
                explainer = shap.KernelExplainer(predict_fn, bg_for_shap, link="logit")
                shap_values = explainer.shap_values(X_input, nsamples=200)
                sv = np.array(shap_values).reshape(-1)
        except Exception as e:
            st.error(f"SHAP failed: {e}")
            sv = None

        if sv is not None:
            contrib = pd.DataFrame({
                "feature": FEATURES,
                "shap_value": sv,
                "abs_shap": np.abs(sv),
                "value": X_input.iloc[0].values
            })
            contrib = contrib.sort_values("abs_shap", ascending=False).reset_index(drop=True)
            st.write("Top contributors:")
            st.dataframe(contrib[["feature", "value", "shap_value"]].head(12))

            # plot
            topk = contrib.head(12).sort_values("shap_value")
            fig, ax = plt.subplots(figsize=(8, 4 + 0.25 * len(topk)))
            colors = ["#d9534f" if v>0 else "#5cb85c" for v in topk["shap_value"]]
            ax.barh(topk["feature"], topk["shap_value"], color=colors)
            ax.axvline(0, color="k", linewidth=0.8)
            ax.set_xlabel("SHAP value (contribution)")
            st.pyplot(fig)

            csv = contrib.to_csv(index=False).encode("utf-8")

            st.download_button("Download contributions CSV", csv, file_name="shap_contributions.csv")
