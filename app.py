# Put this at the very top of app.py (replace any debug prints you added)
import streamlit as st
import traceback, sys

# Minimal visible header so we know wrapper loaded
st.title("LBW App — debug wrapper active")

try:
    

# ---------- artifact loader & predict (use cloudpickle preproc + xgb json) ----------
import os, joblib, json, streamlit as st, numpy as np, pandas as pd
import xgboost as xgb
import shap, matplotlib.pyplot as plt

MODEL_DIR = "model"
PREPROC_CLOUD = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
BOOSTER_JSON = os.path.join(MODEL_DIR, "xgb_model.json")
CLASSIFIER_CLOUD = os.path.join(MODEL_DIR, "classifier_cloud.pkl")
FEATURES_JSON = os.path.join(MODEL_DIR, "features.json")
FEATURES_T_JSON = os.path.join(MODEL_DIR, "features_transformed.json")
BACKGROUND_CSV = os.path.join(MODEL_DIR, "background.csv")

@st.cache_resource
def load_artifacts():
    # load preprocessor via cloudpickle
    if not os.path.exists(PREPROC_CLOUD):
        return None, None, None, "preprocessor_cloud.pkl missing"
    try:
        import cloudpickle
        with open(PREPROC_CLOUD,"rb") as f:
            preproc = cloudpickle.load(f)
    except Exception as e:
        return None, None, None, f"Failed loading preprocessor_cloud.pkl: {e}"

    # load booster if exists
    booster = None
    if os.path.exists(BOOSTER_JSON):
        try:
            booster = xgb.Booster()
            booster.load_model(BOOSTER_JSON)
        except Exception as e:
            booster = None

    # fallback classifier
    clf = None
    if booster is None and os.path.exists(CLASSIFIER_CLOUD):
        clf = joblib.load(CLASSIFIER_CLOUD)

    # features
    features = []
    if os.path.exists(FEATURES_JSON):
        features = json.load(open(FEATURES_JSON))
    features_t = []
    if os.path.exists(FEATURES_T_JSON):
        features_t = json.load(open(FEATURES_T_JSON))

    # background
    bg = None
    if os.path.exists(BACKGROUND_CSV):
        bg = pd.read_csv(BACKGROUND_CSV)

    if preproc is None:
        return None, None, None, "preproc load failed"
    return preproc, (booster, clf), (features, features_t, bg), None

preproc, model_objs, feat_info, load_err = load_artifacts()
if load_err:
    st.error("Model load error: " + str(load_err))
    st.stop()

booster, clf = model_objs
FEATURES, FEATURES_T, BACKGROUND = feat_info

# Prediction helper
def predict_proba_from_raw(df_raw):
    Xtr = preproc.transform(df_raw)
    if booster is not None:
        dmat = xgb.DMatrix(Xtr.astype(np.float32))
        proba = booster.predict(dmat)
        # if shape NxC, take column 1
        if isinstance(proba, np.ndarray) and proba.ndim>1:
            return float(proba[:,1][0])
        return float(proba[0])
    elif clf is not None:
        proba = clf.predict_proba(Xtr)
        return float(proba[:,1][0])
    else:
        raise RuntimeError("No model available for prediction")

# SHAP helper (fast when booster present)
def shap_for_row(df_raw):
    Xtr = preproc.transform(df_raw)
    if booster is not None:
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(Xtr)  # array or list
        arr = sv[0] if isinstance(sv, list) else sv
        return np.array(arr).reshape(arr.shape)[0]
    else:
        # fallback: kernel explainer on pipeline predict_proba using background (may be slow)
        if BACKGROUND is None:
            return None
        bg = BACKGROUND.sample(n=min(50, len(BACKGROUND)), random_state=42)
        def predict_fn(x):
            if isinstance(x, np.ndarray):
                df = pd.DataFrame(x, columns=FEATURES)
            else:
                df = pd.DataFrame(x, columns=FEATURES)
            return float(preproc and (clf.predict_proba(preproc.transform(df))[:,1]) if clf else 0.0)
        expl = shap.KernelExplainer(predict_fn, bg, link="logit")
        vals = expl.shap_values(df_raw, nsamples=100)
        return np.array(vals).reshape(-1)

# === START OF YOUR ORIGINAL app.py CODE ===
    # Paste all your original code (imports, loader, UI, handlers) here,
    # starting immediately after this comment block. Do NOT include the
    # try/except wrapper again inside your original code.
    #
    # Example:
    # import joblib, pandas as pd, numpy as np, ...
    # pipeline, background, load_error = load_pipeline_and_background()
    # ... rest of your app ...
    #
    # === END OF YOUR ORIGINAL app.py CODE ===


except Exception as e:
    # Show the exception and full traceback in the Streamlit UI
    st.error("App crashed during startup or rendering — full traceback below.")
    tb = traceback.format_exc()
    # show a short message and the full traceback
    st.text(str(e))
    st.text(tb)

    # Also print to stdout for Streamlit logs (helpful)
    print("APP EXCEPTION (printed for logs):", str(e))
    print(tb)
    # keep the app alive (so you can see error) — do not re-raise
