import streamlit as st
import traceback

st.title("LBW Prediction App — Debug Wrapper Enabled")

# ======================================================
# TRY BLOCK WRAPS YOUR ENTIRE APP
# ======================================================
try:
    # =======================
    # IMPORTS
    # =======================
    import os, joblib, json
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt

    # =======================
    # MODEL FILE PATHS
    # =======================
    MODEL_DIR = "model"
    PREPROC_CLOUD = os.path.join(MODEL_DIR, "preprocessor_cloud.pkl")
    BOOSTER_JSON = os.path.join(MODEL_DIR, "xgb_model.json")
    CLASSIFIER_CLOUD = os.path.join(MODEL_DIR, "classifier_cloud.pkl")
    FEATURES_JSON = os.path.join(MODEL_DIR, "features.json")
    FEATURES_T_JSON = os.path.join(MODEL_DIR, "features_transformed.json")
    BACKGROUND_CSV = os.path.join(MODEL_DIR, "background.csv")

    # =======================
    # LOAD ARTIFACTS
    # =======================
    @st.cache_resource
    def load_artifacts():
        # --- load preprocessor ---
        if not os.path.exists(PREPROC_CLOUD):
            return None, None, None, "preprocessor_cloud.pkl missing"

        try:
            import cloudpickle
            with open(PREPROC_CLOUD, "rb") as f:
                preproc = cloudpickle.load(f)
        except Exception as e:
            return None, None, None, f"Failed loading preprocessor_cloud.pkl: {e}"

        # --- load booster ---
        booster = None
        if os.path.exists(BOOSTER_JSON):
            try:
                booster = xgb.Booster()
                booster.load_model(BOOSTER_JSON)
            except:
                booster = None

        # --- fallback classifier ---
        clf = None
        if booster is None and os.path.exists(CLASSIFIER_CLOUD):
            clf = joblib.load(CLASSIFIER_CLOUD)

        # --- load features ---
        features = []
        if os.path.exists(FEATURES_JSON):
            features = json.load(open(FEATURES_JSON))

        # --- transformed feature names (optional) ---
        features_t = []
        if os.path.exists(FEATURES_T_JSON):
            features_t = json.load(open(FEATURES_T_JSON))

        # --- background for SHAP ---
        bg = None
        if os.path.exists(BACKGROUND_CSV):
            bg = pd.read_csv(BACKGROUND_CSV)

        return preproc, (booster, clf), (features, features_t, bg), None

    # execute loader
    preproc, model_objs, feat_info, load_err = load_artifacts()

    if load_err:
        st.error("Model load error: " + str(load_err))
        st.stop()

    booster, clf = model_objs
    FEATURES, FEATURES_T, BACKGROUND = feat_info

    st.success("Artifacts loaded successfully")

    # =======================
    # PREDICTION FUNCTION
    # =======================
    def predict_proba_from_raw(df_raw):
        Xtr = preproc.transform(df_raw)
        if booster is not None:
            dmat = xgb.DMatrix(Xtr.astype(np.float32))
            proba = booster.predict(dmat)
            if isinstance(proba, np.ndarray) and proba.ndim > 1:
                return float(proba[:, 1][0])
            return float(proba[0])
        elif clf is not None:
            proba = clf.predict_proba(Xtr)
            return float(proba[:, 1][0])
        else:
            raise RuntimeError("No model available")

    # =======================
    # SHAP EXPLANATION
    # =======================
    def shap_for_row(df_raw):
        Xtr = preproc.transform(df_raw)
        if booster is not None:
            explainer = shap.TreeExplainer(booster)
            sv = explainer.shap_values(Xtr)
            arr = sv[0] if isinstance(sv, list) else sv
            return np.array(arr)[0]
        else:
            return None

    # =======================
    # SIMPLE UI FOR TESTING
    # =======================
    st.subheader("Quick Test Input")

    if FEATURES:
        sample = {f: st.text_input(f"Enter value for {f}") for f in FEATURES}

        if st.button("Predict"):
            df = pd.DataFrame([sample], columns=FEATURES)

            prob = predict_proba_from_raw(df)
            st.metric("LBW Probability", f"{prob:.4f}")

            shap_values = shap_for_row(df)
            if shap_values is not None:
                st.success("SHAP computed")
                st.write(shap_values[:10])  # preview
    else:
        st.warning("FEATURES list missing.")

# ======================================================
# EXCEPT BLOCK — SHOW FULL TRACEBACK
# ======================================================
except Exception as e:
    st.error("APP CRASHED — see full traceback below:")
    st.code(traceback.format_exc())
