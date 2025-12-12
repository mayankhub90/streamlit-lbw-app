# --- app.py: artifact loader + predict + SHAP (replace your old loader) ---
import os, joblib, json, streamlit as st, numpy as np, pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "model"
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
BOOSTER_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
CLASSIFIER_PKL = os.path.join(MODEL_DIR, "classifier.pkl")
FEATURES_JSON = os.path.join(MODEL_DIR, "features.json")
FEATURES_T_JSON = os.path.join(MODEL_DIR, "features_transformed.json")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.csv")

@st.cache_resource
def load_artifacts():
    # load preprocessor
    preproc = None
    if os.path.exists(PREPROC_PATH):
        preproc = joblib.load(PREPROC_PATH)
    else:
        return None, None, None, "preprocessor not found"

    # load booster
    booster = None
    if os.path.exists(BOOSTER_PATH):
        booster = xgb.Booster()
        booster.load_model(BOOSTER_PATH)
    elif os.path.exists(CLASSIFIER_PKL):
        # fallback: sklearn-wrapped xgb
        clf = joblib.load(CLASSIFIER_PKL)
        try:
            booster = clf.get_booster()
        except Exception:
            booster = None

    # load feature lists
    features = []
    if os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON,"r") as f:
            features = json.load(f)
    features_t = []
    if os.path.exists(FEATURES_T_JSON):
        with open(FEATURES_T_JSON,"r") as f:
            features_t = json.load(f)

    # load background raw (for SHAP on raw -> we'll transform it)
    background = None
    if os.path.exists(BACKGROUND_PATH):
        background = pd.read_csv(BACKGROUND_PATH)

    error = None
    if preproc is None:
        error = "preprocessor missing"
    if booster is None:
        error = (error or "") + " ; booster missing or not extracted"

    return preproc, booster, (features, features_t, background), error

preproc, booster, feat_info, load_error = load_artifacts()
if load_error:
    st.error("Model artifacts load error: " + str(load_error))
    st.stop()

FEATURES, FEATURES_T, BACKGROUND = feat_info

# Build form UI same as before — ensure FEATURES list is used for inputs
# (Assume FEATURES variable is present; if not, fallback to FEATURES from file)
if not FEATURES:
    st.error("Original FEATURES list missing; add model/features.json to repo.")
    st.stop()

# helper to build df from inputs (as you had earlier)
def build_input_df(inputs_dict):
    row = {}
    for feat, raw in inputs_dict.items():
        v = str(raw).strip()
        if v == "" or v.lower() in ["nan", "none"]:
            row[feat] = np.nan
        else:
            try:
                row[feat] = float(v) if feat in NUMERIC_COLS else v
            except:
                row[feat] = v
    return pd.DataFrame([row], columns=FEATURES)

# Prediction function using preprocessor + booster
def predict_proba_from_raw(df_raw):
    # df_raw: DataFrame with columns = FEATURES
    X_trans = preproc.transform(df_raw)  # numpy array
    # ensure dtype float32 for xgboost
    dmat = xgb.DMatrix(X_trans.astype(np.float32))
    proba = booster.predict(dmat)
    # if booster returns shape (nclasses,) for multi-class, handle accordingly
    if proba.ndim == 2:
        # assume second column is positive class
        return float(proba[:,1][0])
    else:
        return float(proba[0])

# SHAP using TreeExplainer (fast)
def shap_for_row(df_raw):
    X_trans = preproc.transform(df_raw)
    # TreeExplainer expects booster and array/ DMatrix
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_trans)  # shape (n_samples, n_trans_features) or list
    # shap_vals may be array (binary) or list (multi-class)
    arr = None
    if isinstance(shap_vals, list):
        # for binary it may give list with one element
        arr = np.array(shap_vals[0])
    else:
        arr = np.array(shap_vals)
    # return 1D array for first row
    return arr[0]

# In your predict button handler:
# X_input = build_input_df(inputs)
# prob = predict_proba_from_raw(X_input)
# then shap_arr = shap_for_row(X_input)
# To label the transformed features use FEATURES_T (if present), else use indices f_0...
