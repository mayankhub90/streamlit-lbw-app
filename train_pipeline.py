# train_pipeline.py
"""
Train a preprocessing + XGBoost pipeline and save:
 - model/pipeline.pkl
 - model/background.csv (representative raw sample for SHAP)

If 'data.csv' is not found in cwd, this script will generate a synthetic demo dataset
named data.csv so you can test end-to-end.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

def make_synthetic(n=2000, seed=42):
    np.random.seed(seed)
    def maybe_cat():
        return np.random.choice(["A","B","C","Unknown"], size=n)
    demo = pd.DataFrame({
        "Beneficiary age": np.random.randint(16,45,size=n),
        "measured_HB_risk_bin": np.random.choice([0,1], size=n),
        "Child order/parity": np.random.randint(1,5,size=n),
        "Number of living child at now": np.random.randint(0,5,size=n),
        "MonthConception": np.random.randint(1,9,size=n),
        "BMI_PW1_Prog": np.random.normal(22,3,size=n).clip(12,40),
        "BMI_PW2_Prog": np.random.normal(22,3,size=n).clip(12,40),
        "BMI_PW3_Prog": np.random.normal(22,3,size=n).clip(12,40),
        "BMI_PW4_Prog": np.random.normal(22,3,size=n).clip(12,40),
        "consume_tobacco": np.random.choice(["Yes","No"], size=n),
        "Status of current chewing of tobacco": np.random.choice(["Yes","No"], size=n),
        "consume_alcohol": np.random.choice(["Yes","No"], size=n),
        "RegistrationBucket": maybe_cat(),
        "counselling_gap_days": np.random.randint(0,200,size=n),
        "ANCBucket": maybe_cat(),
        "LMPtoINST1": np.random.randint(0,200,size=n),
        "LMPtoINST2": np.random.randint(0,200,size=n),
        "LMPtoINST3": np.random.randint(0,200,size=n),
        "No of ANCs completed": np.random.randint(0,5,size=n),
        "Service received during last ANC: TT Injection given": np.random.choice(["Yes","No"], size=n),
        "No. of IFA tablets received/procured in last one month_log1p": np.log1p(np.random.randint(0,120,size=n)),
        "No. of calcium tablets consumed in last one month_log1p": np.log1p(np.random.randint(0,120,size=n)),
        "Food_Groups_Category": np.random.choice(["Low","Medium","High"], size=n),
        "Household_Assets_Score_log1p": np.log1p(np.random.randint(0,50,size=n)),
        "toilet_type_clean": np.random.choice(["Yes","No"], size=n),
        "water_source_clean": np.random.choice(["Yes","No"], size=n),
        "education_clean": np.random.choice(["NoSchool","Primary","Secondary","Higher"], size=n),
        "Social_Media_Category": np.random.choice(["None","Low","High"], size=n),
        "Registered for cash transfer scheme: JSY": np.random.choice(["Yes","No"], size=n),
        "Registered for cash transfer scheme: RAJHSRI": np.random.choice(["Yes","No"], size=n),
        "PMMVY-Number of installment received": np.random.randint(0,3,size=n),
        "JSY-Number of installment received": np.random.randint(0,3,size=n),
    })
    demo['target'] = ((demo["Beneficiary age"]<20).astype(int) + (demo["No of ANCs completed"]==0).astype(int) + (demo["Household_Assets_Score_log1p"]<2).astype(int)) > 0
    demo['target'] = demo['target'].astype(int)
    return demo

def train(data_path="data.csv", model_out="model/pipeline.pkl", background_out="model/background.csv"):
    # load or create data
    if not os.path.exists(data_path):
        print(f"{data_path} not found. Creating a synthetic demo dataset 'data.csv' (for dev/testing).")
        demo = make_synthetic()
        demo.to_csv(data_path, index=False)
        df = demo
    else:
        df = pd.read_csv(data_path)

    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data.csv: {missing}")

    X = df[FEATURES].copy()
    y = df["target"].astype(int)

    numeric_cols = [c for c in FEATURES if c in NUMERIC_COLS]
    categorical_cols = [c for c in FEATURES if c not in NUMERIC_COLS]

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=2,
        random_state=42
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print("Validation AUC:", auc)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved pipeline to {model_out}")

    # save a small background sample for SHAP (raw features)
    bg = X_train.sample(n=min(100, len(X_train)), random_state=42)
    bg.to_csv(background_out, index=False)
    print(f"Saved background sample to {background_out}")

if __name__ == "__main__":
    train()
