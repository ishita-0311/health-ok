"""Train Chronic Kidney Disease models: Decision Tree, LightGBM.

Data source: UCI Chronic Kidney Disease dataset (400 patients, Tamil Nadu,
India hospital), sourced from the public GitHub mirror
https://raw.githubusercontent.com/ArjunAnilPillai/Chronic-Kidney-Disease-dataset/master/kidney_disease.csv
which reproduces the original UCI archive
(https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) CSV export.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from common import ensure_dir, save_schema, save_sklearn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "ckd")
CSV_PATH = os.path.join(ROOT, "data", "raw", "ckd.csv")

NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]
BINARY_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
# Maps each binary column's "positive" (=> 1) string value
BINARY_POSITIVE = {
    "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
    "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes",
}

LABELS = {
    "age": "Age (years)",
    "bp": "Blood Pressure (mm Hg)",
    "sg": "Specific Gravity of Urine",
    "al": "Albumin Level (0-5)",
    "su": "Sugar Level (0-5)",
    "bgr": "Blood Glucose Random (mg/dl)",
    "bu": "Blood Urea (mg/dl)",
    "sc": "Serum Creatinine (mg/dl)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (g/dl)",
    "pcv": "Packed Cell Volume (%)",
    "wc": "White Blood Cell Count (cells/cmm)",
    "rc": "Red Blood Cell Count (millions/cmm)",
    "rbc": "Red Blood Cells Abnormal (1=yes, 0=no)",
    "pc": "Pus Cell Abnormal (1=yes, 0=no)",
    "pcc": "Pus Cell Clumps Present (1=yes, 0=no)",
    "ba": "Bacteria Present (1=yes, 0=no)",
    "htn": "Hypertension (1=yes, 0=no)",
    "dm": "Diabetes Mellitus (1=yes, 0=no)",
    "cad": "Coronary Artery Disease (1=yes, 0=no)",
    "appet": "Poor Appetite (1=yes, 0=no)",
    "pe": "Pedal Edema (1=yes, 0=no)",
    "ane": "Anemia (1=yes, 0=no)",
}


def load_and_clean():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["id"])
    df.columns = [c.strip() for c in df.columns]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    for col in BINARY_COLS:
        cleaned = df[col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        df[col] = (cleaned == BINARY_POSITIVE[col]).astype(float)
        df.loc[cleaned.isna(), col] = np.nan

    df["class"] = df["class"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})

    feature_cols = NUMERIC_COLS + BINARY_COLS
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df, feature_cols


def main():
    ensure_dir(MODEL_DIR)
    df, feature_names = load_and_clean()

    X = df[feature_names]
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)

    metrics = {}
    fitted = {}

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    proba = dt.predict_proba(X_test)[:, 1]
    pred = dt.predict(X_test)
    metrics["decision_tree"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["decision_tree"] = dt

    import lightgbm as lgb

    try:
        lgbm = lgb.LGBMClassifier(n_estimators=300, device="gpu", random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
    except Exception:
        lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
    proba = lgbm.predict_proba(X_test)[:, 1]
    pred = lgbm.predict(X_test)
    metrics["lightgbm"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["lightgbm"] = lgbm

    winner = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    print(f"[ckd] winner={winner} metrics={metrics[winner]}")

    save_sklearn(MODEL_DIR, fitted[winner], scaler)
    save_schema(MODEL_DIR, feature_names, LABELS, X_train, winner, metrics)


if __name__ == "__main__":
    main()
