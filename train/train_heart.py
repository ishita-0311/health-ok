"""Train heart disease models: Logistic Regression, Random Forest, XGBoost."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import ensure_dir, save_schema, save_sklearn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "heart.csv")
MODEL_DIR = os.path.join(ROOT, "models", "heart")

LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1=true, 0=false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=yes, 0=no)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (0-2)",
    "ca": "Number of Major Vessels Colored (0-4)",
    "thal": "Thalassemia (0-3)",
}


def main():
    ensure_dir(MODEL_DIR)
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics = {}
    fitted = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    proba = lr.predict_proba(X_test_s)[:, 1]
    pred = lr.predict(X_test_s)
    metrics["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["logistic_regression"] = lr

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = rf.predict(X_test)
    metrics["random_forest"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["random_forest"] = rf

    import xgboost as xgb

    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, tree_method="hist", device="cuda",
            eval_metric="logloss", random_state=42,
        )
        xgb_model.fit(X_train, y_train)
    except Exception:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, tree_method="hist",
            eval_metric="logloss", random_state=42,
        )
        xgb_model.fit(X_train, y_train)
    proba = xgb_model.predict_proba(X_test)[:, 1]
    pred = xgb_model.predict(X_test)
    metrics["xgboost"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["xgboost"] = xgb_model

    winner = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    print(f"[heart] winner={winner} metrics={metrics[winner]}")

    save_sklearn(MODEL_DIR, fitted[winner], scaler)
    save_schema(MODEL_DIR, feature_names, LABELS, X_train, winner, metrics)


if __name__ == "__main__":
    main()
