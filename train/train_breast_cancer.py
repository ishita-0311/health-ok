"""Train breast cancer models: SVC, Random Forest, Keras ANN."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import ensure_dir, save_schema, save_sklearn, save_keras  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "breast_cancer")
CSV_PATH = os.path.join(ROOT, "data", "raw", "breast_cancer.csv")


def main():
    ensure_dir(MODEL_DIR)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # target: 0=malignant, 1=benign in sklearn's dataset
    df.to_csv(CSV_PATH, index=False)

    feature_names = list(data.feature_names)
    X = df[feature_names]
    y = df["target"].astype(int)

    labels = {f: f.replace("_", " ").title() for f in feature_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics = {}
    fitted = {}

    svc = SVC(kernel="rbf", probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    proba = svc.predict_proba(X_test_s)[:, 1]
    pred = svc.predict(X_test_s)
    metrics["svm"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["svm"] = svc

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

    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    ann = keras.Sequential([
        keras.layers.Input(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ann.fit(
        X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=16,
        callbacks=[early_stop], verbose=0,
    )
    proba = ann.predict(X_test_s, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    metrics["keras_ann"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["keras_ann"] = ann

    winner = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    print(f"[breast_cancer] winner={winner} metrics={metrics[winner]}")

    if winner == "keras_ann":
        save_keras(MODEL_DIR, fitted[winner], scaler)
    else:
        save_sklearn(MODEL_DIR, fitted[winner], scaler)
    save_schema(MODEL_DIR, feature_names, labels, X_train, winner, metrics)


if __name__ == "__main__":
    main()
