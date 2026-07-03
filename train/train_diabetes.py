"""Train diabetes models: SVM (RBF), KNN, Keras MLP."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import ensure_dir, save_schema, save_sklearn, save_keras  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "diabetes.csv")
MODEL_DIR = os.path.join(ROOT, "models", "diabetes")

LABELS = {
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Plasma Glucose Concentration",
    "BloodPressure": "Diastolic Blood Pressure (mm Hg)",
    "SkinThickness": "Triceps Skin Fold Thickness (mm)",
    "Insulin": "2-Hour Serum Insulin (mu U/ml)",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}


def main():
    ensure_dir(MODEL_DIR)
    df = pd.read_csv(DATA_PATH)
    feature_names = [c for c in df.columns if c != "Outcome"]
    X = df[feature_names]
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics = {}
    fitted = {}

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_s, y_train)
    proba = svm.predict_proba(X_test_s)[:, 1]
    pred = svm.predict(X_test_s)
    metrics["svm"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["svm"] = svm

    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train_s, y_train)
    proba = knn.predict_proba(X_test_s)[:, 1]
    pred = knn.predict(X_test_s)
    metrics["knn"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["knn"] = knn

    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    mlp = keras.Sequential([
        keras.layers.Input(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    mlp.fit(
        X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=16,
        callbacks=[early_stop], verbose=0,
    )
    proba = mlp.predict(X_test_s, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    metrics["keras_ann"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted["keras_ann"] = mlp

    winner = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    print(f"[diabetes] winner={winner} metrics={metrics[winner]}")

    if winner == "keras_ann":
        save_keras(MODEL_DIR, fitted[winner], scaler)
    else:
        save_sklearn(MODEL_DIR, fitted[winner], scaler)
    save_schema(MODEL_DIR, feature_names, LABELS, X_train, winner, metrics)


if __name__ == "__main__":
    main()
