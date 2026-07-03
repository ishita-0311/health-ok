"""Shared helpers for training scripts."""
import json
import os

import joblib
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_schema(model_dir, feature_names, labels, df, winner, metrics):
    """Build and save schema.json describing inputs and model metadata.

    feature_names: ordered list of column names (as fed to the model)
    labels: dict feature -> human readable label
    df: dataframe (feature columns only) used to derive min/max/default
    winner: string model type key
    metrics: dict of metrics for all trained models
    """
    features = []
    for f in feature_names:
        col = df[f]
        lo = float(np.nanpercentile(col, 5))
        hi = float(np.nanpercentile(col, 95))
        if lo == hi:
            lo, hi = float(col.min()), float(col.max())
        default = float(col.mean())
        features.append({
            "name": f,
            "label": labels.get(f, f),
            "min": round(lo, 3),
            "max": round(hi, 3),
            "default": round(default, 3),
        })
    schema = {
        "features": features,
        "feature_order": feature_names,
        "winner": winner,
        "metrics": metrics,
    }
    with open(os.path.join(model_dir, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=2)


def save_sklearn(model_dir, model, scaler):
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))


def save_keras(model_dir, model, scaler):
    model.save(os.path.join(model_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
