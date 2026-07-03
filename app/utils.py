"""Shared helpers for the Health-OK Streamlit app."""
import json
import os

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "raw")
ASSETS_DIR = os.path.join(ROOT, "assets", "images")

DISEASES = {
    "heart": {
        "title": "Heart Disease",
        "icon": "assets/images/heart.png",
        "csv": "heart.csv",
        "target": "target",
        "description": "Cardiovascular disease risk based on clinical measurements such as "
                        "blood pressure, cholesterol, and ECG results. Coronary heart "
                        "disease develops through atherosclerosis: lipid deposition and "
                        "chronic inflammation narrow the coronary arteries, and advanced "
                        "lesions can rupture and clot, triggering a heart attack.",
        "symptoms": ["Chest pain or discomfort", "Shortness of breath",
                     "Fatigue", "Irregular heartbeat", "Swelling in legs/ankles"],
        "risk_factors": ["High blood pressure", "High cholesterol", "Smoking",
                          "Diabetes", "Obesity", "Family history"],
        "prevalence": "Leading cause of death worldwide, responsible for 32% of all "
                      "global deaths in 2019; annual deaths projected to reach ~24 "
                      "million by 2030.",
        "prevention": "DASH-style diet, regular exercise, smoking cessation, and statin "
                      "therapy (guided by 10-year ASCVD risk) all measurably reduce risk.",
    },
    "diabetes": {
        "title": "Diabetes",
        "icon": "assets/images/diabetes.png",
        "csv": "diabetes.csv",
        "target": "Outcome",
        "description": "Risk of diabetes onset based on diagnostic measurements like "
                        "glucose level, BMI, insulin, and age. Type 2 diabetes stems from "
                        "two defects that need not arise together: peripheral insulin "
                        "resistance and progressive pancreatic beta-cell failure, driven "
                        "by both genetics and environment (diet, obesity, inactivity).",
        "symptoms": ["Frequent urination", "Increased thirst", "Unexplained weight loss",
                     "Fatigue", "Blurred vision"],
        "risk_factors": ["Obesity", "Sedentary lifestyle", "Family history",
                          "High blood pressure", "Age over 45"],
        "prevalence": "IDF estimates 537 million adults had diabetes in 2021, projected "
                      "to reach 643 million by 2030 and 783 million by 2045.",
        "prevention": "Lifestyle intervention (diet + exercise) and metformin both "
                      "reduce progression from prediabetes to Type 2 diabetes.",
    },
    "breast_cancer": {
        "title": "Breast Cancer",
        "icon": "assets/images/breast_cancer.png",
        "csv": "breast_cancer.csv",
        "target": "target",
        "description": "Classifies tumor biopsies as benign or malignant using cell "
                        "nucleus features (radius, texture, concavity, symmetry, etc.) "
                        "computed from digitized fine needle aspirate images. Malignant "
                        "nuclei tend to be larger, more irregular, and less symmetric.",
        "symptoms": ["Breast lump or thickening", "Change in size/shape",
                     "Skin dimpling", "Nipple discharge", "Breast pain"],
        "risk_factors": ["Age", "Family history / genetics (BRCA1/2)", "Early menstruation",
                          "Late menopause", "Obesity", "Alcohol use"],
        "prevalence": "An estimated 382,640 US women will be diagnosed in 2026 (321,910 "
                      "invasive + 60,730 in-situ), with ~42,140 deaths.",
        "prevention": "Survival is highly stage-dependent (Stage I ~99% vs. Stage IV "
                      "~31% five-year survival). Regular mammography screening lowers "
                      "mortality by ~26% through earlier detection.",
    },
    "ckd": {
        "title": "Chronic Kidney Disease",
        "icon": "assets/images/ckd.png",
        "csv": "ckd.csv",
        "target": "class",
        "description": "Detects presence of chronic kidney disease using blood test "
                        "values, urinalysis, and physical symptoms. Defined (KDIGO) as "
                        "reduced kidney function (eGFR < 60 mL/min/1.73m^2) and/or "
                        "albuminuria persisting for 3+ months.",
        "symptoms": ["Swelling in legs/ankles", "Fatigue", "Changes in urination",
                     "Nausea", "Loss of appetite", "Muscle cramps"],
        "risk_factors": ["Diabetes", "Hypertension", "Family history",
                          "Age over 60", "Obesity", "Heart disease"],
        "prevalence": "788 million adults were living with CKD in 2023, more than "
                      "double the 378 million recorded in 1990.",
        "prevention": "Onset and progression are often preventable through blood "
                      "pressure and glycemic control; dialysis patients have a five-year "
                      "survival rate below 50%, making prevention especially valuable.",
    },
    "parkinsons": {
        "title": "Parkinson's Disease",
        "icon": "assets/images/parkinsons.png",
        "csv": "parkinsons.csv",
        "target": "status",
        "description": "Detects Parkinson's disease from biomedical voice measurements "
                        "(jitter, shimmer, harmonic-to-noise ratio, etc.). Caused by "
                        "progressive degeneration of dopaminergic neurons in the "
                        "substantia nigra; ~90% of patients develop vocal impairment, "
                        "making speech a practical, non-invasive early signal.",
        "symptoms": ["Tremor", "Slowed movement (bradykinesia)", "Rigid muscles",
                     "Impaired posture/balance", "Speech changes"],
        "risk_factors": ["Age (60+)", "Sex (more common in men)", "Genetics",
                          "Exposure to toxins", "Head injury history"],
        "prevalence": "Global prevalence reached ~10 million cases in 2023 and is "
                      "expected to rise 50% by 2030 as populations age.",
        "prevention": "No established primary prevention. Motor symptoms only appear "
                      "after 60-80% of dopamine-producing neurons are already lost, so "
                      "early detection (e.g. via voice biomarkers) matters greatly.",
    },
}


def load_schema(disease_key):
    path = os.path.join(MODELS_DIR, disease_key, "schema.json")
    with open(path) as fh:
        return json.load(fh)


def load_model_and_scaler(disease_key, schema):
    model_dir = os.path.join(MODELS_DIR, disease_key)
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    winner = schema["winner"]
    if winner == "keras_ann":
        from tensorflow import keras
        model = keras.models.load_model(os.path.join(model_dir, "model.keras"))
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model, scaler


NEEDS_SCALING = {"keras_ann", "svm", "knn", "logistic_regression"}


def predict_proba(disease_key, schema, model, scaler, feature_values):
    """feature_values: dict feature_name -> value, in any order."""
    order = schema["feature_order"]
    x = np.array([[feature_values[f] for f in order]], dtype=float)
    winner = schema["winner"]
    if winner in NEEDS_SCALING:
        x = scaler.transform(x)
    if winner == "keras_ann":
        proba = float(model.predict(x, verbose=0).ravel()[0])
    else:
        proba = float(model.predict_proba(x)[0, 1])
    return proba
