"""
predict.py
Inference: load trained model, preprocess single row or batch, return risk scores.
"""

import numpy as np
import pandas as pd
import os
import json
from models.nn_model import load_saved_model
from utils.data_utils import preprocess

MODEL_DIR = "models/saved"
ARTIFACT_DIR = "pipeline/artifacts"


def load_artifacts():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("No trained model found. Run `python train.py` first.")
    with open(meta_path) as f:
        meta = json.load(f)
    model = load_saved_model(os.path.join(MODEL_DIR, "credit_risk_model.keras"))
    return model, meta


def predict_single(input_dict: dict) -> dict:
    """
    Predict default probability for a single applicant.

    Args:
        input_dict: dict with keys matching dataset feature names

    Returns:
        dict with 'default_probability', 'decision', 'risk_level'
    """
    model, meta = load_artifacts()
    threshold = meta.get("optimal_threshold", 0.5)

    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess(df, artifact_dir=ARTIFACT_DIR, fit=False)

    prob = float(model.predict(X, verbose=0).flatten()[0])
    decision = "DENY" if prob >= threshold else "APPROVE"
    risk_level = (
        "Very High" if prob >= 0.75 else
        "High" if prob >= 0.55 else
        "Medium" if prob >= 0.35 else
        "Low"
    )

    return {
        "default_probability": round(prob, 4),
        "default_probability_pct": round(prob * 100, 2),
        "decision": decision,
        "risk_level": risk_level,
        "threshold_used": threshold,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict on a DataFrame of applicants."""
    model, meta = load_artifacts()
    threshold = meta.get("optimal_threshold", 0.5)

    X, _, _ = preprocess(df.copy(), artifact_dir=ARTIFACT_DIR, fit=False)
    probs = model.predict(X, verbose=0).flatten()

    result = df.copy()
    result["default_probability"] = np.round(probs, 4)
    result["decision"] = np.where(probs >= threshold, "DENY", "APPROVE")
    result["risk_level"] = pd.cut(
        probs, bins=[0, 0.35, 0.55, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Very High"]
    )
    return result


def load_training_history():
    path = os.path.join(MODEL_DIR, "training_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
