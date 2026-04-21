"""
train.py
Full training pipeline for Credit Risk Classification.

Usage:
    python train.py [--tune] [--epochs 100] [--batch_size 256] [--smote]
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from utils.data_utils import (
    generate_credit_data, preprocess, split_data, apply_smote
)
from utils.metrics import compute_all_metrics, find_optimal_threshold
from models.nn_model import (
    build_model, train_model, save_model,
    run_hyperparameter_search
)

MODEL_DIR = "models/saved"
ARTIFACT_DIR = "pipeline/artifacts"


def parse_args():
    p = argparse.ArgumentParser(description="Credit Risk DNN Training")
    p.add_argument("--tune", action="store_true", help="Run Keras Tuner hyperparameter search")
    p.add_argument("--max_trials", type=int, default=10, help="Tuner max trials")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    p.add_argument("--n_samples", type=int, default=10000, help="Dataset size")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print("  Credit Risk Classification — Training Pipeline")
    print(f"{'='*60}\n")

    # 1. Generate data
    print("[1/5] Generating synthetic credit dataset...")
    df = generate_credit_data(n_samples=args.n_samples)
    print(f"      {len(df)} samples | Default rate: {df['default'].mean()*100:.1f}%")

    # 2. Preprocess
    print("[2/5] Preprocessing & feature engineering...")
    X, y, feature_names = preprocess(df, artifact_dir=ARTIFACT_DIR, fit=True)
    print(f"      Feature matrix: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if args.smote:
        print("      Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)
        print(f"      After SMOTE: {len(X_train)} samples | "
              f"Balance: {y_train.mean()*100:.1f}% positive")

    n_features = X_train.shape[1]

    # 3. Hyperparameter tuning (optional)
    best_params = {}
    if args.tune:
        print(f"[3/5] Running HyperBand search ({args.max_trials} trials)...")
        best_hps, tuner = run_hyperparameter_search(
            X_train, y_train, X_val, y_val,
            n_features=n_features,
            max_trials=args.max_trials,
        )
        best_params = best_hps.values
        print(f"      Best hyperparameters: {best_params}")
        model = tuner.hypermodel.build(best_hps)
    else:
        print("[3/5] Using default architecture (skip --tune to run search)...")
        model = build_model(n_features=n_features)

    model.summary()

    # 4. Train
    print("\n[4/5] Training model...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        model_dir=MODEL_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 5. Evaluate
    print("[5/5] Evaluating on test set...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    # Find optimal decision threshold
    optimal_t, best_f1 = find_optimal_threshold(y_val, model.predict(X_val, verbose=0).flatten())
    print(f"      Optimal threshold: {optimal_t} (F1={best_f1})")

    metrics = compute_all_metrics(y_test, y_pred_prob, threshold=optimal_t)
    print("\n  ── Test Set Metrics ──")
    for k, v in metrics.items():
        print(f"     {k}: {v}")

    # Save
    model_path = os.path.join(MODEL_DIR, "credit_risk_model.keras")
    save_model(model, model_path)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)

    metadata = {
        "n_features": n_features,
        "feature_names": feature_names,
        "optimal_threshold": optimal_t,
        "best_params": best_params,
        "test_metrics": {k: v for k, v in metrics.items() if k not in ["TP","TN","FP","FN"]},
        "confusion_matrix": {"TP": metrics["TP"], "TN": metrics["TN"],
                             "FP": metrics["FP"], "FN": metrics["FN"]},
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epochs_run": len(history.history["loss"]),
        "smote_used": args.smote,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Model → {model_path}")
    print(f"  ✓ Metadata → {MODEL_DIR}/metadata.json")
    print(f"\n{'='*60}")
    print("  Done! Run `streamlit run app.py` to launch the dashboard.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
