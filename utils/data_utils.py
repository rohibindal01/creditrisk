"""
utils/data_utils.py
Synthetic credit risk dataset generation + full preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os


# ──────────────────────────────────────────────
# Synthetic Dataset Generator
# ──────────────────────────────────────────────

def generate_credit_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit risk dataset.
    Target: 'default' (1 = defaulted, 0 = repaid)
    """
    rng = np.random.RandomState(random_state)

    age = rng.randint(21, 70, n_samples)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n_samples).round(2)
    employment_years = np.clip(rng.normal(8, 5, n_samples), 0, 40).round(1)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n_samples).round(2)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples)
    interest_rate = np.clip(rng.normal(12, 4, n_samples), 3, 30).round(2)
    credit_score = np.clip(rng.normal(670, 90, n_samples), 300, 850).round(0).astype(int)
    num_credit_lines = rng.randint(1, 15, n_samples)
    num_late_payments = rng.poisson(1.2, n_samples)
    debt_to_income = np.clip(rng.normal(0.35, 0.15, n_samples), 0.05, 0.95).round(3)
    has_mortgage = rng.choice([0, 1], n_samples, p=[0.55, 0.45])
    num_dependents = rng.choice([0, 1, 2, 3, 4], n_samples, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    education = rng.choice(["High School", "Bachelor", "Master", "PhD"], n_samples,
                            p=[0.30, 0.40, 0.22, 0.08])
    employment_type = rng.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], n_samples,
                                  p=[0.55, 0.25, 0.15, 0.05])
    loan_purpose = rng.choice(["Home", "Car", "Education", "Medical", "Business", "Personal"],
                               n_samples, p=[0.20, 0.18, 0.15, 0.10, 0.17, 0.20])

    # Realistic default probability based on features
    log_odds = (
        -3.5
        + 0.8 * (credit_score < 600).astype(float)
        - 0.5 * (credit_score > 750).astype(float)
        + 1.2 * (debt_to_income > 0.6).astype(float)
        + 0.6 * (num_late_payments > 2).astype(float)
        - 0.4 * (income > 80000).astype(float)
        + 0.3 * (employment_type == "Unemployed").astype(float)
        + 0.2 * (loan_amount / income > 1.5)
        - 0.2 * (employment_years > 10).astype(float)
        + rng.normal(0, 0.3, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(size=n_samples) < prob_default).astype(int)

    # Inject ~5% missing values in some columns
    for col_arr in [income, employment_years, credit_score, debt_to_income]:
        mask = rng.rand(n_samples) < 0.03
        col_arr = col_arr.astype(float)
        col_arr[mask] = np.nan

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "credit_score": credit_score.astype(float),
        "num_credit_lines": num_credit_lines,
        "num_late_payments": num_late_payments,
        "debt_to_income": debt_to_income,
        "has_mortgage": has_mortgage,
        "num_dependents": num_dependents,
        "education": education,
        "employment_type": employment_type,
        "loan_purpose": loan_purpose,
        "default": default,
    })
    return df


# ──────────────────────────────────────────────
# Preprocessing Pipeline
# ──────────────────────────────────────────────

NUMERIC_COLS = [
    "age", "income", "employment_years", "loan_amount",
    "loan_term", "interest_rate", "credit_score",
    "num_credit_lines", "num_late_payments", "debt_to_income",
    "has_mortgage", "num_dependents",
]

CATEGORICAL_COLS = ["education", "employment_type", "loan_purpose"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1e-6)
    df["monthly_payment"] = (df["loan_amount"] * df["interest_rate"] / 100) / df["loan_term"]
    df["payment_to_income"] = df["monthly_payment"] / (df["income"] / 12 + 1e-6)
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 900],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    return df


def preprocess(
    df: pd.DataFrame,
    artifact_dir: str = "pipeline/artifacts",
    fit: bool = True,
):
    """
    Full preprocessing: imputation → feature engineering →
    encoding → scaling → optional SMOTE.

    Returns X (np.ndarray), y (np.ndarray), feature_names (list)
    """
    df = engineer_features(df)
    y = df["default"].values if "default" in df.columns else None
    df = df.drop(columns=["default"], errors="ignore")

    extra_numeric = ["loan_to_income", "monthly_payment", "payment_to_income", "credit_score_bucket"]
    all_numeric = NUMERIC_COLS + extra_numeric

    os.makedirs(artifact_dir, exist_ok=True)

    # ── Impute numerics ──────────────────────────────────────
    num_imputer_path = os.path.join(artifact_dir, "num_imputer.pkl")
    if fit:
        num_imputer = SimpleImputer(strategy="median")
        df[all_numeric] = num_imputer.fit_transform(df[all_numeric])
        joblib.dump(num_imputer, num_imputer_path)
    else:
        num_imputer = joblib.load(num_imputer_path)
        df[all_numeric] = num_imputer.transform(df[all_numeric])

    # ── Encode categoricals ──────────────────────────────────
    encoders = {}
    encoded_cols = []
    for col in CATEGORICAL_COLS:
        enc_path = os.path.join(artifact_dir, f"le_{col}.pkl")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, enc_path)
        else:
            le = joblib.load(enc_path)
            df[col] = le.transform(df[col].astype(str))
        encoded_cols.append(col)

    feature_names = all_numeric + encoded_cols
    X = df[feature_names].values.astype(np.float32)

    # ── Scale ────────────────────────────────────────────────
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

    return X, y, feature_names


def apply_smote(X_train, y_train, random_state=42):
    """Balance classes with SMOTE."""
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
