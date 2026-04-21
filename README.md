# 🏦 CreditIQ — Neural Network Credit Risk Assessment

> End-to-end **Credit Risk Classification** system using a Deep Neural Network with residual connections, Keras Tuner hyperparameter search, SMOTE oversampling, and a full Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

---

## 🧠 Model Architecture

```
Input(n_features)
  ↓
Dense(256) → BatchNorm → Swish → Dropout
  ↓
ResidualBlock(128)   ← Dense + BN + Swish + skip connection
  ↓
ResidualBlock(64)    ← Dense + BN + Swish + skip connection
  ↓
Dense(1, sigmoid)   → default probability ∈ [0, 1]
```

**Key techniques:**
- **Huber-inspired Binary Crossentropy** loss
- **Swish activations** — smoother gradients for tabular data
- **Residual connections** — prevents vanishing gradients in deep networks
- **Batch Normalization** — stable training across varying feature scales
- **SMOTE** — synthetic minority oversampling for class imbalance
- **Keras Tuner HyperBand** — efficient hyperparameter search
- **Optimal threshold selection** — maximises F1 on validation set

---

## 🚀 Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/creditiq.git
cd creditiq
pip install -r requirements.txt
```

### Train

```bash
# Basic training
python train.py

# With SMOTE + hyperparameter search
python train.py --smote --tune --max_trials 15 --epochs 100

# Larger dataset
python train.py --n_samples 50000 --smote --epochs 150
```

| Flag | Default | Description |
|------|---------|-------------|
| `--smote` | off | Apply SMOTE oversampling |
| `--tune` | off | Run Keras Tuner HyperBand search |
| `--max_trials` | 10 | Number of tuner trials |
| `--epochs` | 100 | Max training epochs |
| `--batch_size` | 256 | Mini-batch size |
| `--n_samples` | 10000 | Synthetic dataset size |

### Launch Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Features (19 total)

| Group | Features |
|-------|----------|
| Demographics | age, num_dependents |
| Financial | income, debt_to_income, has_mortgage |
| Employment | employment_years, employment_type |
| Loan | loan_amount, loan_term, interest_rate, loan_purpose |
| Credit History | credit_score, num_credit_lines, num_late_payments |
| Education | education |
| **Engineered** | loan_to_income, monthly_payment, payment_to_income, credit_score_bucket |

---

## 📁 Project Structure

```
creditiq/
├── app.py                    # Streamlit dashboard (5 tabs)
├── train.py                  # Training pipeline CLI
├── predict.py                # Single + batch inference
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── nn_model.py           # DNN + residual blocks + Keras Tuner
├── utils/
│   ├── __init__.py
│   ├── data_utils.py         # Data generation + preprocessing pipeline
│   └── metrics.py            # Classification metrics
├── models/saved/             # Auto-created on training
│   ├── credit_risk_model.keras
│   ├── metadata.json
│   └── training_history.csv
├── pipeline/
│   └── artifacts/            # Scalers + encoders (auto-created)
└── .streamlit/
    └── config.toml
```

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Data Explorer | EDA charts, correlations, distribution plots |
| 🧠 Train | Launch training from UI, live logs, loss/AUC curves |
| 🔍 Predict | Interactive single-applicant risk assessment with gauge |
| 📋 Batch Score | Score 50–500 applicants, pie + histogram charts |
| 📈 Analysis | Confusion matrix, AUC curve, training summary |

