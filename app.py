"""
app.py — Credit Risk Assessment Dashboard
Streamlit frontend: EDA, training, single prediction, batch scoring, model analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, json

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ · Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f8f7f4; }

.brand {
    font-family: 'DM Mono', monospace;
    font-size: 2rem; font-weight: 500;
    color: #1a1a2e; letter-spacing: -0.03em;
}
.brand span { color: #e63946; }
.tagline { color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem; }

.kpi {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-family: 'DM Mono', monospace; font-size: 1.7rem; color: #111827; font-weight: 500; }
.kpi-value.red   { color: #e63946; }
.kpi-value.green { color: #059669; }
.kpi-value.amber { color: #d97706; }

.section { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #6b7280;
           text-transform: uppercase; letter-spacing: 0.12em;
           border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; margin-bottom: 1rem; }

.risk-badge {
    display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
    font-family: 'DM Mono', monospace; font-size: 0.78rem; font-weight: 500;
}
.badge-low    { background: #d1fae5; color: #065f46; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-high   { background: #fee2e2; color: #991b1b; }
.badge-vhigh  { background: #fce7f3; color: #9d174d; }

.approve-box {
    background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.deny-box {
    background: #fff1f2; border: 1px solid #fca5a5; border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.decision-label { font-family: 'DM Mono', monospace; font-size: 1.4rem; font-weight: 500; }

div[data-testid="stSidebar"] { background: #1a1a2e; }
div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label { color: #9ca3af !important; }

.stButton > button {
    background: #e63946; color: white; border: none;
    border-radius: 10px; font-family: 'DM Mono', monospace;
    font-size: 0.82rem; padding: 0.6rem 1.2rem; width: 100%;
}
.stButton > button:hover { background: #c1121f; }

.info { background: white; border: 1px solid #e5e7eb; border-left: 4px solid #3b82f6;
        border-radius: 8px; padding: 0.9rem 1.1rem; font-size: 0.87rem; color: #374151; }
.warn { background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 0.9rem 1.1rem; font-size: 0.87rem; color: #78350f; }
</style>
""", unsafe_allow_html=True)

PLOT_CFG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#374151"),
    xaxis=dict(gridcolor="#f3f4f6", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#f3f4f6", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)
MODEL_DIR = "models/saved"

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="brand">Credit<span>IQ</span></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Neural Network Credit Risk Assessment · TensorFlow/Keras</div>',
            unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    n_samples = st.select_slider("Dataset Size", [5000, 10000, 20000, 50000], value=10000)
    use_smote = st.checkbox("Apply SMOTE Oversampling", value=True)
    run_tuner = st.checkbox("Run Hyperparameter Search", value=False)
    max_trials = st.slider("Tuner Trials", 5, 30, 10, disabled=not run_tuner)
    epochs = st.slider("Max Epochs", 20, 200, 80)
    batch_size = st.selectbox("Batch Size", [128, 256, 512], index=1)
    st.markdown("---")
    train_btn = st.button("🚀 Train Model")

    model_exists = os.path.exists(os.path.join(MODEL_DIR, "credit_risk_model.keras"))
    st.markdown("---")
    if model_exists:
        st.success("✅ Trained model ready")
    else:
        st.warning("⚠️ Train a model first")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "🧠 Train", "🔍 Predict", "📋 Batch Score", "📈 Analysis"
])

@st.cache_data
def get_sample_data(n):
    from utils.data_utils import generate_credit_data
    return generate_credit_data(n_samples=n)

# ══════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section">Dataset Overview</div>', unsafe_allow_html=True)
    df = get_sample_data(n_samples)

    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("Total Samples", f"{len(df):,}", ""),
        ("Default Rate", f"{df['default'].mean()*100:.1f}%", "red"),
        ("Avg Credit Score", f"{df['credit_score'].mean():.0f}", ""),
        ("Avg Income", f"${df['income'].mean()/1000:.0f}K", "green"),
        ("Avg Loan Amount", f"${df['loan_amount'].mean()/1000:.0f}K", ""),
    ]
    for col, (label, val, cls) in zip([c1,c2,c3,c4,c5], stats):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                        f'<div class="kpi-value {cls}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(df, x="credit_score", color="default",
                           color_discrete_map={0: "#059669", 1: "#e63946"},
                           nbins=40, barmode="overlay", opacity=0.7,
                           labels={"default": "Defaulted", "credit_score": "Credit Score"},
                           title="Credit Score Distribution by Default Status")
        fig.update_layout(**PLOT_CFG, height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.box(df, x="employment_type", y="debt_to_income", color="default",
                     color_discrete_map={0: "#059669", 1: "#e63946"},
                     labels={"default": "Defaulted", "debt_to_income": "Debt-to-Income"},
                     title="Debt-to-Income by Employment Type & Default")
        fig.update_layout(**PLOT_CFG, height=300)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        default_by_purpose = df.groupby("loan_purpose")["default"].mean().sort_values(ascending=False)
        fig = px.bar(x=default_by_purpose.index, y=default_by_purpose.values * 100,
                     labels={"x": "Loan Purpose", "y": "Default Rate (%)"},
                     title="Default Rate by Loan Purpose",
                     color=default_by_purpose.values,
                     color_continuous_scale=[[0,"#059669"],[0.5,"#f59e0b"],[1,"#e63946"]])
        fig.update_layout(**PLOT_CFG, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        numeric_cols = ["credit_score", "income", "debt_to_income",
                        "loan_amount", "num_late_payments", "interest_rate"]
        corr = df[numeric_cols + ["default"]].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature Correlation Matrix", aspect="auto")
        fig.update_layout(**PLOT_CFG, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Sample Data (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Train
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section">Model Training</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Pipeline Steps**")
        steps = [
            ("1", "Generate synthetic credit dataset"),
            ("2", "Feature engineering (15 → 19 features)"),
            ("3", "Median imputation for missing values"),
            ("4", "Label encode categoricals"),
            ("5", "StandardScaler normalization"),
            ("6", "SMOTE oversampling (optional)"),
            ("7", "HyperBand search (optional)"),
            ("8", "Train DNN with class weights"),
            ("9", "Optimal threshold selection"),
            ("10", "Evaluate: AUC, F1, MAPE, etc."),
        ]
        for num, step in steps:
            st.markdown(f"`{num:>2}` &nbsp; {step}")

    with col_b:
        st.markdown("**Model Architecture**")
        st.code("""
Input(n_features)
  ↓
Dense(256) → BatchNorm → Swish → Dropout
  ↓
ResidualBlock(128)    ← skip connection
  BatchNorm → Swish → Dropout
  ↓
ResidualBlock(64)     ← skip connection
  BatchNorm → Swish → Dropout
  ↓
Dense(1, sigmoid)     ← default probability
""", language="text")
        st.markdown("""
| Detail | Value |
|--------|-------|
| Loss | Binary Crossentropy |
| Optimizer | Adam + gradient clip |
| Imbalance | Class weights + SMOTE |
| Stopping | EarlyStopping (AUC) |
| Threshold | F1-optimal on val set |
""")

    if train_btn:
        st.markdown("---")
        try:
            import subprocess, sys
            cmd = [sys.executable, "train.py",
                   "--epochs", str(epochs),
                   "--batch_size", str(batch_size),
                   "--n_samples", str(n_samples)]
            if use_smote: cmd.append("--smote")
            if run_tuner:
                cmd += ["--tune", "--max_trials", str(max_trials)]

            with st.spinner("Training in progress..."):
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        cwd=os.path.dirname(os.path.abspath(__file__)) or ".")

            if result.returncode == 0:
                st.success("✅ Training complete!")
                with st.expander("📄 Training Log", expanded=True):
                    st.code(result.stdout)

                hist_df = pd.read_csv(os.path.join(MODEL_DIR, "training_history.csv")) \
                    if os.path.exists(os.path.join(MODEL_DIR, "training_history.csv")) else None
                if hist_df is not None:
                    fig = make_subplots(1, 2, subplot_titles=["Loss", "AUC"])
                    fig.add_trace(go.Scatter(y=hist_df["loss"], name="Train", line=dict(color="#e63946")), 1, 1)
                    if "val_loss" in hist_df: fig.add_trace(go.Scatter(y=hist_df["val_loss"], name="Val", line=dict(color="#3b82f6")), 1, 1)
                    if "auc" in hist_df: fig.add_trace(go.Scatter(y=hist_df["auc"], name="Train AUC", line=dict(color="#059669"), showlegend=False), 1, 2)
                    if "val_auc" in hist_df: fig.add_trace(go.Scatter(y=hist_df["val_auc"], name="Val AUC", line=dict(color="#7c3aed"), showlegend=False), 1, 2)
                    fig.update_layout(**PLOT_CFG, height=280)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Training failed.")
                st.code(result.stderr)
        except Exception as e:
            st.error(str(e))
            st.exception(e)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Single Prediction
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section">Applicant Risk Assessment</div>', unsafe_allow_html=True)

    if not model_exists:
        st.markdown('<div class="warn">⚠️ No model found — please train first.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("Fill in the applicant details below:")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 21, 70, 35)
            income = st.number_input("Annual Income ($)", 10000, 500000, 65000, step=1000)
            employment_years = st.slider("Employment Years", 0.0, 40.0, 5.0, 0.5)
            employment_type = st.selectbox("Employment Type",
                                           ["Salaried", "Self-Employed", "Business", "Unemployed"])
            education = st.selectbox("Education",
                                     ["High School", "Bachelor", "Master", "PhD"])

        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000, step=500)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
            interest_rate = st.slider("Interest Rate (%)", 3.0, 30.0, 12.0, 0.5)
            loan_purpose = st.selectbox("Loan Purpose",
                                        ["Home", "Car", "Education", "Medical", "Business", "Personal"])

        with col3:
            credit_score = st.slider("Credit Score", 300, 850, 680)
            num_credit_lines = st.slider("Number of Credit Lines", 1, 15, 4)
            num_late_payments = st.slider("Late Payments (lifetime)", 0, 10, 1)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.05, 0.95, 0.35, 0.01)
            has_mortgage = st.checkbox("Has Mortgage", value=False)
            num_dependents = st.slider("Dependents", 0, 4, 1)

        predict_btn = st.button("⚡ Assess Risk")

        if predict_btn:
            try:
                from predict import predict_single
                applicant = {
                    "age": age, "income": income, "employment_years": employment_years,
                    "loan_amount": loan_amount, "loan_term": loan_term,
                    "interest_rate": interest_rate, "credit_score": float(credit_score),
                    "num_credit_lines": num_credit_lines, "num_late_payments": num_late_payments,
                    "debt_to_income": debt_to_income, "has_mortgage": int(has_mortgage),
                    "num_dependents": num_dependents, "education": education,
                    "employment_type": employment_type, "loan_purpose": loan_purpose,
                }

                result = predict_single(applicant)
                prob = result["default_probability_pct"]
                decision = result["decision"]
                risk = result["risk_level"]

                st.markdown("<br>", unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)

                with r1:
                    box_cls = "approve-box" if decision == "APPROVE" else "deny-box"
                    icon = "✅" if decision == "APPROVE" else "❌"
                    color = "#059669" if decision == "APPROVE" else "#e63946"
                    st.markdown(
                        f'<div class="{box_cls}"><div class="decision-label" style="color:{color}">'
                        f'{icon} {decision}</div></div>', unsafe_allow_html=True
                    )

                with r2:
                    gauge_color = "#e63946" if prob > 60 else "#f59e0b" if prob > 35 else "#059669"
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob,
                        title={"text": "Default Probability (%)"},
                        number={"suffix": "%", "font": {"size": 28}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": gauge_color},
                            "steps": [
                                {"range": [0, 35], "color": "#d1fae5"},
                                {"range": [35, 55], "color": "#fef3c7"},
                                {"range": [55, 75], "color": "#fee2e2"},
                                {"range": [75, 100], "color": "#fce7f3"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 2},
                                          "value": result["threshold_used"] * 100},
                        }
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                                      margin=dict(l=20, r=20, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                with r3:
                    badge_map = {"Low": "badge-low", "Medium": "badge-medium",
                                 "High": "badge-high", "Very High": "badge-vhigh"}
                    st.markdown(f"""
                    <div class="kpi" style="margin-top:0.5rem">
                        <div class="kpi-label">Risk Level</div>
                        <div style="margin-top:0.5rem">
                            <span class="risk-badge {badge_map.get(risk,'badge-medium')}">{risk}</span>
                        </div>
                        <br>
                        <div class="kpi-label">Threshold Used</div>
                        <div class="kpi-value" style="font-size:1.1rem">{result['threshold_used']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

# ══════════════════════════════════════════════════════════════
# TAB 4 — Batch Scoring
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section">Batch Applicant Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="info">Score a batch of applicants from a synthetic sample or uploaded CSV.</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not model_exists:
        st.markdown('<div class="warn">⚠️ Train a model first.</div>', unsafe_allow_html=True)
    else:
        n_batch = st.slider("Sample size to score", 50, 500, 100, 50)
        if st.button("🎯 Score Batch"):
            try:
                from predict import predict_batch
                from utils.data_utils import generate_credit_data
                batch_df = generate_credit_data(n_samples=n_batch).drop(columns=["default"])
                scored = predict_batch(batch_df)

                c1, c2, c3, c4 = st.columns(4)
                for col, (label, val, cls) in zip([c1,c2,c3,c4], [
                    ("Total Scored", f"{len(scored):,}", ""),
                    ("Approve Rate", f"{(scored['decision']=='APPROVE').mean()*100:.1f}%", "green"),
                    ("Deny Rate",    f"{(scored['decision']=='DENY').mean()*100:.1f}%", "red"),
                    ("Avg Risk Score", f"{scored['default_probability'].mean()*100:.1f}%", "amber"),
                ]):
                    with col:
                        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                                    f'<div class="kpi-value {cls}">{val}</div></div>',
                                    unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_l, col_r = st.columns(2)
                with col_l:
                    risk_counts = scored["risk_level"].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                 title="Risk Level Distribution",
                                 color_discrete_sequence=["#059669","#f59e0b","#ef4444","#9d174d"])
                    fig.update_layout(**PLOT_CFG, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with col_r:
                    fig = px.histogram(scored, x="default_probability", nbins=30,
                                       color="decision",
                                       color_discrete_map={"APPROVE":"#059669","DENY":"#e63946"},
                                       title="Default Probability Distribution", opacity=0.75)
                    fig.update_layout(**PLOT_CFG, height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📄 Scored Results"):
                    st.dataframe(scored[["age","income","credit_score","loan_amount",
                                         "default_probability","decision","risk_level"]].head(100),
                                 use_container_width=True)
            except Exception as e:
                st.error(str(e))
                st.exception(e)

# ══════════════════════════════════════════════════════════════
# TAB 5 — Analysis
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section">Model Analysis</div>', unsafe_allow_html=True)

    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        st.markdown('<div class="warn">No trained model found.</div>', unsafe_allow_html=True)
    else:
        with open(meta_path) as f:
            meta = json.load(f)

        # Metrics grid
        metrics = meta.get("test_metrics", {})
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{k}</div>'
                            f'<div class="kpi-value" style="font-size:1.15rem">{v}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        # Confusion Matrix
        with col_l:
            cm = meta.get("confusion_matrix", {})
            if cm:
                z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
                fig = go.Figure(go.Heatmap(
                    z=z, x=["Pred: No Default", "Pred: Default"],
                    y=["Actual: No Default", "Actual: Default"],
                    colorscale=[[0,"#ecfdf5"],[1,"#e63946"]],
                    text=[[str(v) for v in row] for row in z],
                    texttemplate="%{text}", textfont={"size": 18},
                    showscale=False,
                ))
                fig.update_layout(**PLOT_CFG, height=300, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)

        # Training curves
        with col_r:
            hist_df = pd.read_csv(os.path.join(MODEL_DIR, "training_history.csv")) \
                if os.path.exists(os.path.join(MODEL_DIR, "training_history.csv")) else None
            if hist_df is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist_df.get("auc", []), name="Train AUC",
                                         line=dict(color="#e63946", width=2)))
                if "val_auc" in hist_df:
                    fig.add_trace(go.Scatter(y=hist_df["val_auc"], name="Val AUC",
                                             line=dict(color="#3b82f6", width=2)))
                fig.update_layout(**PLOT_CFG, height=300, title="AUC Training Curve",
                                  xaxis_title="Epoch", yaxis_title="AUC")
                st.plotly_chart(fig, use_container_width=True)

        # Training summary
        st.markdown('<div class="section" style="margin-top:1rem">Training Summary</div>',
                    unsafe_allow_html=True)
        info_cols = st.columns(4)
        for col, (label, val) in zip(info_cols, [
            ("Features", meta.get("n_features","—")),
            ("Train Samples", f"{meta.get('train_samples','—'):,}" if isinstance(meta.get('train_samples'),int) else "—"),
            ("Epochs Run", meta.get("epochs_run","—")),
            ("Optimal Threshold", meta.get("optimal_threshold","—")),
        ]):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                            f'<div class="kpi-value" style="font-size:1.1rem">{val}</div></div>',
                            unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:DM Mono,monospace;font-size:0.7rem;color:#9ca3af;">'
    'CreditIQ · Deep Neural Network Credit Risk Assessment · TensorFlow/Keras + Streamlit'
    '</div>', unsafe_allow_html=True
)
