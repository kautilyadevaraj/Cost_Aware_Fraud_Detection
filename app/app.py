"""
Streamlit demo for the Cost-Aware Fraud Detection project.

Run with:
    streamlit run app/app.py

Two panels:
  1. A threshold slider showing the live precision/recall trade-off on
     the held-out test set (re-derived from the saved results table).
  2. A transaction checker: enter a transaction's feature values (or
     load a random test-set example) and get a fraud probability plus
     a verdict at the chosen threshold.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import load_data
from preprocess import clean_data, split_and_scale

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "fraud_model.joblib")

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("Cost-Aware Fraud Detection — Demo")
st.caption(
    "Credit card / UPI-style transaction fraud detector. "
    "Compares precision vs. recall as you move the decision threshold, "
    "and lets you score an individual transaction."
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_split():
    df, is_synthetic = load_data(verbose=False)
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    return X_test, y_test, is_synthetic


if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Run `python3 run_all.py` first to train "
        "the model and generate outputs/fraud_model.joblib."
    )
    st.stop()

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
scaled_columns = bundle["scaled_columns"]
default_threshold = bundle["threshold"]
model_name = bundle["model_name"]

X_test, y_test, is_synthetic = load_test_split()
y_score = (model.predict_proba(X_test)[:, 1]
           if hasattr(model, "predict_proba")
           else model.decision_function(X_test))

if is_synthetic:
    st.warning(
        "Running on a SYNTHETIC dataset (no internet access in this "
        "environment to fetch the real Kaggle file). Drop the real "
        "creditcard.csv into data/ and re-run run_all.py for real results.",
        icon="⚠️",
    )

st.markdown(f"**Model in use:** `{model_name}`")

# ---------------- Panel 1: threshold slider ----------------
st.header("1. Threshold trade-off")
col1, col2 = st.columns([2, 1])

with col2:
    threshold = st.slider(
        "Decision threshold", min_value=0.0, max_value=1.0,
        value=float(default_threshold), step=0.01,
    )
    cost_fn = st.number_input("Cost of a missed fraud (false negative)",
                               min_value=1.0, value=100.0, step=1.0)
    cost_fp = st.number_input("Cost of a false alarm (false positive)",
                               min_value=1.0, value=1.0, step=1.0)

y_pred = (y_score >= threshold).astype(int)
tp = int(np.sum((y_pred == 1) & (y_test == 1)))
fp = int(np.sum((y_pred == 1) & (y_test == 0)))
fn = int(np.sum((y_pred == 0) & (y_test == 1)))
tn = int(np.sum((y_pred == 0) & (y_test == 0)))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
total_cost = fn * cost_fn + fp * cost_fp

with col1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.3f}")
    m2.metric("Recall", f"{recall:.3f}")
    m3.metric("F1-score", f"{f1:.3f}")
    m4.metric("Total cost", f"{total_cost:,.0f}")

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Legitimate", "Actual: Fraud"],
        columns=["Pred: Legitimate", "Pred: Fraud"],
    )
    st.dataframe(cm_df, use_container_width=True)

st.divider()

# ---------------- Panel 2: transaction checker ----------------
st.header("2. Transaction checker")
st.caption(
    "Load a random test-set transaction, or edit Amount / Time, then score it."
)

if "sample_idx" not in st.session_state or st.session_state.sample_idx not in X_test.index:
    # start from a real test-set row (index labels are not 0..n after splitting)
    st.session_state.sample_idx = X_test.index[y_test == 0][0]

bcol1, bcol2 = st.columns(2)
if bcol1.button("Load random legitimate transaction"):
    legit_idx = X_test.index[y_test == 0]
    st.session_state.sample_idx = np.random.choice(legit_idx)
if bcol2.button("Load random fraud transaction"):
    fraud_idx = X_test.index[y_test == 1]
    if len(fraud_idx) > 0:
        st.session_state.sample_idx = np.random.choice(fraud_idx)
    else:
        st.info("No fraud examples available in this split.")

row = X_test.loc[[st.session_state.sample_idx]].copy()
true_label = y_test.loc[st.session_state.sample_idx]

ecol1, ecol2 = st.columns(2)
if "Amount" in row.columns:
    # The stored row has Amount already scaled; show/edit in raw space
    # by inverse-transforming just for display convenience.
    raw_vals = scaler.inverse_transform(row[scaled_columns])
    raw_amount = float(raw_vals[0][scaled_columns.index("Amount")]) if "Amount" in scaled_columns else None
    if raw_amount is not None:
        new_amount = ecol1.number_input("Amount", value=round(raw_amount, 2))
    else:
        new_amount = None
if "Time" in row.columns and "Time" in scaled_columns:
    raw_time = float(raw_vals[0][scaled_columns.index("Time")])
    new_time = ecol2.number_input("Time (seconds elapsed)", value=round(raw_time, 1))
else:
    new_time = None

edited_raw = {}
if new_amount is not None:
    edited_raw["Amount"] = new_amount
if new_time is not None:
    edited_raw["Time"] = new_time

if edited_raw:
    raw_df = pd.DataFrame([edited_raw])
    for col in scaled_columns:
        if col not in raw_df.columns:
            raw_df[col] = raw_vals[0][scaled_columns.index(col)]
    raw_df = raw_df[scaled_columns]
    scaled_vals = scaler.transform(raw_df)
    for i, col in enumerate(scaled_columns):
        row[col] = scaled_vals[0][i]

score = float(model.predict_proba(row[feature_columns])[:, 1][0]
              if hasattr(model, "predict_proba")
              else model.decision_function(row[feature_columns])[0])
verdict = "FRAUD" if score >= threshold else "LEGITIMATE"

st.markdown(f"**True label (from test set):** "
            f"{'Fraud' if true_label == 1 else 'Legitimate'}")
st.markdown(f"**Model fraud probability:** `{score:.4f}`")
if verdict == "FRAUD":
    st.error(f"Verdict @ threshold {threshold:.2f}: {verdict}")
else:
    st.success(f"Verdict @ threshold {threshold:.2f}: {verdict}")
