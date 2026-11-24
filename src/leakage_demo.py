"""
Demonstrates a common and subtle mistake: applying SMOTE BEFORE the
train/test split instead of after. Oversampling before the split lets
synthetic points derived from a test-set fraud case leak into the
training set (and vice versa), inflating the reported test performance.

This directly relates to Lab 6 (overfitting effects on false
positive/negative rates): a leaky pipeline looks great on paper and
fails in the real world.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from evaluate import compute_metrics

RANDOM_STATE = 42


def run_leakage_demo(X, y, test_size=0.2):
    """
    Returns a small DataFrame comparing:
      - "Correct (SMOTE after split)"
      - "Leaky (SMOTE before split)"
    on the SAME held-out test set, using the SAME model.
    """
    # ---- Correct pipeline: split first, then resample only the train part
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model_correct = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_correct.fit(X_train_res, y_train_res)
    y_pred_correct = model_correct.predict(X_test)
    y_score_correct = model_correct.predict_proba(X_test)[:, 1]
    m_correct = compute_metrics(y_test, y_pred_correct, y_score_correct)

    # ---- Leaky pipeline: resample the WHOLE dataset first, then split
    sm2 = SMOTE(random_state=RANDOM_STATE)
    X_res_all, y_res_all = sm2.fit_resample(X, y)
    X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
        X_res_all, y_res_all, test_size=test_size,
        stratify=y_res_all, random_state=RANDOM_STATE
    )
    model_leaky = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_leaky.fit(X_train_leak, y_train_leak)
    y_pred_leaky = model_leaky.predict(X_test_leak)
    y_score_leaky = model_leaky.predict_proba(X_test_leak)[:, 1]
    m_leaky = compute_metrics(y_test_leak, y_pred_leaky, y_score_leaky)

    rows = [
        {"Pipeline": "Correct (SMOTE after split)", **m_correct},
        {"Pipeline": "Leaky (SMOTE before split)", **m_leaky},
    ]
    df = pd.DataFrame(rows).set_index("Pipeline")
    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    df = df[[c for c in cols if c in df.columns]]
    return df
