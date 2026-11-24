"""
Evaluation metrics and plots.

Accuracy is reported for completeness but is explicitly NOT used to
pick a winning model, since with ~0.17% fraud a model that predicts
"legitimate" every time already scores >99.8% accuracy while catching
zero fraud. Precision, recall, F1 and PR-AUC drive model selection.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    if y_score is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_score)
        metrics["PR_AUC"] = average_precision_score(y_true, y_score)
    return metrics


def evaluate_all(results: dict, save_csv=None):
    """
    results: {model_name: {"y_true":..., "y_pred":..., "y_score":...}}
    Returns a DataFrame of metrics, one row per model, and saves it.
    """
    rows = []
    for name, r in results.items():
        m = compute_metrics(r["y_true"], r["y_pred"], r.get("y_score"))
        m["Model"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Model")
    cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "PR_AUC"]
    df = df[[c for c in cols if c in df.columns] +
            [c for c in df.columns if c not in cols]]
    save_csv = save_csv or os.path.join(TAB_DIR, "model_comparison.csv")
    df.to_csv(save_csv)
    print(f"[evaluate] Saved results table -> {save_csv}")
    return df


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved confusion matrix -> {save_path}")


def plot_roc_curves(results: dict, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        if r.get("y_score") is None:
            continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        auc = roc_auc_score(r["y_true"], r["y_score"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved ROC curves -> {save_path}")


def plot_pr_curves(results: dict, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        if r.get("y_score") is None:
            continue
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_score"])
        ap = average_precision_score(r["y_true"], r["y_score"])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved PR curves -> {save_path}")
