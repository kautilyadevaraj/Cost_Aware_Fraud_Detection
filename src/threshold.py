"""
Cost-aware threshold selection.

Instead of using the default 0.5 decision threshold, this module scans
candidate thresholds on the validation set and picks the one that
minimises a business cost:

    total_cost = FN * cost_missed_fraud + FP * cost_false_alarm

This turns the precision/recall trade-off into an explicit business
decision instead of an arbitrary default.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")


def find_cost_optimal_threshold(y_true, y_score, cost_fn=100.0, cost_fp=1.0,
                                 n_thresholds=200):
    """
    cost_fn: cost of a missed fraud (false negative)
    cost_fp: cost of a false alarm (false positive)
    Returns (best_threshold, cost_table_dataframe)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        cost = fn * cost_fn + fp * cost_fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({"threshold": t, "cost": cost, "FP": fp, "FN": fn,
                     "precision": precision, "recall": recall})

    cost_df = pd.DataFrame(rows)
    best_row = cost_df.loc[cost_df["cost"].idxmin()]
    best_threshold = float(best_row["threshold"])
    return best_threshold, cost_df


def plot_cost_curve(cost_df, best_threshold, save_path, model_name=""):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(cost_df["threshold"], cost_df["cost"], color="#C44E52",
              label="Total cost")
    ax1.axvline(best_threshold, color="gray", linestyle="--",
                label=f"Chosen threshold = {best_threshold:.2f}")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Total cost (FN×cost_fn + FP×cost_fp)")
    ax1.set_title(f"Cost-optimal threshold selection{' — ' + model_name if model_name else ''}")
    ax1.legend(loc="upper center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[threshold] Saved cost curve -> {save_path}")
