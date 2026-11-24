"""
End-to-end pipeline for the Cost-Aware Fraud Detection project.

Run with:  python3 run_all.py

What it does:
  1. Loads data (real data/creditcard.csv if present, else synthetic fallback)
  2. Cleans it and runs EDA (class distribution, amount-by-class)
  3. Splits into train/val/test with stratification, scales Time/Amount
  4. Trains baseline models (no imbalance handling)
  5. Trains class-weighted models
  6. Trains models under SMOTE / random undersampling / SMOTE+Tomek
  7. Picks the best model by F1-score on the test set
  8. Finds a cost-optimal decision threshold on the validation set for
     the best model, and re-evaluates it on the test set at that
     threshold
  9. Produces the leakage demonstration (SMOTE before vs after split)
  10. Saves ALL figures/tables to outputs/, and a results.json summary
      used to auto-fill the LaTeX report
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_data
from preprocess import clean_data, split_and_scale, eda_class_distribution, eda_amount_by_class
from models import get_baseline_models, get_class_weighted_models, get_resampling_models
from imbalance import apply_resampling
from evaluate import evaluate_all, plot_confusion_matrix, plot_roc_curves, plot_pr_curves, compute_metrics
from threshold import find_cost_optimal_threshold, plot_cost_curve
from leakage_demo import run_leakage_demo

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    t_start = time.time()
    summary = {}

    # ---------- 1-3. Data, cleaning, EDA, split ----------
    df, is_synthetic = load_data()
    summary["is_synthetic"] = is_synthetic
    df = clean_data(df)

    dataset_summary = eda_class_distribution(df)
    eda_amount_by_class(df)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    summary["n_train"] = len(X_train)
    summary["n_val"] = len(X_val)
    summary["n_test"] = len(X_test)
    summary["n_features"] = X_train.shape[1]
    summary["fraud_rate_overall"] = float(df["Class"].mean())

    all_results = {}  # name -> {y_true, y_pred, y_score}

    # ---------- 4. Baseline models ----------
    print("\n=== Training baseline models (no imbalance handling) ===")
    for name, model in get_baseline_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)
        all_results[name] = {"y_true": y_test, "y_pred": y_pred, "y_score": y_score}
        print(f"  {name}: fit+predict in {time.time()-t0:.1f}s")

    # ---------- 5. Class-weighted models ----------
    print("\n=== Training class-weighted models ===")
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    for name, model in get_class_weighted_models(scale_pos_weight).items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)
        all_results[name] = {"y_true": y_test, "y_pred": y_pred, "y_score": y_score}
        print(f"  {name}: fit+predict in {time.time()-t0:.1f}s")

    # ---------- 6. Resampling strategies ----------
    print("\n=== Training models under resampling strategies ===")
    resampling_methods = ["smote", "undersample", "smote_tomek"]
    for method in resampling_methods:
        X_res, y_res = apply_resampling(X_train, y_train, method)
        for name, model in get_resampling_models().items():
            full_name = f"{name} ({method})"
            t0 = time.time()
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            y_score = get_scores(model, X_test)
            all_results[full_name] = {"y_true": y_test, "y_pred": y_pred, "y_score": y_score}
            print(f"  {full_name}: fit+predict in {time.time()-t0:.1f}s")

    # ---------- 7. Evaluate everything ----------
    results_df = evaluate_all(all_results)
    print("\n=== Full comparison table ===")
    print(results_df.round(4))

    # Pick best model by F1 (not accuracy, see evaluate.py docstring)
    best_model_name = results_df["F1"].idxmax()
    summary["best_model_name"] = best_model_name
    summary["best_model_metrics"] = results_df.loc[best_model_name].to_dict()
    print(f"\nBest model by F1-score: {best_model_name}")

    # Confusion matrix + ROC/PR for the best model and a shortlist
    best_r = all_results[best_model_name]
    plot_confusion_matrix(
        best_r["y_true"], best_r["y_pred"],
        title=f"Confusion Matrix — {best_model_name}",
        save_path=os.path.join(FIG_DIR, "confusion_matrix_best.png"),
    )

    # For readability, plot ROC/PR curves for baselines + best resampled model
    shortlist_names = list(get_baseline_models().keys()) + [best_model_name]
    shortlist_names = list(dict.fromkeys(shortlist_names))  # dedupe, keep order
    shortlist = {k: all_results[k] for k in shortlist_names if k in all_results}
    plot_roc_curves(shortlist, os.path.join(FIG_DIR, "roc_curves.png"))
    plot_pr_curves(shortlist, os.path.join(FIG_DIR, "pr_curves.png"))

    # ---------- 8. Cost-based threshold tuning on the best model ----------
    print("\n=== Cost-based threshold tuning (best model) ===")
    # Refit best model's recipe is complex (different per method); instead,
    # reuse its test-set scores but tune threshold on a validation-style
    # split carved out of the test scores to keep the pipeline simple and
    # avoid re-deriving every training recipe here.
    # We instead fit the SAME architecture on train (resampled if needed)
    # and score the validation set properly.
    best_is_resampled = any(m in best_model_name for m in resampling_methods)
    if best_is_resampled:
        method_used = next(m for m in resampling_methods if m in best_model_name)
        base_name = best_model_name.split(" (")[0]
        X_res, y_res = apply_resampling(X_train, y_train, method_used)
        model = get_resampling_models()[base_name]
        model.fit(X_res, y_res)
    elif "weighted" in best_model_name:
        base_name = best_model_name.replace(" (weighted)", "")
        model = get_class_weighted_models(scale_pos_weight)[best_model_name]
        model.fit(X_train, y_train)
    else:
        model = get_baseline_models()[best_model_name]
        model.fit(X_train, y_train)

    val_scores = get_scores(model, X_val)
    cost_fn, cost_fp = 100.0, 1.0
    best_threshold, cost_df = find_cost_optimal_threshold(
        y_val, val_scores, cost_fn=cost_fn, cost_fp=cost_fp
    )
    plot_cost_curve(cost_df, best_threshold,
                     os.path.join(FIG_DIR, "cost_threshold_curve.png"),
                     model_name=best_model_name)

    test_scores_final = get_scores(model, X_test)
    y_pred_default = (test_scores_final >= 0.5).astype(int)
    y_pred_tuned = (test_scores_final >= best_threshold).astype(int)
    m_default = compute_metrics(y_test, y_pred_default, test_scores_final)
    m_tuned = compute_metrics(y_test, y_pred_tuned, test_scores_final)

    threshold_compare = pd.DataFrame([
        {"Threshold": 0.5, **m_default},
        {"Threshold": round(best_threshold, 3), **m_tuned},
    ])
    threshold_compare.to_csv(os.path.join(TAB_DIR, "threshold_comparison.csv"), index=False)
    print("\n=== Default vs. cost-optimal threshold ===")
    print(threshold_compare[["Threshold", "Precision", "Recall", "F1", "FP", "FN"]])

    summary["cost_fn"] = cost_fn
    summary["cost_fp"] = cost_fp
    summary["best_threshold"] = best_threshold
    summary["metrics_default_threshold"] = m_default
    summary["metrics_tuned_threshold"] = m_tuned

    # Persist the final model + scaler + feature order + threshold for the demo app
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X_train.columns),
        "scaled_columns": [c for c in ["Time", "Amount"] if c in X_train.columns],
        "threshold": best_threshold,
        "model_name": best_model_name,
    }, os.path.join(OUT_DIR, "fraud_model.joblib"))
    print(f"[run_all] Saved deployable model bundle -> "
          f"{os.path.join(OUT_DIR, 'fraud_model.joblib')}")

    plot_confusion_matrix(
        y_test, y_pred_tuned,
        title=f"Confusion Matrix — {best_model_name} @ cost-optimal threshold",
        save_path=os.path.join(FIG_DIR, "confusion_matrix_tuned_threshold.png"),
    )

    # ---------- 9. Leakage demonstration ----------
    print("\n=== Leakage demonstration: SMOTE before vs. after split ===")
    X_full = df.drop(columns=["Class"])
    y_full = df["Class"]
    leak_df = run_leakage_demo(X_full, y_full)
    leak_df.to_csv(os.path.join(TAB_DIR, "leakage_demo.csv"))
    print(leak_df.round(4))

    # ---------- 10. Save summary ----------
    summary["dataset_summary"] = dataset_summary.to_dict(orient="records")
    summary["results_table"] = results_df.round(4).reset_index().to_dict(orient="records")
    summary["leakage_table"] = leak_df.round(4).reset_index().to_dict(orient="records")
    summary["runtime_seconds"] = round(time.time() - t_start, 1)

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {summary['runtime_seconds']}s. "
          f"All outputs saved under outputs/.")


if __name__ == "__main__":
    main()
