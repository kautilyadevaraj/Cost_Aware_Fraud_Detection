"""
Preprocessing and EDA utilities.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, check/handle missing values."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    n_missing = int(df.isnull().sum().sum())
    print(f"[preprocess] Dropped {n_before - n_after} duplicate rows "
          f"({n_before} -> {n_after}). Missing values: {n_missing}.")
    if n_missing > 0:
        df = df.fillna(df.median(numeric_only=True))
    return df


def split_and_scale(df: pd.DataFrame, target_col="Class", test_size=0.2,
                     val_size=0.1, random_state=RANDOM_STATE):
    """
    Stratified split into train/val/test, then scale Time and Amount
    (the V1..V28 columns are already PCA components on comparable scales
    in the real dataset, and are generated on comparable scales here).

    IMPORTANT (leakage): the scaler is fit ONLY on the training split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y,
        random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp,
        random_state=random_state
    )

    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X_train.columns]
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(f"[preprocess] Split sizes -> train: {len(X_train)}, "
          f"val: {len(X_val)}, test: {len(X_test)}")
    print(f"[preprocess] Fraud rate -> train: {y_train.mean():.5f}, "
          f"val: {y_val.mean():.5f}, test: {y_test.mean():.5f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def eda_class_distribution(df: pd.DataFrame, target_col="Class",
                            save_path=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = save_path or os.path.join(FIG_DIR, "class_distribution.png")

    counts = df[target_col].value_counts().sort_index()
    pct_fraud = 100 * counts.get(1, 0) / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Legitimate (0)", "Fraud (1)"], counts.values,
                color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Class distribution (raw counts)")
    axes[0].set_ylabel("Number of transactions")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(["Legitimate (0)", "Fraud (1)"],
                [100 - pct_fraud, pct_fraud],
                color=["#4C72B0", "#C44E52"])
    axes[1].set_title(f"Class distribution (%) — fraud = {pct_fraud:.3f}%")
    axes[1].set_ylabel("Percentage")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[preprocess] Saved class distribution plot -> {save_path}")

    summary = pd.DataFrame({
        "Property": ["Number of samples", "Number of features",
                     "Classes", "Legitimate count", "Fraud count",
                     "Fraud percentage"],
        "Value": [len(df), df.shape[1] - 1, "0 = Legitimate, 1 = Fraud",
                  int(counts.get(0, 0)), int(counts.get(1, 0)),
                  f"{pct_fraud:.4f}%"]
    })
    os.makedirs(TAB_DIR, exist_ok=True)
    summary.to_csv(os.path.join(TAB_DIR, "dataset_summary.csv"), index=False)
    return summary


def eda_amount_by_class(df: pd.DataFrame, target_col="Class",
                         save_path=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = save_path or os.path.join(FIG_DIR, "amount_by_class.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    data0 = df.loc[df[target_col] == 0, "Amount"]
    data1 = df.loc[df[target_col] == 1, "Amount"]
    ax.boxplot([data0, data1], tick_labels=["Legitimate", "Fraud"], showfliers=False)
    ax.set_ylabel("Transaction amount")
    ax.set_title("Transaction amount by class (outliers hidden)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[preprocess] Saved amount-by-class plot -> {save_path}")
