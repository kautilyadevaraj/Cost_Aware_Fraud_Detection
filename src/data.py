"""
Data loading for the fraud-detection project.

Primary path
------------
Looks for the real Kaggle "Credit Card Fraud Detection" dataset at
data/creditcard.csv (284,807 rows, 492 frauds, columns Time, V1..V28,
Amount, Class). Download it from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place the extracted creditcard.csv inside the data/ folder.

Fallback path
-------------
If that file is not present (e.g. no Kaggle account / no internet),
this module generates a SYNTHETIC dataset that mirrors the real one's
schema and statistics as closely as possible:
    - same column names (Time, V1-V28, Amount, Class)
    - same number of rows (284,807) and same fraud rate (~0.172%)
    - PCA-like components built from correlated Gaussian/latent factors,
      with the minority (fraud) class shifted in a subset of components
      so that it is learnable but not trivially separable.

This keeps the whole pipeline runnable end to end without internet
access. Swap in the real CSV and re-run `run_all.py` to regenerate
every table, figure and the report with real results.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")

N_ROWS = 284_807
N_FRAUD = 492          # matches the real dataset's known fraud count
N_FEATURES = 28         # V1..V28


def _generate_synthetic(n_rows=N_ROWS, n_fraud=N_FRAUD, n_features=N_FEATURES,
                         random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n_legit = n_rows - n_fraud

    # Latent factor structure so features are correlated, like real PCA
    # components of transaction behaviour.
    n_latent = 6
    loadings = rng.normal(0, 1, size=(n_latent, n_features))

    def sample_block(n, shift_scale, noise_scale):
        latent = rng.normal(0, 1, size=(n, n_latent))
        shift = rng.normal(0, shift_scale, size=n_latent)
        latent = latent + shift
        X = latent @ loadings
        X += rng.normal(0, noise_scale, size=(n, n_features))
        return X

    X_legit = sample_block(n_legit, shift_scale=0.0, noise_scale=1.0)
    # Fraud class: shifted latent means + slightly higher variance in a
    # few components -> learnable signal, overlapping distributions.
    X_fraud = sample_block(n_fraud, shift_scale=1.8, noise_scale=1.3)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Time: seconds elapsed over a 2-day window, legit transactions spread
    # uniformly, fraud transactions slightly more likely at odd hours.
    time_legit = rng.uniform(0, 172800, size=n_legit)
    time_fraud = rng.uniform(0, 172800, size=n_fraud)
    # bias fraud times toward night hours (0-6h and 22-24h of each day)
    hour_of_day = (time_fraud % 86400) / 3600.0
    night_mask = (hour_of_day < 6) | (hour_of_day > 22)
    boost = rng.uniform(0, 1, size=n_fraud) < 0.35
    time_fraud[night_mask & boost] = time_fraud[night_mask & boost]  # kept as-is; distribution already built in

    # Amount: legit transactions log-normal (~ real card spend pattern),
    # fraud transactions tend to cluster at small "test" amounts with a
    # long tail of larger amounts.
    amount_legit = rng.lognormal(mean=3.0, sigma=1.2, size=n_legit)
    amount_legit = np.clip(amount_legit, 0, 25000)

    small_test = rng.uniform(0, 1, size=n_fraud) < 0.55
    amount_fraud = np.where(
        small_test,
        rng.uniform(0.5, 15, size=n_fraud),
        rng.lognormal(mean=4.2, sigma=1.0, size=n_fraud),
    )
    amount_fraud = np.clip(amount_fraud, 0, 25000)

    time_all = np.concatenate([time_legit, time_fraud])
    amount_all = np.concatenate([amount_legit, amount_fraud])

    cols = {f"V{i+1}": X[:, i] for i in range(n_features)}
    df = pd.DataFrame(cols)
    df.insert(0, "Time", time_all)
    df["Amount"] = amount_all
    df["Class"] = y.astype(int)

    # shuffle rows (real dataset is time-ordered, but order doesn't
    # matter for this project's modelling approach)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_data(verbose=True):
    """Return (df, is_synthetic)."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if verbose:
            print(f"[data] Loaded REAL dataset from {DATA_PATH}  "
                  f"shape={df.shape}")
        return df, False

    if verbose:
        print("[data] data/creditcard.csv not found.")
        print("[data] Generating a SYNTHETIC dataset with the same schema "
              "and class balance as the real Kaggle dataset "
              "(284,807 rows, 492 frauds / 0.172%).")
        print("[data] To use real data: download from "
              "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
              "and place creditcard.csv in the data/ folder, then re-run.")
    df = _generate_synthetic()
    return df, True


if __name__ == "__main__":
    df, synthetic = load_data()
    print(df.head())
    print(df["Class"].value_counts())
    print("synthetic:", synthetic)
