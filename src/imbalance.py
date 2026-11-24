"""
Imbalance-handling strategies compared in this project:
  1. none            - train as-is (baseline)
  2. class_weight     - handled inside models.py (no resampling needed)
  3. smote            - SMOTE oversampling of the minority class
  4. undersample      - random undersampling of the majority class
  5. smote_tomek      - SMOTE followed by Tomek-link cleaning

All resampling is fit on the TRAINING split only, to avoid leakage
(see leakage_demo.py for a concrete demonstration of what goes wrong
if this rule is broken).
"""
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

RANDOM_STATE = 42


def apply_resampling(X_train, y_train, method: str):
    """
    method in {"none", "smote", "undersample", "smote_tomek"}
    Returns resampled (X_train, y_train). Must only ever be called on
    the training split.
    """
    if method == "none":
        return X_train, y_train

    if method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == "smote_tomek":
        sampler = SMOTETomek(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"[imbalance] {method}: {len(X_train)} -> {len(X_res)} rows "
          f"(fraud rate {y_train.mean():.5f} -> {y_res.mean():.5f})")
    return X_res, y_res
