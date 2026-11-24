"""
Model definitions: baseline classifiers with no imbalance handling,
and the same classifiers with class-weight balancing.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_baseline_models():
    """Models with default (no) imbalance handling."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=8, n_jobs=2,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=2, verbosity=0
        ),
    }


def get_class_weighted_models(scale_pos_weight=1.0):
    """Same models, but each penalised more for missing the minority class."""
    return {
        "Logistic Regression (weighted)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (weighted)": RandomForestClassifier(
            n_estimators=60, max_depth=8, n_jobs=2,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost (weighted)": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=2, verbosity=0, scale_pos_weight=scale_pos_weight
        ),
    }


def get_resampling_models():
    """Lighter model set used for the resampling comparison (SMOTE /
    undersampling / SMOTE+Tomek), which already multiplies the training
    set size, so we keep this to two fast, complementary models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=8, n_jobs=2,
            random_state=RANDOM_STATE
        ),
    }
