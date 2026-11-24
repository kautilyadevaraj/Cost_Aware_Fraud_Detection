# Cost-Aware Fraud Detection Under Class Imbalance

CS4431 — Machine Learning for Cyber Security, RV University.

Comparing resampling (SMOTE, random undersampling, SMOTE+Tomek), class
weighting, and cost-based threshold tuning for detecting fraudulent
credit card / UPI-style transactions, where fraud is under 0.2% of all
transactions.

## 1. Dataset

- **Source:** [Credit Card Fraud Detection, Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  (ULB Machine Learning Group). 284,807 transactions, 492 frauds
  (0.172%), features `Time`, `V1`–`V28` (PCA components of the original
  features, released this way for confidentiality), `Amount`, and the
  label `Class`.
- **To use the real data:** download `creditcard.csv` from the link
  above (requires a free Kaggle account) and place it at
  `data/creditcard.csv`.
- **Fallback:** if `data/creditcard.csv` is not present, `src/data.py`
  automatically generates a **synthetic** dataset with the same schema,
  the same row count (284,807) and the same fraud rate (0.172%), so the
  whole pipeline runs end to end without internet access or a Kaggle
  account. This is what the included report and figures were generated
  from — console output clearly prints `[data] ... SYNTHETIC dataset`
  whenever this path is used.
- **Before final submission:** drop the real CSV into `data/` and run
  `python3 run_all.py` again — every table, figure and the model bundle
  will regenerate automatically from the real data. See
  "Reproducing with the real dataset" below.

## 2. Project structure

```
.
├── data/                      # put creditcard.csv here (not included)
├── src/
│   ├── data.py                 # load real data / generate synthetic fallback
│   ├── preprocess.py            # cleaning, split, scaling, EDA plots
│   ├── models.py                 # baseline + class-weighted + resampling model sets
│   ├── imbalance.py               # SMOTE / undersampling / SMOTE+Tomek
│   ├── threshold.py                # cost-based threshold selection
│   ├── evaluate.py                  # metrics + confusion matrix / ROC / PR plots
│   └── leakage_demo.py               # SMOTE-before-split vs. after-split demo
├── app/
│   └── app.py                   # Streamlit demo (threshold slider + transaction checker)
├── outputs/
│   ├── figures/                  # all generated plots (PNG)
│   ├── tables/                    # all generated result tables (CSV)
│   ├── results_summary.json        # machine-readable run summary (used to fill the report)
│   └── fraud_model.joblib           # saved best model + scaler, used by the demo app
├── report/                     # LaTeX report (RV University template)
├── run_all.py                  # runs the entire pipeline end to end
└── requirements.txt
```

## 3. How to run

```bash
pip install -r requirements.txt

# 1. Run the full pipeline: data -> EDA -> train -> evaluate -> threshold tuning -> leakage demo
python3 run_all.py

# 2. Launch the interactive demo (uses the model saved by run_all.py)
streamlit run app/app.py
```

`run_all.py` takes about 5–10 minutes on a 2-core machine (most of the
time goes into Random Forest under SMOTE, which roughly doubles the
training set size). Logistic Regression and XGBoost finish in seconds.

## 4. What `run_all.py` does

1. Loads the dataset (real or synthetic fallback) and reports which one.
2. Cleans it (duplicate removal, missing-value check) and saves EDA
   plots (`class_distribution.png`, `amount_by_class.png`).
3. Stratified 70/10/20 train/val/test split; `StandardScaler` fit only
   on the training split and applied to `Time` and `Amount`.
4. Trains three baseline models (Logistic Regression, Random Forest,
   XGBoost) with no imbalance handling.
5. Trains the same three models with class weighting
   (`class_weight="balanced"` / `scale_pos_weight`).
6. Trains Logistic Regression and Random Forest under three resampling
   strategies: SMOTE, random undersampling, and SMOTE+Tomek — fit on
   the training split only.
7. Builds a full comparison table (`outputs/tables/model_comparison.csv`)
   and picks the best model by **F1-score** (not accuracy — with 0.17%
   fraud, a model that predicts "legitimate" every time scores >99.8%
   accuracy while catching nothing).
8. Tunes a cost-based decision threshold for the best model on the
   validation set, where missing a fraud is assumed to cost 100× a
   false alarm (`cost_fn=100, cost_fp=1`) — adjustable in `run_all.py`
   or live in the Streamlit app.
9. Demonstrates data leakage: applying SMOTE on the *whole* dataset
   before splitting vs. correctly applying it only to the training
   split after splitting, evaluated on the same held-out data
   (`outputs/tables/leakage_demo.csv`).
10. Saves a `fraud_model.joblib` bundle (model + scaler + chosen
    threshold) that the Streamlit app loads directly.

## 5. Reproducing with the real dataset

```bash
# after downloading creditcard.csv from Kaggle
mv ~/Downloads/creditcard.csv data/creditcard.csv
python3 run_all.py
```

No other code changes are required — `src/data.py` checks for
`data/creditcard.csv` first and only falls back to synthetic data when
it's absent. Re-running regenerates every figure, table and the model
bundle from the real data; re-compile the report (`report/`) afterwards
so the submitted numbers match the real dataset.

## 6. Key results (synthetic run included in this repo)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.927 | 0.515 | 0.662 | 0.996 |
| Random Forest (baseline) | 1.000 | 0.263 | 0.416 | 0.984 |
| XGBoost (baseline) | 0.881 | 0.525 | 0.658 | 0.996 |
| Logistic Regression (class-weighted) | 0.045 | 0.980 | 0.087 | 0.996 |
| Logistic Regression (SMOTE) | 0.056 | 0.939 | 0.106 | 0.996 |

Full table: `outputs/tables/model_comparison.csv`.

Baselines maximize precision at the cost of recall; weighting and
resampling flip that trade-off toward recall. The cost-based threshold
(driven by `cost_fn=100, cost_fp=1`) pushes the decision threshold down
to 0.01, raising recall on the best baseline model to 0.91 at the
expense of precision — a concrete illustration of why the "right"
threshold is a business decision, not a statistical default.

## 7. Libraries used

scikit-learn, imbalanced-learn, XGBoost, pandas, NumPy, Matplotlib,
Streamlit, joblib. See `requirements.txt` for exact versions.

## 8. Academic integrity

All code in this repository was written for this project. Third-party
libraries (scikit-learn, imbalanced-learn, XGBoost, Streamlit) are
used as documented above and cited in the report's references. The
dataset, when downloaded from Kaggle, is attributed to the ULB Machine
Learning Group (see Section 1).
