"""
Baseline models: XGBoost and LightGBM on IEEE-CIS fraud data.

Uses the same temporal 70/15/15 split and feature selection as the deep learning models.
These results serve as the benchmark for evaluating whether deep learning adds value.

Run from project root:
    python run_baseline.py
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loading import load_merged_train
from src.feature_selection import auto_select_features


def tune_threshold_and_eval(y_true, y_score, label):
    """Find the threshold that maximises F1, then report all metrics."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_score >= best_threshold).astype(int)

    metrics = {
        f"{label}_precision":      precision_score(y_true, y_pred, zero_division=0),
        f"{label}_recall":         recall_score(y_true, y_pred, zero_division=0),
        f"{label}_f1":             f1_score(y_true, y_pred, zero_division=0),
        f"{label}_roc_auc":        roc_auc_score(y_true, y_score),
        f"{label}_pr_auc":         average_precision_score(y_true, y_score),
        f"{label}_best_threshold": best_threshold,
    }
    return metrics


def prepare_features(df, numeric_cols, categorical_cols, target_col="isFraud"):
    """Label-encode categoricals and fill NaNs — minimal prep for tree models."""
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes  # -1 for NaN

    X[numeric_cols] = X[numeric_cols].fillna(0)
    return X.values.astype(np.float32), y


def run_model(name, clf, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{'='*40}")
    print(f" Training {name}")
    print(f"{'='*40}")

    clf.fit(X_train, y_train)

    val_score  = clf.predict_proba(X_val)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]

    val_metrics  = tune_threshold_and_eval(y_val,  val_score,  "val")
    test_metrics = tune_threshold_and_eval(y_test, test_score, "test")

    print(f"\n----- {name} — Validation -----")
    for k, v in val_metrics.items():
        print(f"  {k:28s}: {v:.4f}")

    print(f"\n----- {name} — Test -----")
    for k, v in test_metrics.items():
        print(f"  {k:28s}: {v:.4f}")

    return {**val_metrics, **test_metrics}


def main():
    print("\n==============================")
    print(" STEP 1: Loading Data")
    print("==============================")
    df = load_merged_train(data_dir="data")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print(f"Data sorted by TransactionDT. Shape: {df.shape}")

    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n))
    print(f"Split → train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    print("\n==============================")
    print(" STEP 2: Feature Selection")
    print("==============================")
    numeric_cols, categorical_cols = auto_select_features(
        df, target_col="isFraud", max_numeric=40, max_categorical=20
    )

    print("\n==============================")
    print(" STEP 3: Preparing Features")
    print("==============================")
    X, y = prepare_features(df, numeric_cols, categorical_cols, target_col="isFraud")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # ---- XGBoost ----
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_metrics = run_model("XGBoost", xgb, X_train, y_train, X_val, y_val, X_test, y_test)

    # ---- LightGBM ----
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_metrics = run_model("LightGBM", lgbm, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n\n==============================")
    print(" SUMMARY (Test Set PR-AUC)")
    print("==============================")
    print(f"  XGBoost  test_pr_auc : {xgb_metrics['test_pr_auc']:.4f}")
    print(f"  LightGBM test_pr_auc : {lgbm_metrics['test_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
