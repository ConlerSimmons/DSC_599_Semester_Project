import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    auc,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5):
    """
    I compute the core fraud-detection metrics:
    - precision, recall, f1 (for class 1)
    - PR-AUC  (most important for imbalanced fraud detection)
    - ROC-AUC
    """

    # Convert probabilities to binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # PR curve → more important than ROC for skewed datasets
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC can fail if only one class is present, so wrap
    try:
        roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc = float("nan")

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "pr_auc":    float(pr_auc),
        "roc_auc":   float(roc),
    }


def print_metrics(name: str, metrics: dict):
    """
    I print metrics in a clean, readable format.
    """
    print(f"\n===== {name} Metrics =====")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")