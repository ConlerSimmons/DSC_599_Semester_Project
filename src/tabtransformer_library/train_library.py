import numpy as np
from sklearn.model_selection import train_test_split

from src.utils_metrics import compute_metrics, print_metrics
from src.tabtransformer_library.model_library import build_tabtransformer_library


def train_tabtransformer_library(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
):
    """
    I train the library TabTransformer (pytorch-tabular version) and return:
    - metrics dictionary
    - trained model

    This gives us our "baseline" TabTransformer performance.
    """

    # ---------------------------
    # Subset only selected features
    # ---------------------------
    df = df[numeric_cols + categorical_cols + [target_col]].copy()

    # ---------------------------
    # Impute missing values
    # ---------------------------
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype("object").fillna("missing").astype("category")

    # ---------------------------
    # Stratified train/validation split
    # ---------------------------
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df[target_col], random_state=42
    )

    # ---------------------------
    # Build the model
    # ---------------------------
    model = build_tabtransformer_library(numeric_cols, categorical_cols, target_col)

    # ---------------------------
    # Fit the model
    # ---------------------------
    print("\nTraining Library TabTransformer...")
    model.fit(train=train_df, validation=val_df)

    # ---------------------------
    # Get predictions
    # ---------------------------
    preds_df = model.predict(val_df)

    # pytorch-tabular usually names the probability column like:
    #   prediction_probability_1
    # but we detect it automatically just in case.
    proba_col = None
    for col in preds_df.columns:
        if "prediction_probability" in col:
            proba_col = col
            break

    if proba_col is None:
        raise RuntimeError(
            "Could not find prediction probability column from pytorch-tabular output."
        )

    y_proba = preds_df[proba_col].to_numpy()
    y_true = val_df[target_col].to_numpy()

    # ---------------------------
    # Compute metrics
    # ---------------------------
    metrics = compute_metrics(y_true, y_proba)
    print_metrics("Library TabTransformer", metrics)

    return metrics, model