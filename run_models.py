"""
Main script to run TabTransformer model:

1. Loads and merges IEEE-CIS fraud dataset
2. Automatically selects numeric + categorical features
3. Trains the TabTransformer (custom PyTorch version)
4. Prints performance metrics

Run from project root:

    python run_models.py
"""

import torch

# Allow OmegaConf DictConfig objects to be unpickled safely with torch.load(weights_only=True)
try:
    from omegaconf.dictconfig import DictConfig
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DictConfig])
except Exception:
    # If anything goes wrong here (older torch, missing omegaconf, etc.),
    # I just skip the extra registration so the script still runs.
    pass

from src.data_loading import load_merged_train
from src.feature_selection import auto_select_features
from src.tabtransformer_custom.train_custom import train_tabtransformer_custom


def main():

    print("\n==============================")
    print(" STEP 1: Loading Data")
    print("==============================")
    df = load_merged_train(data_dir="data")

    # -------------------------------------------------
    # DEBUG MODE: use a smaller subset for faster debugging
    # -------------------------------------------------
    DEBUG_MODE = False
    if DEBUG_MODE:
        df = df.sample(n=5000, random_state=42)
        print(f"DEBUG MODE ACTIVE: using {len(df)} rows instead of full dataset")
    # -------------------------------------------------

    print("\n==============================")
    print(" STEP 2: Feature Selection")
    print("==============================")
    numeric_cols, categorical_cols = auto_select_features(
        df,
        target_col="isFraud",
        max_numeric=20,
        max_categorical=20,
    )

    print("\n==============================")
    print(" STEP 3: Training Custom TabTransformer")
    print("==============================")
    custom_metrics, custom_model, y_true, y_pred = train_tabtransformer_custom(
        df, numeric_cols, categorical_cols, target_col="isFraud"
    )

    print("\n==============================")
    print(" Custom Model Metrics")
    print("==============================")
    for k, v in custom_metrics.items():
        print(f"{k:10s} : {v:.4f}")

    # ----------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------
    from sklearn.metrics import confusion_matrix

    print("\n=== Confusion Matrix ===")
    try:
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    except Exception as e:
        print("Could not compute confusion matrix:", e)
        for k, v in custom_metrics.items():
            print(f"{k:10s} : {v:.4f}")


if __name__ == "__main__":
    main()