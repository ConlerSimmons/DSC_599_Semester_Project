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
    # Fail‑safe: skip registration if something is missing
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
    # DEBUG MODE (set to False for full training)
    # -------------------------------------------------
    DEBUG_MODE = False
    if DEBUG_MODE:
        df = df.sample(n=5000, random_state=42)
        print(f"DEBUG MODE ACTIVE: using {len(df)} rows instead of full dataset")
    # -------------------------------------------------

    # Sort by TransactionDT for temporal ordering
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print(f"Data sorted by TransactionDT. Shape: {df.shape}")

    # 70/15/15 temporal train/val/test split
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
        df,
        target_col="isFraud",
        max_numeric=50,
        max_categorical=20,
    )

    print("\n==============================")
    print(" STEP 3: Training Custom TabTransformer")
    print("==============================")

    custom_metrics, custom_model = train_tabtransformer_custom(
        df, numeric_cols, categorical_cols, target_col="isFraud",
        num_epochs=25,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )

    print("\n==============================")
    print(" Custom Model Metrics (Val + Test)")
    print("==============================")
    for k, v in custom_metrics.items():
        print(f"{k:10s} : {v:.4f}")


if __name__ == "__main__":
    main()