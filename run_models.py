"""
Main script to run BOTH TabTransformer models:

1. Loads and merges IEEE-CIS fraud dataset
2. Automatically selects numeric + categorical features
3. Trains the TabTransformer (library version)
4. Trains the TabTransformer (custom PyTorch version)
5. Prints side-by-side performance metrics

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
from src.tabtransformer_library.train_library import train_tabtransformer_library
from src.tabtransformer_custom.train_custom import train_tabtransformer_custom


def main():

    print("\n==============================")
    print(" STEP 1: Loading Data")
    print("==============================")
    df = load_merged_train(data_dir="data")

    # -------------------------------------------------
    # DEBUG MODE: use a smaller subset for faster debugging
    # -------------------------------------------------
    DEBUG_MODE = True
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
    print(" STEP 3: Training Library TabTransformer")
    print("==============================")
    lib_metrics, lib_model = train_tabtransformer_library(
       df, numeric_cols, categorical_cols, target_col="isFraud"
    )

    print("\n==============================")
    print(" STEP 4: Training Custom TabTransformer")
    print("==============================")
    custom_metrics, custom_model = train_tabtransformer_custom(
       df, numeric_cols, categorical_cols, target_col="isFraud"
    )

    print("\n==============================")
    print(" STEP 5: Side-by-Side Comparison")
    print("==============================")
    keys = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    for k in keys:
        lm = lib_metrics.get(k, float("nan"))
        cm = custom_metrics.get(k, float("nan"))
        print(f"{k:10s} | Library: {lm:.4f}   | Custom: {cm:.4f}")


if __name__ == "__main__":
    main()