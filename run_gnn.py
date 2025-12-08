"""
Runner script for the Custom GNN model.

This script:
1. Loads and merges the IEEE-CIS fraud dataset
2. Selects numeric + categorical features
3. Trains the GNN
4. Prints evaluation metrics

Run from project root:

    python run_gnn.py
"""

import torch

from src.data_loading import load_merged_train
from src.feature_selection import auto_select_features
from src.gnn_custom.train_gnn import train_gnn


def main():

    print("\n==============================")
    print(" STEP 1: Loading Data")
    print("==============================")
    df = load_merged_train(data_dir="data")

    # Optional debug mode
    DEBUG_MODE = True
    if DEBUG_MODE:
        df = df.sample(n=5000, random_state=42)
        print(f"DEBUG MODE ACTIVE: using {len(df)} rows instead of full dataset")

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
    print(" STEP 3: Training Custom GNN")
    print("==============================")
    gnn_metrics, gnn_model = train_gnn(
        df,
        numeric_cols,
        categorical_cols,
        target_col="isFraud"
    )

    print("\n==============================")
    print(" GNN Model Metrics")
    print("==============================")
    for k, v in gnn_metrics.items():
        print(f"{k:10s} : {v:.4f}")


if __name__ == "__main__":
    main()