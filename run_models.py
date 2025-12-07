"""
Main script to run TabTransformer model:
Loads & merges IEEE-CIS fraud dataset
Selects features automatically
Converts to tensors
Trains the Custom TabTransformer
Prints performance metrics
"""

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.data_loading import load_merged_train
from src.feature_selection import auto_select_features
from src.tabtransformer_custom.train_custom import train_tabtransformer_custom


def main():

    print("\n==============================")
    print(" STEP 1: Loading Data")
    print("==============================")
    df = load_merged_train(data_dir="data")

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

    print("\nPreparing tensors...")

    X_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float32)

    X_cat = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        X_cat[:, i] = le.fit_transform(df[col].astype(str))
    X_cat = torch.tensor(X_cat, dtype=torch.long)

    y = torch.tensor(df["isFraud"].values, dtype=torch.float32)

    print("Tensor preparation complete.")

    print("\n==============================")
    print(" STEP 3: Training Custom TabTransformer")
    print("==============================")

    metrics, model = train_tabtransformer_custom(
        df,
        numeric_cols,
        categorical_cols,
        target_col="isFraud"
    )

    print("\n==============================")
    print(" Custom Model Metrics")
    print("==============================")
    for k, v in metrics.items():
        print(f"{k:10s} : {v:.4f}")


if __name__ == "__main__":
    main()