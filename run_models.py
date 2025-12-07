"""
Main script to run TabTransformer model:

1. Loads and merges IEEE-CIS fraud dataset
2. Automatically selects numeric + categorical features
3. Converts data into tensors
4. Trains the TabTransformer (custom PyTorch version)
5. Prints performance metrics + confusion matrix

Run from project root:

    python run_models.py
"""

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Allow OmegaConf DictConfig objects to be unpickled safely with torch.load(weights_only=True)
try:
    from omegaconf.dictconfig import DictConfig
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DictConfig])
except Exception:
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

    # ======================================================
    # Convert numeric + categorical + target → tensors
    # ======================================================
    print("\nPreparing tensors...")

    # --- numeric ---
    X_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float)

    # --- categorical: integer labels ---
    X_cat = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        X_cat[:, i] = le.fit_transform(df[col].astype(str))

    X_cat = torch.tensor(X_cat, dtype=torch.long)

    # --- target ---
    y = torch.tensor(df["isFraud"].values, dtype=torch.float)

    print("Tensor preparation complete.")

    print("\n==============================")
    print(" STEP 3: Training Custom TabTransformer")
    print("==============================")
    custom_metrics, custom_model = train_tabtransformer_custom(
        X_num,
        X_cat,
        y
    )

    print("\n==============================")
    print(" Custom Model Metrics")
    print("==============================")
    for k, v in custom_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:10s} : {v:.4f}")

    # =====================================================
    # Confusion Matrix Visualization
    # =====================================================
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        y_true = custom_metrics.get("y_true")
        y_pred = custom_metrics.get("y_pred")

        print("\n=== Confusion Matrix ===")

        if y_true is not None and y_pred is not None:
            y_true = y_true.numpy()
            y_pred = (y_pred.numpy() >= 0.5).astype(int)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            fig.tight_layout()
            fig.show()

        else:
            print("Confusion matrix could not be computed — metrics missing y_true/y_pred.")

    except Exception as e:
        print(f"(Unable to compute confusion matrix: {e})")


if __name__ == "__main__":
    main()