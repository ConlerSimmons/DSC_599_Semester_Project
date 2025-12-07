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
    print(" STEP 3: Training Custom TabTransformer")
    print("==============================")
    custom_metrics, custom_model = train_tabtransformer_custom(
       df, numeric_cols, categorical_cols, target_col="isFraud"
    )

    print("\n==============================")
    print(" Custom Model Metrics")
    print("==============================")
    for k, v in custom_metrics.items():
        print(f"{k:10s} : {v:.4f}")

    # === Confusion Matrix Visualization ===
    if "confusion_matrix_fig" in custom_metrics:
        print("\n=== Confusion Matrix ===")
        fig = custom_metrics["confusion_matrix_fig"]
        fig.show()
    else:
        # Compute matrix manually if not returned
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import numpy as np

            print("\nComputing confusion matrix manually...")

            y_true = custom_metrics.get("y_true")
            y_pred = custom_metrics.get("y_pred")

            if y_true is not None and y_pred is not None:
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
                print("Confusion matrix could not be computed (y_true/y_pred missing).")

        except Exception as e:
            print(f"(Unable to compute confusion matrix manually: {e})")


if __name__ == "__main__":
    main()