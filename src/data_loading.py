import os
import pandas as pd


def load_merged_train(data_dir: str = "data") -> pd.DataFrame:
    """
    I load and merge the IEEE-CIS training data from:
    - data/raw/train_transaction.csv
    - data/raw/train_identity.csv

    Returns a single DataFrame with all training features and the isFraud label.
    """
    raw_dir = os.path.join(data_dir, "raw")
    trans_path = os.path.join(raw_dir, "train_transaction.csv")
    ident_path = os.path.join(raw_dir, "train_identity.csv")

    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Could not find {trans_path}")
    if not os.path.exists(ident_path):
        raise FileNotFoundError(f"Could not find {ident_path}")

    print(f"Loading {trans_path} ...")
    train_trans = pd.read_csv(trans_path)

    print(f"Loading {ident_path} ...")
    train_ident = pd.read_csv(ident_path)

    print("Merging on TransactionID ...")
    df = train_trans.merge(train_ident, how="left", on="TransactionID")

    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' not found in merged DataFrame.")

    print(f"Merged training data shape: {df.shape}")
    return df