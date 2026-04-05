import os
import pandas as pd


def load_merged_train(data_dir: str = "data") -> pd.DataFrame:
    """
    I load and merge the IEEE-CIS training data from:
    - data/raw/train_transaction.csv
    - data/raw/train_identity.csv

    On first run, saves the merged result to data/interim/merged_train.parquet.
    Subsequent runs load from cache for speed.

    Returns a single DataFrame with all training features and the isFraud label.
    """
    cache_path = os.path.join(data_dir, "interim", "merged_train.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached merged data from {cache_path} ...")
        df = pd.read_parquet(cache_path)
        print(f"Loaded from cache. Shape: {df.shape}")
        return df

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
    print(f"Saving merged data to {cache_path} ...")
    df.to_parquet(cache_path, index=False)
    print("Cache saved.")

    return df