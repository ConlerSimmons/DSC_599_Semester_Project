from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from .gnn_model import CustomGNN
from .graph_utils import build_edge_index_from_key, add_self_loops


def train_gnn(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str = "isFraud",
    epochs: int = 5,
    lr: float = 1e-3,
) -> Tuple[dict, CustomGNN]:
    """
    I train the CustomGNN on the provided dataframe and return (metrics, model).

    - df: pre-merged train dataframe (optionally already sub-sampled by caller)
    - numeric_cols / categorical_cols: lists chosen by feature_selection
    - target_col: fraud label
    """

    # 1) Encode categorical columns to integer codes per column
    cat_cardinalities = []
    encoded_cat_cols = []

    for col in categorical_cols:
        codes, uniques = pd.factorize(df[col].astype(str), sort=True)
        enc_col = f"{col}__encoded"
        df[enc_col] = codes
        encoded_cat_cols.append(enc_col)
        cat_cardinalities.append(len(uniques))

    # 2) Build tensors
    x_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float32)
    x_cat = torch.tensor(df[encoded_cat_cols].values, dtype=torch.long)
    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    N = len(df)

    # 3) Train/val split (random 80/20 split)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)

    train_size = int(0.8 * N)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:]

    x_num_train, x_cat_train, y_train = x_num[train_idx], x_cat[train_idx], y[train_idx]
    x_num_val, x_cat_val, y_val = x_num[val_idx], x_cat[val_idx], y[val_idx]

    # 4) Build graph edges on the full dataframe (indexes 0..N-1)
    df_reset = df.reset_index(drop=True)
    edge_index = build_edge_index_from_key(df_reset, key_col="card1")
    device = torch.device("cpu")

    # I add self-loops once here so every layer has them
    edge_index = add_self_loops(edge_index, num_nodes=N, device=device)

    model = CustomGNN(
        num_numeric=len(numeric_cols),
        num_categories=cat_cardinalities,
        emb_dim=16,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    # Class imbalance weighting based on training labels
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = (num_neg / (num_pos + 1e-8)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Move tensors to device
    edge_index = edge_index.to(device)
    x_num_train = x_num_train.to(device)
    x_cat_train = x_cat_train.to(device)
    y_train = y_train.to(device)
    x_num_val = x_num_val.to(device)
    x_cat_val = x_cat_val.to(device)
    y_val = y_val.to(device)

    # 5) Full-batch training (simple but fine for a debug-sized subset)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_num_train, x_cat_train, edge_index)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}")

    # 6) Evaluation
    model.eval()
    with torch.no_grad():
        logits_val = model(x_num_val, x_cat_val, edge_index)
        probs_val = torch.sigmoid(logits_val)
        preds_val = (probs_val > 0.5).float()

    y_true = y_val.cpu().numpy()
    y_pred = preds_val.cpu().numpy()
    y_score = probs_val.cpu().numpy()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }

    return metrics, model