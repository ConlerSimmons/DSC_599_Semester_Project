import numpy as np
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

from src.gnn_custom.graph_utils import build_transaction_graph
from src.gnn_custom.gnn_model import SimpleGNN


def train_gnn(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    num_epochs: int = 5,
    k_neighbors: int = 5,
):
    """
    Train the SimpleGNN model on the transaction data.

    Parameters
    ----------
    df : pandas.DataFrame
    numeric_cols : list[str]
    categorical_cols : list[str]
    target_col : str
    num_epochs : int
    k_neighbors : int

    Returns
    -------
    metrics : dict
    model : nn.Module
    """
    # Keep only the relevant columns and reset index → node ids 0..N-1
    cols = numeric_cols + categorical_cols + [target_col]
    df = df[cols].copy().reset_index(drop=True)

    num_nodes = len(df)
    if num_nodes == 0:
        raise ValueError("train_gnn: dataframe is empty")

    # ---------- Build tensors ----------
    # Numeric
    x_num = torch.tensor(
        df[numeric_cols].fillna(0.0).values.astype("float32"),
        dtype=torch.float32,
    )

    # Categorical → per-column mapping
    cat_sizes = []
    cat_arrays = []
    for col in categorical_cols:
        values = df[col].astype(str)
        uniques = sorted(values.unique())
        mapping = {v: idx for idx, v in enumerate(uniques)}
        cat_sizes.append(len(uniques))
        cat_arrays.append(values.map(mapping).astype("int64").values)

    if cat_arrays:
        x_cat_np = np.stack(cat_arrays, axis=1)  # (N, num_categorical)
        x_cat = torch.tensor(x_cat_np, dtype=torch.long)
    else:
        x_cat = torch.empty((num_nodes, 0), dtype=torch.long)

    # Target
    y = torch.tensor(
        df[target_col].values.astype("float32"),
        dtype=torch.float32,
    )

    # ---------- Build graph ----------
    edge_index = build_transaction_graph(df, numeric_cols, k_neighbors=k_neighbors)

    # ---------- Train / val split ----------
    train_size = int(0.8 * num_nodes)
    train_idx = torch.arange(0, train_size)
    val_idx = torch.arange(train_size, num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_num = x_num.to(device)
    x_cat = x_cat.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    # ---------- Model ----------
    model = SimpleGNN(
        num_numeric=len(numeric_cols),
        num_categories_per_col=cat_sizes,
        embed_dim=16,
        hidden_dim=64,
    ).to(device)

    # Handle class imbalance similar to TabTransformer run
    pos = (y[train_idx] == 1).sum()
    neg = (y[train_idx] == 0).sum()
    pos_weight = (neg / pos).clamp(min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Training loop (full-batch) ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_num, x_cat, edge_index)  # (N,)
        loss = criterion(logits[train_idx], y[train_idx])

        loss.backward()
        optimizer.step()

        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}")

    # ---------- Evaluation on validation split ----------
    model.eval()
    with torch.no_grad():
        logits = model(x_num, x_cat, edge_index)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    y_true = y[val_idx].cpu().numpy()
    y_pred = preds[val_idx].cpu().numpy()
    y_score = probs[val_idx].cpu().numpy()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }

    return metrics, model