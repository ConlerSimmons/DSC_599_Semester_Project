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
    precision_recall_curve,
)

from src.gnn_custom.graph_utils import build_transaction_graph
from src.gnn_custom.gnn_model import SimpleGNN


def train_gnn(
    df,
    numeric_cols,
    categorical_cols,
    target_col="isFraud",
    num_epochs=5,
    lr=3e-4,
):
    """
    Train the SimpleGNN model using identity-based edges only.
    """

    # ---------- Select + reset index ----------
    cols = numeric_cols + categorical_cols + [target_col]
    df = df[cols].copy().reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("train_gnn: dataframe is empty")

    # ---------- Normalize numeric ----------
    num_df = df[numeric_cols].fillna(0.0).astype("float32")
    means = num_df.mean()
    stds = num_df.std().replace(0, 1.0)
    num_df = (num_df - means) / stds
    df[numeric_cols] = num_df

    x_num = torch.tensor(num_df.values, dtype=torch.float32)

    # ---------- Encode categorical ----------
    cat_sizes = []
    cat_arrays = []

    for col in categorical_cols:
        vals = df[col].astype(str)
        uniq = sorted(vals.unique())
        map_dict = {v: i for i, v in enumerate(uniq)}
        cat_sizes.append(len(uniq))
        cat_arrays.append(vals.map(map_dict).astype("int64").values)

    if cat_arrays:
        x_cat_np = np.stack(cat_arrays, axis=1)
        x_cat = torch.tensor(x_cat_np, dtype=torch.long)
    else:
        x_cat = torch.empty((num_nodes, 0), dtype=torch.long)

    # ---------- Target ----------
    y = torch.tensor(df[target_col].values.astype("float32"))

    # ---------- Graph (IDENTITY ONLY) ----------
    edge_index = build_transaction_graph(df, numeric_cols=None)

    # ---------- Train/val ----------
    train_size = int(num_nodes * 0.8)
    train_idx = torch.arange(0, train_size)
    val_idx = torch.arange(train_size, num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_num, x_cat, y, edge_index = (
        x_num.to(device),
        x_cat.to(device),
        y.to(device),
        edge_index.to(device),
    )

    # ---------- Model ----------
    model = SimpleGNN(
        num_numeric=len(numeric_cols),
        num_categories_per_col=cat_sizes,
        embed_dim=32,
        hidden_dim=128,
    ).to(device)

    # ---------- Loss ----------
    pos = (y[train_idx] == 1).sum()
    neg = (y[train_idx] == 0).sum()

    pos_weight = (neg / pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------- Training ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_num, x_cat, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])

        if not torch.isfinite(loss):
            print(f"[GNN Epoch {epoch}] NON-FINITE LOSS → stopping")
            break

        loss.backward()
        optimizer.step()

        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}")

    # ---------- Validation ----------
    model.eval()
    with torch.no_grad():
        logits = model(x_num, x_cat, edge_index)
        probs = torch.sigmoid(logits)

    y_true = y[val_idx].cpu().numpy()
    y_score = probs[val_idx].cpu().numpy()

    # Learn threshold maximizing F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype("int32")

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "best_threshold": float(best_threshold),
    }

    return metrics, model