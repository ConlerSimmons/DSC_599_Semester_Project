import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
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
    target_col: str = "isFraud",
    num_epochs: int = 5,
    k_neighbors: int = 5,
    lr: float = 1e-3,
):
    """
    Train the SimpleGNN model on the transaction data.
    """

    # ---------- Column subset & index reset ----------
    cols = numeric_cols + categorical_cols + [target_col]
    df = df[cols].copy().reset_index(drop=True)

    num_nodes = len(df)
    if num_nodes == 0:
        raise ValueError("train_gnn: dataframe is empty")

    # ---------- Numeric features: standardize ----------
    num_df = df[numeric_cols].fillna(0.0).astype("float32")
    means = num_df.mean()
    stds = num_df.std().replace(0, 1.0)
    num_df = (num_df - means) / stds
    df[numeric_cols] = num_df

    x_num = torch.tensor(num_df.values, dtype=torch.float32)

    # ---------- Categorical → integer encodings ----------
    cat_sizes = []
    cat_arrays = []
    for col in categorical_cols:
        values = df[col].astype(str)
        uniques = sorted(values.unique())
        mapping = {v: idx for idx, v in enumerate(uniques)}
        cat_sizes.append(len(uniques))
        cat_arrays.append(values.map(mapping).astype("int64").values)

    if cat_arrays:
        x_cat_np = np.stack(cat_arrays, axis=1)
        x_cat = torch.tensor(x_cat_np, dtype=torch.long)
    else:
        x_cat = torch.empty((num_nodes, 0), dtype=torch.long)

    # ---------- Target ----------
    y = torch.tensor(df[target_col].values.astype("float32"))

    # ---------- Build Graph ----------
    edge_index = build_transaction_graph(
        df=df,
        numeric_cols=numeric_cols,
        k_neighbors=k_neighbors,
    )

    # ---------- Train / validation split ----------
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
        embed_dim=32,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    # ---------- Class imbalance weighting ----------
    pos = (y[train_idx] == 1).sum()
    neg = (y[train_idx] == 0).sum()
    pos_weight = (neg / pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    step_size = max(num_epochs // 3, 1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    # ---------- Training Loop ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_num, x_cat, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])

        if not torch.isfinite(loss):
            print(f"[GNN Epoch {epoch}] Non-finite loss: {loss.item()}")
            break

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}")

    # ---------- Evaluation ----------
    model.eval()
    with torch.no_grad():
        logits = model(x_num, x_cat, edge_index)
        probs = torch.sigmoid(logits)

    y_true = y[val_idx].cpu().numpy()
    y_score = probs[val_idx].cpu().numpy()

    # Precision-Recall curve for threshold optimization
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-8
    )
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype("int32")

    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1_val = f1_score(y_true, y_pred, zero_division=0)
    roc_auc_val = roc_auc_score(y_true, y_score)
    pr_auc_val = average_precision_score(y_true, y_score)

    metrics = {
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "best_threshold": float(best_threshold),
    }

    return metrics, model