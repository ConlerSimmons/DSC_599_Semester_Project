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
from src.utils_metrics import get_device


def train_gnn(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    num_epochs: int = 150,
    lr: float = 1e-3,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    early_stopping_patience: int = 20,
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

    # ---------- Scale numeric features ----------
    num_df = df[numeric_cols].fillna(0.0).astype("float32")
    means = num_df.mean()
    stds = num_df.std().replace(0, 1.0)
    num_df = (num_df - means) / stds

    df[numeric_cols] = num_df
    x_num = torch.tensor(num_df.values, dtype=torch.float32)

    # ---------- Categorical encoding ----------
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
    y = torch.tensor(
        df[target_col].values.astype("float32"),
        dtype=torch.float32,
    )

    # ---------- Graph ----------
    edge_index = build_transaction_graph(df)

    # ---------- Train/val/test split ----------
    if train_idx is None or val_idx is None:
        train_size = int(0.8 * num_nodes)
        train_idx = list(range(0, train_size))
        val_idx   = list(range(train_size, num_nodes))

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long) if test_idx is not None else None

    # GNN uses full-graph training — edge_index for 590k nodes is too large for MPS.
    # CPU is fast enough (~13 sec/epoch) and avoids MPS memory pressure.
    device = torch.device("cpu")
    print(f"Using device: {device}")

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
        dropout=0.10,
    ).to(device)

    # ---------- Class imbalance ----------
    pos = (y[train_idx] == 1).sum()
    neg = (y[train_idx] == 0).sum()
    pos_weight = (neg / pos).clamp(min=1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---------- Training loop with early stopping ----------
    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_num, x_cat, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])

        if not torch.isfinite(loss):
            print(f"[GNN Epoch {epoch}] loss became non-finite: {loss.item()}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute val PR-AUC for early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(x_num, x_cat, edge_index)
            val_probs  = torch.sigmoid(val_logits)
        ep_val_true  = y[val_idx].cpu().numpy()
        ep_val_score = val_probs[val_idx].cpu().numpy()
        ep_val_pr_auc = average_precision_score(ep_val_true, ep_val_score)

        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}  val_pr_auc={ep_val_pr_auc:.4f}")

        if ep_val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = ep_val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_pr_auc={best_val_pr_auc:.4f})")
                break

    if best_state is not None:
        best_state = {k: v.to(device) for k, v in best_state.items()}
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch}")

    # ---------- Evaluation ----------
    model.eval()
    with torch.no_grad():
        logits = model(x_num, x_cat, edge_index)
        probs = torch.sigmoid(logits)

    y_true = y[val_idx].cpu().numpy()
    y_score = probs[val_idx].cpu().numpy()

    # Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-8
    )
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype("int32")

    metrics = {
        "val_precision":      precision_score(y_true, y_pred, zero_division=0),
        "val_recall":         recall_score(y_true, y_pred, zero_division=0),
        "val_f1":             f1_score(y_true, y_pred, zero_division=0),
        "val_roc_auc":        roc_auc_score(y_true, y_score),
        "val_pr_auc":         average_precision_score(y_true, y_score),
        "val_best_threshold": float(best_threshold),
    }

    print("\n===== GNN — Validation =====")
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.4f}")

    # ---------- Test set evaluation ----------
    if test_idx is not None:
        y_test_true  = y[test_idx].cpu().numpy()
        y_test_score = probs[test_idx].cpu().numpy()
        y_test_pred  = (y_test_score >= best_threshold).astype("int32")

        test_metrics = {
            "test_precision": precision_score(y_test_true, y_test_pred, zero_division=0),
            "test_recall":    recall_score(y_test_true, y_test_pred, zero_division=0),
            "test_f1":        f1_score(y_test_true, y_test_pred, zero_division=0),
            "test_roc_auc":   roc_auc_score(y_test_true, y_test_score),
            "test_pr_auc":    average_precision_score(y_test_true, y_test_score),
        }
        print("\n===== GNN — Test =====")
        for k, v in test_metrics.items():
            print(f"{k:22s}: {v:.4f}")
        metrics.update(test_metrics)

    return metrics, model