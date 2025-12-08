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

from src.gnn_custom.gnn_model import CustomGNN
from src.gnn_custom.graph_utils import build_transaction_graph


def train_gnn(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    num_epochs: int = 5,
    k_neighbors: int = 5,
    max_edges: int = 200_000,
):
    """
    Train a simple GNN for fraud detection on the IEEE-CIS dataset.

    IMPORTANT: we build the graph over *all* rows in `df`, and we train in a
    transductive way:
      - Node features and edge_index always cover the full set of nodes.
      - We use index masks (train_idx, val_idx) to decide which nodes contribute
        to the loss and which are used for validation metrics.

    This avoids the "index out of bounds" error where edge_index refers to nodes
    that aren't present in a sliced feature matrix.
    """

    device = torch.device("cpu")  # you can parameterize this later if needed

    # ------------------------------------------------------------------
    # 1. Build node features for ALL rows
    # ------------------------------------------------------------------
    # Numeric features
    x_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float32)

    # Categorical features: encode each column as integer indices
    cat_maps = []
    cat_tensors = []
    for col in categorical_cols:
        # Map unique categories in this column to consecutive ints
        uniques = df[col].astype(str).unique()
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_maps.append(mapping)
        cat_tensors.append(
            torch.tensor([mapping[v] for v in df[col].astype(str)], dtype=torch.long)
        )

    # Stack categorical columns -> shape [num_nodes, num_categorical]
    x_cat = torch.stack(cat_tensors, dim=1)

    # Labels
    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    num_nodes = x_num.shape[0]
    assert x_cat.shape[0] == num_nodes, "Numeric and categorical node counts must match"
    assert y.shape[0] == num_nodes, "Labels must match number of nodes"

    # ------------------------------------------------------------------
    # 2. Build graph over ALL nodes
    # ------------------------------------------------------------------
    # We treat each row as a node and connect nodes that share card numbers,
    # email domains, etc. You can tweak which columns are used here.
    # For now we'll just use a few "relationship-ish" columns if present.
    relation_cols = []
    for candidate in ["card1", "card2", "card3", "card5", "P_emaildomain", "R_emaildomain"]:
        if candidate in df.columns:
            relation_cols.append(candidate)

    edge_index = build_transaction_graph(
        df=df,
        categorical_cols=relation_cols,
        k_neighbors=k_neighbors,
        max_edges=max_edges,
    )

    # edge_index is shape [2, num_edges] with indices in [0, num_nodes-1]
    # Move everything to device
    x_num = x_num.to(device)
    x_cat = x_cat.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    # ------------------------------------------------------------------
    # 3. Train / validation split (BY INDICES ONLY)
    # ------------------------------------------------------------------
    train_size = int(0.8 * num_nodes)
    train_idx = torch.arange(0, train_size, dtype=torch.long, device=device)
    val_idx = torch.arange(train_size, num_nodes, dtype=torch.long, device=device)

    print(f"Training size: {len(train_idx)}, Validation size: {len(val_idx)}")

    # ------------------------------------------------------------------
    # 4. Model, loss, optimizer
    # ------------------------------------------------------------------
    model = CustomGNN(
        num_numeric=x_num.shape[1],
        num_categories_per_col=[len(m) for m in cat_maps],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Class imbalance: weight fraud class more heavily
    pos_count = (y[train_idx] == 1).sum().item()
    neg_count = (y[train_idx] == 0).sum().item()
    if neg_count == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    print(f"Class imbalance → pos_weight = {pos_weight_value:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 5. Training loop (full-graph, masked loss)
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward for ALL nodes
        logits_all = model(x_num, x_cat, edge_index).view(-1)

        # Compute loss ONLY on training nodes
        logits_train = logits_all[train_idx]
        y_train = y[train_idx]
        loss = criterion(logits_train, y_train)

        loss.backward()
        optimizer.step()

        print(f"[GNN Epoch {epoch}] loss={loss.item():.4f}")

    # ------------------------------------------------------------------
    # 6. Evaluation on validation nodes
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits_all = model(x_num, x_cat, edge_index).view(-1)
        probs_all = torch.sigmoid(logits_all)

    # Only use validation indices for metrics
    y_val = y[val_idx].detach().cpu().numpy()
    logits_val = logits_all[val_idx].detach().cpu().numpy()
    probs_val = probs_all[val_idx].detach().cpu().numpy()
    preds_val = (probs_val > 0.5).astype(float)

    metrics = {
        "precision": precision_score(y_val, preds_val, zero_division=0),
        "recall": recall_score(y_val, preds_val, zero_division=0),
        "f1": f1_score(y_val, preds_val, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probs_val),
        "pr_auc": average_precision_score(y_val, probs_val),
    }

    return metrics, model