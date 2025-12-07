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

from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(df, numeric_cols, categorical_cols, target_col: str = "isFraud"):
    """
    Train the from-scratch CustomTabTransformer on the given DataFrame.

    Returns:
        metrics (dict), model (nn.Module)
    """

    # ---------- Build vocab sizes for each categorical column ----------
    vocab_sizes = []
    cat_mappings = []

    for col in categorical_cols:
        col_values = df[col].astype(str)
        uniques = list(col_values.unique())
        mapping = {v: idx for idx, v in enumerate(uniques)}
        vocab_sizes.append(len(uniques))
        cat_mappings.append(mapping)

    # ---------- Numeric tensor ----------
    x_num = torch.tensor(
        df[numeric_cols].fillna(0).values,
        dtype=torch.float32,
    )

    # ---------- Categorical tensor ----------
    num_rows = len(df)
    num_cat_cols = len(categorical_cols)
    x_cat = torch.zeros((num_rows, num_cat_cols), dtype=torch.long)

    for i, col in enumerate(categorical_cols):
        col_values = df[col].astype(str)
        mapping = cat_mappings[i]
        x_cat[:, i] = torch.tensor([mapping[v] for v in col_values], dtype=torch.long)

    # ---------- Target tensor ----------
    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    # ---------- Train/validation split ----------
    train_size = int(0.8 * num_rows)
    x_num_train, x_num_val = x_num[:train_size], x_num[train_size:]
    x_cat_train, x_cat_val = x_cat[:train_size], x_cat[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    device = torch.device("cpu")
    print(f"Using device for custom model: {device}")
    print(f"Training size: {train_size}, Validation size: {num_rows - train_size}")

    # ---------- Class imbalance weighting (NEG / POS) ----------
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = (num_neg.float() / num_pos.float()).to(device)
    print(f"Class imbalance → pos_weight = {pos_weight.item():.2f}")

    # ---------- Model, loss, optimizer ----------
    model = CustomTabTransformer(
        vocab_sizes=vocab_sizes,
        num_numeric_features=len(numeric_cols),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Training loop ----------
    for epoch in range(1, 6):
        model.train()
        optimizer.zero_grad()
        logits = model(x_num_train.to(device), x_cat_train.to(device))
        loss = criterion(logits, y_train.to(device))
        loss.backward()
        optimizer.step()
        print(f"[Custom Epoch {epoch}] loss={loss.item():.4f}")

    # ---------- Evaluation ----------
    model.eval()
    with torch.no_grad():
        logits_val = model(x_num_val.to(device), x_cat_val.to(device))
        probs_val = torch.sigmoid(logits_val)
        preds_val = (probs_val > 0.5).float()

    y_true = y_val.cpu().numpy()
    y_pred = preds_val.cpu().numpy()
    y_score = probs_val.cpu().numpy()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_score),
        "pr_auc":    average_precision_score(y_true, y_score),
    }

    print("\n===== Custom TabTransformer Metrics =====")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    return metrics, model