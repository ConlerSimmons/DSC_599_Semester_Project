import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(df, numeric_cols, categorical_cols, target_col, device="cpu"):
    """
    Trains the Custom TabTransformer on a subset of the dataset.
    Returns:
        metrics (dict), model, y_true_val (list), y_pred_val (list)
    """

    # ------------ Prepare data tensors ------------ #
    x_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)
    x_cat = torch.tensor(df[categorical_cols].values, dtype=torch.long)
    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    N = len(df)
    train_size = int(0.8 * N)
    val_size = N - train_size

    x_num_train = x_num[:train_size]
    x_cat_train = x_cat[:train_size]
    y_train = y[:train_size]

    x_num_val = x_num[train_size:]
    x_cat_val = x_cat[train_size:]
    y_val = y[train_size:]

    # Dataset + loader
    train_ds = TensorDataset(x_num_train, x_cat_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    print(f"Training size: {train_size}, Validation size: {val_size}")

    # ------------ Handle class imbalance ------------ #
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    pos_weight_value = (num_neg / max(num_pos, 1))
    print(f"Class imbalance → pos_weight = {pos_weight_value:.2f}")

    pos_weight = torch.tensor([pos_weight_value], device=device)

    # ------------ Build model ------------ #
    model = CustomTabTransformer(
        num_numeric=len(numeric_cols),
        num_categories=500,            # large enough embedding table
        num_categorical=len(categorical_cols),
        dim=64,
        depth=3,
        heads=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------------ Training loop ------------ #
    model.train()
    for epoch in range(5):
        total_loss = 0.0

        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Custom Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f}")

    # ------------ Validation (predictions + metrics) ------------ #
    model.eval()
    with torch.no_grad():
        logits_val = model(x_num_val.to(device), x_cat_val.to(device))
        probs_val = torch.sigmoid(logits_val).cpu().numpy()
        preds_val = (probs_val >= 0.5).astype(int)
        y_true = y_val.cpu().numpy()

    # ------------ Metrics ------------ #
    precision = precision_score(y_true, preds_val, zero_division=0)
    recall = recall_score(y_true, preds_val, zero_division=0)
    f1 = f1_score(y_true, preds_val, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs_val) if len(set(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, probs_val)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return metrics, model, y_true, preds_val