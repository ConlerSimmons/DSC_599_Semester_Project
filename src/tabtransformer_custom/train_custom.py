import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    batch_size: int = 512,
    num_epochs: int = 5,
    device: str = "cpu",
):
    """
    Train the from-scratch CustomTabTransformer on the given DataFrame using mini-batches.

    Args:
        df: pandas DataFrame containing the data.
        numeric_cols: list of numeric feature column names.
        categorical_cols: list of categorical feature column names.
        target_col: name of the target column (default: "isFraud").
        batch_size: batch size to use for DataLoader (default: 512).
        num_epochs: number of training epochs (default: 5).
        device: device string for torch.device (default: "cpu").

    Returns:
        metrics (dict), model (nn.Module)
    """

    device = torch.device(device)

    # ---------- Build vocab sizes and mappings for each categorical column ----------
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

    print(f"Using device for custom model: {device}")
    print(f"Training size: {train_size}, Validation size: {num_rows - train_size}")

    # ---------- Class imbalance weighting (NEG / POS) ----------
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = (num_neg.float() / num_pos.float()).to(device)
    print(f"Class imbalance → pos_weight = {pos_weight.item():.2f}")

    # ---------- Build DataLoaders ----------
    train_dataset = TensorDataset(
        x_num_train, x_cat_train, y_train
    )
    val_dataset = TensorDataset(
        x_num_val, x_cat_val, y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ---------- Model, loss, optimizer ----------
    model = CustomTabTransformer(
        vocab_sizes=vocab_sizes,
        num_numeric_features=len(numeric_cols),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Training loop (mini-batch) ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_x_num, batch_x_cat, batch_y in train_loader:
            batch_x_num = batch_x_num.to(device)
            batch_x_cat = batch_x_cat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x_num, batch_x_cat)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        print(f"[Custom Epoch {epoch}] avg_loss={avg_loss:.4f}")

    # ---------- Evaluation over validation set ----------
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_x_num, batch_x_cat, batch_y in val_loader:
            batch_x_num = batch_x_num.to(device)
            batch_x_cat = batch_x_cat.to(device)
            logits_val = model(batch_x_num, batch_x_cat)

            all_logits.append(logits_val.cpu())
            all_labels.append(batch_y.cpu())

    if len(all_logits) == 0:
        raise RuntimeError("No validation logits collected; check the DataLoader and data tensors.")

    logits_val = torch.cat(all_logits, dim=0)
    y_val_tensor = torch.cat(all_labels, dim=0)

    probs_val = torch.sigmoid(logits_val)
    preds_val = (probs_val > 0.5).float()

    y_true = y_val_tensor.numpy()
    y_pred = preds_val.numpy()
    y_score = probs_val.numpy()

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