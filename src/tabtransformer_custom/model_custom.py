import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(
    X_num,
    X_cat,
    y,
    batch_size=64,
    lr=1e-3,
    num_epochs=5,
    device=None,
):
    """
    Train the custom TabTransformer model.

    Returns:
        metrics: dict containing precision/recall/f1/pr_auc/roc_auc,
                 AND y_true, y_pred (logits) for confusion matrix.
        model:   trained model
    """

    # -------------------------------
    # 1) Setup device
    # -------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for custom model: {device}")

    # -------------------------------
    # 2) Train/Val split
    # -------------------------------
    N = len(y)
    train_size = int(0.8 * N)
    val_size = N - train_size
    print(f"Training size: {train_size}, Validation size: {val_size}")

    X_num_train = X_num[:train_size]
    X_num_val = X_num[train_size:]

    X_cat_train = X_cat[:train_size]
    X_cat_val = X_cat[train_size:]

    y_train = y[:train_size]
    y_val = y[train_size:]

    # -------------------------------
    # 3) DataLoader setup
    # -------------------------------
    train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
    val_dataset = TensorDataset(X_num_val, X_cat_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # 4) Build model
    # -------------------------------
    vocab_sizes = [int(X_cat[:, i].max().item()) + 1 for i in range(X_cat.shape[1])]
    model = CustomTabTransformer(
        vocab_sizes=vocab_sizes,
        num_numeric_features=X_num.shape[1],
        d_token=32,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    ).to(device)

    # -------------------------------
    # 5) Loss + Optimizer
    # -------------------------------
    # Compute class imbalance weighting
    pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance → pos_weight = {pos_weight_value:.2f}")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------------
    # 6) Training Loop
    # -------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for x_num_batch, x_cat_batch, y_batch in train_loader:
            x_num_batch = x_num_batch.to(device).float()
            x_cat_batch = x_cat_batch.to(device).long()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            logits = model(x_num_batch, x_cat_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Custom Epoch {epoch}] loss={avg_loss:.4f}")

    # -------------------------------
    # 7) Validation — collect logits
    # -------------------------------
    model.eval()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x_num_batch, x_cat_batch, y_batch in val_loader:
            x_num_batch = x_num_batch.to(device).float()
            x_cat_batch = x_cat_batch.to(device).long()
            y_batch = y_batch.to(device).float()

            logits = model(x_num_batch, x_cat_batch)

            y_true_list.append(y_batch)
            y_pred_list.append(logits)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    # -------------------------------
    # 8) Convert predictions
    # -------------------------------
    y_true_np = y_true.cpu().numpy()
    y_prob_np = torch.sigmoid(y_pred).cpu().numpy()
    y_pred_labels = (y_prob_np >= 0.5).astype(int)

    # -------------------------------
    # 9) Compute Metrics
    # -------------------------------
    precision = precision_score(y_true_np, y_pred_labels, zero_division=0)
    recall = recall_score(y_true_np, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_labels, zero_division=0)
    pr_auc = average_precision_score(y_true_np, y_prob_np)
    roc_auc = roc_auc_score(y_true_np, y_prob_np)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "y_true": y_true,     # needed for confusion matrix
        "y_pred": y_pred,     # logits
    }

    return metrics, model