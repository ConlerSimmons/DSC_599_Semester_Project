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

from .model_custom import CustomTabTransformer


def train_tabtransformer_custom(X_num, X_cat, y, num_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for custom model: {device}")

    # -----------------------------
    # Train/validation split
    # -----------------------------
    total_rows = X_num.shape[0]
    val_size = int(0.2 * total_rows)
    train_size = total_rows - val_size

    X_num_train = X_num[:train_size].to(device)
    X_cat_train = X_cat[:train_size].to(device)
    y_train = y[:train_size].to(device)

    X_num_val = X_num[train_size:].to(device)
    X_cat_val = X_cat[train_size:].to(device)
    y_val = y[train_size:].to(device)

    print(f"Training size: {train_size}, Validation size: {val_size}")

    # -----------------------------
    # Handle class imbalance
    # -----------------------------
    fraud_ratio = y_train.sum() / len(y_train)
    pos_weight = (1 - fraud_ratio) / fraud_ratio
    pos_weight = pos_weight.item()
    print(f"Class imbalance → pos_weight = {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    model = CustomTabTransformer(
        num_numeric=X_num.shape[1],
        num_categories=X_cat.max().item() + 1,
        num_categorical=X_cat.shape[1],
        dim=64,
        depth=4,
        heads=8,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_num_train, X_cat_train)
        loss = criterion(logits.squeeze(), y_train)

        loss.backward()
        optimizer.step()

        print(f"[Custom Epoch {epoch}] loss={loss.item():.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    with torch.no_grad():
        logits_val = model(X_num_val, X_cat_val).squeeze()
        prob_val = torch.sigmoid(logits_val)

    y_true = y_val.cpu()
    y_pred_prob = prob_val.cpu()
    y_pred = (y_pred_prob >= 0.5).int()

    # -----------------------------
    # Metrics
    # -----------------------------
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred_prob,  # return probabilities for confusion matrix
    }

    return metrics, model