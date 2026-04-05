import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.tabtransformer_custom.model_custom import CustomTabTransformer
from src.utils_metrics import get_device


def train_tabtransformer_custom(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    batch_size: int = 512,
    num_epochs: int = 15,
    device=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    early_stopping_patience: int = 5,
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

    device = get_device() if device is None else torch.device(device)
    print(f"Using device: {device}")

    # ---------- Build vocab sizes and mappings for each categorical column ----------
    vocab_sizes = []
    cat_mappings = []

    for col in categorical_cols:
        col_values = df[col].astype(str)
        uniques = list(col_values.unique())
        mapping = {v: idx for idx, v in enumerate(uniques)}
        vocab_sizes.append(len(uniques))
        cat_mappings.append(mapping)

    # ---------- Numeric tensor (fit scaler on train only) ----------
    num_data = df[numeric_cols].fillna(0).values
    if train_idx is not None:
        scaler = StandardScaler()
        num_data[train_idx] = scaler.fit_transform(num_data[train_idx])
        if val_idx is not None:
            num_data[val_idx] = scaler.transform(num_data[val_idx])
        if test_idx is not None:
            num_data[test_idx] = scaler.transform(num_data[test_idx])
    else:
        scaler = StandardScaler()
        num_data = scaler.fit_transform(num_data)

    x_num = torch.tensor(num_data, dtype=torch.float32)

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
    if train_idx is None or val_idx is None:
        train_size = int(0.8 * num_rows)
        train_idx = list(range(0, train_size))
        val_idx   = list(range(train_size, num_rows))

    x_num_train = x_num[train_idx]
    x_num_val   = x_num[val_idx]
    x_cat_train = x_cat[train_idx]
    x_cat_val   = x_cat[val_idx]
    y_train     = y[train_idx]
    y_val       = y[val_idx]

    print(f"Using device for custom model: {device}")
    print(f"Training size: {len(train_idx)}, Validation size: {len(val_idx)}")

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
        num_workers=4,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # ---------- Model, loss, optimizer ----------
    model = CustomTabTransformer(
        vocab_sizes=vocab_sizes,
        num_numeric_features=len(numeric_cols),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # AdamW decouples weight decay from the gradient update (correct for transformers)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # ReduceLROnPlateau drops LR only when val_pr_auc stalls — avoids the aggressive
    # decay of CosineAnnealingLR which was cutting LR to ~1e-5 by epoch 10-12
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    # ---------- Training loop (mini-batch) with early stopping ----------
    if device.type == "mps":
        torch.mps.empty_cache()
    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Explicitly free batch tensors — MPS accumulates if not deleted
            del batch_x_num, batch_x_cat, batch_y, logits, loss
            if device.type == "mps" and num_batches % 100 == 0:
                torch.mps.empty_cache()

        avg_loss = running_loss / max(num_batches, 1)
        # Free MPS cached memory after each epoch to prevent accumulation
        if device.type == "mps":
            torch.mps.empty_cache()

        # Quick val PR-AUC check for early stopping
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for bx_num, bx_cat, by in val_loader:
                out = model(bx_num.to(device), bx_cat.to(device)).cpu()
                val_logits_list.append(out)
                val_labels_list.append(by)
                del out
        ep_val_probs = torch.sigmoid(torch.cat(val_logits_list)).numpy()
        ep_val_true  = torch.cat(val_labels_list).numpy()
        ep_val_pr_auc = average_precision_score(ep_val_true, ep_val_probs)

        # ReduceLROnPlateau needs the monitored metric
        scheduler.step(ep_val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Custom Epoch {epoch}] loss={avg_loss:.4f}  val_pr_auc={ep_val_pr_auc:.4f}  lr={current_lr:.2e}")

        if ep_val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = ep_val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            # Save to CPU to avoid accumulating MPS allocations across epochs
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_pr_auc={best_val_pr_auc:.4f})")
                break

    if best_state is not None:
        # best_state was saved on CPU — move back to device before loading
        best_state = {k: v.to(device) for k, v in best_state.items()}
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch}")

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

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_threshold = float(thresholds[f1_scores.argmax()])
    y_pred = (y_score >= best_threshold).astype(float)

    metrics = {
        "val_precision":     precision_score(y_true, y_pred, zero_division=0),
        "val_recall":        recall_score(y_true, y_pred, zero_division=0),
        "val_f1":            f1_score(y_true, y_pred, zero_division=0),
        "val_roc_auc":       roc_auc_score(y_true, y_score),
        "val_pr_auc":        average_precision_score(y_true, y_score),
        "val_best_threshold": best_threshold,
    }

    print("\n===== Custom TabTransformer — Validation =====")
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.4f}")

    # ---------- Test set evaluation (batched to avoid OOM) ----------
    if test_idx is not None:
        test_dataset = TensorDataset(x_num[test_idx], x_cat[test_idx], y[test_idx])
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        y_test_true  = y[test_idx].numpy()

        model.eval()
        test_logits_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in test_loader:
                out = model(bx_num.to(device), bx_cat.to(device)).cpu()
                test_logits_list.append(out)
                del out
        y_test_score = torch.sigmoid(torch.cat(test_logits_list)).numpy()
        y_test_pred = (y_test_score >= best_threshold).astype(float)

        test_metrics = {
            "test_precision": precision_score(y_test_true, y_test_pred, zero_division=0),
            "test_recall":    recall_score(y_test_true, y_test_pred, zero_division=0),
            "test_f1":        f1_score(y_test_true, y_test_pred, zero_division=0),
            "test_roc_auc":   roc_auc_score(y_test_true, y_test_score),
            "test_pr_auc":    average_precision_score(y_test_true, y_test_score),
        }
        print("\n===== Custom TabTransformer — Test =====")
        for k, v in test_metrics.items():
            print(f"{k:22s}: {v:.4f}")
        metrics.update(test_metrics)

    return metrics, model