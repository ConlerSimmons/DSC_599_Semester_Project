import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(df, numeric_cols, categorical_cols, target_col="isFraud"):

    # Build categorical index maps per column
    category_maps = []
    for col in categorical_cols:
        uniques = df[col].astype(str).unique()
        category_maps.append(len(uniques))

    # build tensors
    x_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float32)

    x_cat = torch.zeros((len(df), len(categorical_cols)), dtype=torch.long)
    for i, col in enumerate(categorical_cols):
        mapping = {v: idx for idx, v in enumerate(df[col].astype(str).unique())}
        x_cat[:, i] = torch.tensor([mapping[v] for v in df[col].astype(str)], dtype=torch.long)

    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    # train/val split
    train_size = int(len(df) * 0.8)
    x_num_train, x_num_val = x_num[:train_size], x_num[train_size:]
    x_cat_train, x_cat_val = x_cat[:train_size], x_cat[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    device = torch.device("cpu")
    model = CustomTabTransformer(
        num_numeric=len(numeric_cols),
        num_categories=category_maps,
    ).to(device)

    pos_weight = torch.tensor([(y_train == 1).sum() / (y_train == 0).sum()], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    for epoch in range(1, 6):
        model.train()
        optimizer.zero_grad()
        preds = model(x_num_train.to(device), x_cat_train.to(device))
        loss = criterion(preds, y_train.to(device))
        loss.backward()
        optimizer.step()
        print(f"[Custom Epoch {epoch}] loss={loss.item():.4f}")

    # evaluate
    model.eval()
    with torch.no_grad():
        logits = model(x_num_val.to(device), x_cat_val.to(device))
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    y_true = y_val.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_score = probs.cpu().numpy()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }

    return metrics, model