import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from src.tabtransformer_custom.model_custom import CustomTabTransformer


def train_tabtransformer_custom(df, numeric_cols, categorical_cols, target_col="isFraud"):
    """
    Trains the custom TabTransformer on the provided dataframe.
    Returns metrics, model, y_true, y_pred.
    """

    # ============================================================
    # 1. Extract numeric tensor
    # ============================================================
    x_num = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float)

    # ============================================================
    # 2. Encode categorical columns properly
    # ============================================================
    encoded = []
    for col in categorical_cols:
        le = LabelEncoder()
        encoded.append(le.fit_transform(df[col].astype(str)))

    x_cat = torch.tensor(list(zip(*encoded)), dtype=torch.long)

    # ============================================================
    # 3. Target Tensor
    # ============================================================
    y = torch.tensor(df[target_col].values, dtype=torch.float)

    # ============================================================
    # 4. Train/Val Split
    # ============================================================
    train_size = int(0.8 * len(df))
    x_num_train, x_num_val = x_num[:train_size], x_num[train_size:]
    x_cat_train, x_cat_val = x_cat[:train_size], x_cat[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_ds = TensorDataset(x_num_train, x_cat_train, y_train)
    val_ds   = TensorDataset(x_num_val, x_cat_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ============================================================
    # 5. Model
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomTabTransformer(
        num_numeric=len(numeric_cols),
        num_categories=[df[col].astype(str).nunique() for col in categorical_cols],
        dim=32,
        depth=2,
        heads=4,
        mlp_dim=64
    ).to(device)

    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ============================================================
    # 6. Training Loop
    # ============================================================
    for epoch in range(5):
        model.train()
        total_loss = 0

        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb_num, xb_cat).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Custom Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f}")

    # ============================================================
    # 7. Evaluation
    # ============================================================
    model.eval()
    with torch.no_grad():
        logits = model(x_num_val.to(device), x_cat_val.to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).int()

    y_true = y_val.cpu()
    y_pred = preds.cpu()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
    }

    return metrics, model, y_true, probs