import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils_metrics import compute_metrics, print_metrics
from src.tabtransformer_custom.model_custom import CustomTabTransformer


class FraudDataset(Dataset):
    """
    A simple PyTorch Dataset that stores:
    - numeric features (X_num)
    - categorical features (X_cat)
    - labels (y)
    """

    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_num[idx], dtype=torch.float32),
            torch.tensor(self.X_cat[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def train_tabtransformer_custom(
    df,
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
    device: str = None,
):
    """
    Train the custom TabTransformer:
    - preprocess df
    - encode categoricals
    - standard train/val split
    - class-weighted loss (handles fraud imbalance)
    - custom transformer training
    - evaluation + metrics
    """

    # ---------------------------
    # DEVICE SETUP
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for custom model: {device}")

    # ---------------------------
    # SELECT COLUMNS
    # ---------------------------
    df = df[numeric_cols + categorical_cols + [target_col]].copy()

    # ---------------------------
    # IMPUTE NUMERICAL FEATURES
    # ---------------------------
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # ---------------------------
    # ENCODE CATEGORICAL FEATURES
    # ---------------------------
    vocab_sizes = []
    for col in categorical_cols:
        df[col] = df[col].astype("object").fillna("missing").astype("category")
        vocab_sizes.append(df[col].cat.categories.size)
        df[col] = df[col].cat.codes

    # Convert df → numpy
    X_num = df[numeric_cols].to_numpy().astype("float32")
    X_cat = df[categorical_cols].to_numpy().astype("int64")
    y = df[target_col].to_numpy().astype("int64")

    # ---------------------------
    # TRAIN/VAL SPLIT
    # ---------------------------
    (
        X_num_train, X_num_val,
        X_cat_train, X_cat_val,
        y_train, y_val
    ) = train_test_split(
        X_num, X_cat, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ---------------------------
    # DATALOADERS
    # ---------------------------
    train_loader = DataLoader(
        FraudDataset(X_num_train, X_cat_train, y_train),
        batch_size=2048,
        shuffle=True,
    )

    val_loader = DataLoader(
        FraudDataset(X_num_val, X_cat_val, y_val),
        batch_size=2048,
        shuffle=False,
    )

    # ---------------------------
    # MODEL INITIALIZATION
    # ---------------------------
    model = CustomTabTransformer(
        vocab_sizes=vocab_sizes,
        num_numeric_features=len(numeric_cols),
    ).to(device)

    # ---------------------------
    # CLASS-WEIGHTED LOSS
    # ---------------------------
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight_val = n_neg / max(n_pos, 1)  # avoid divide-by-zero

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Training size: {len(y_train)}, Validation size: {len(y_val)}")
    print(f"Class imbalance → pos_weight = {pos_weight_val:.2f}")

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    for epoch in range(5):  # small number for initial test
        model.train()
        total_loss = 0.0

        for Xn, Xc, yy in train_loader:
            Xn = Xn.to(device)
            Xc = Xc.to(device)
            yy = yy.float().to(device)

            optimizer.zero_grad()
            logits = model(Xn, Xc)
            loss = criterion(logits, yy)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yy.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Custom Epoch {epoch+1}] loss={avg_loss:.4f}")

    # ---------------------------
    # EVALUATION
    # ---------------------------
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for Xn, Xc, yy in val_loader:
            Xn = Xn.to(device)
            Xc = Xc.to(device)

            logits = model(Xn, Xc)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yy.numpy())

    y_proba = 1 / (1 + np.exp(-np.concatenate(all_logits)))
    y_true = np.concatenate(all_labels)

    # ---------------------------
    # METRICS
    # ---------------------------
    metrics = compute_metrics(y_true, y_proba)
    print_metrics("Custom TabTransformer", metrics)

    return metrics, model