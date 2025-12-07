import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# 1. DEFINE THE CUSTOM TABTRANSFORMER MODEL
# ============================================================
class CustomTabTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        num_numeric_features,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        # -----------------------------
        # Embedding layers for each categorical feature
        # -----------------------------
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab, d_token) for vocab in vocab_sizes]
        )

        # -----------------------------
        # Numeric feature projection
        # -----------------------------
        self.numeric_projection = nn.Linear(num_numeric_features, d_token)

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # -----------------------------
        # Output Head
        # -----------------------------
        self.output = nn.Sequential(
            nn.Linear(d_token, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        # Embed categorical cols
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeds = torch.stack(cat_embeds, dim=1)

        # Project numeric
        num_embed = self.numeric_projection(x_num)
        num_embed = num_embed.unsqueeze(1)

        # Concatenate: numeric token + categorical tokens
        x = torch.cat([num_embed, cat_embeds], dim=1)

        # Transformer
        x = self.transformer(x)

        # Classification from first token
        logits = self.output(x[:, 0])

        return logits.squeeze(1)


# ============================================================
# 2. TRAINING FUNCTION
# ============================================================
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
        {
            precision, recall, f1, pr_auc, roc_auc,
            y_true, y_pred
        },
        model
    """

    # -------------------------------
    # 1) Device Setup
    # -------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for custom model: {device}")

    # -------------------------------
    # 2) Train/Val Split
    # -------------------------------
    N = len(y)
    train_size = int(0.8 * N)
    val_size = N - train_size
    print(f"Training size: {train_size}, Validation size: {val_size}")

    X_num_train, X_num_val = X_num[:train_size], X_num[train_size:]
    X_cat_train, X_cat_val = X_cat[:train_size], X_cat[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_ds = TensorDataset(X_num_train, X_cat_train, y_train)
    val_ds = TensorDataset(X_num_val, X_cat_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # -------------------------------
    # 3) Build Model
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
    # 4) Loss + Optimizer
    # -------------------------------
    pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance → pos_weight = {pos_weight_value:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------------
    # 5) Training Loop
    # -------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xn, xc, yy in train_loader:
            xn = xn.float().to(device)
            xc = xc.long().to(device)
            yy = yy.float().to(device)

            optimizer.zero_grad()
            logits = model(xn, xc)
            loss = criterion(logits, yy)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Custom Epoch {epoch}] loss={epoch_loss/len(train_loader):.4f}")

    # -------------------------------
    # 6) Validation
    # -------------------------------
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xn, xc, yy in val_loader:
            xn = xn.float().to(device)
            xc = xc.long().to(device)

            logits = model(xn, xc)

            y_true_list.append(yy)
            y_pred_list.append(logits.cpu())

    y_true = torch.cat(y_true_list).numpy()
    y_logits = torch.cat(y_pred_list).numpy()
    y_prob = 1 / (1 + torch.exp(torch.tensor(-y_logits))).numpy()
    y_pred_labels = (y_prob >= 0.5).astype(int)

    # -------------------------------
    # 7) Compute Metrics
    # -------------------------------
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "y_true": torch.tensor(y_true),
        "y_pred": torch.tensor(y_prob),
    }

    return metrics, model