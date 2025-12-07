import torch
import torch.nn as nn


class CustomTabTransformer(nn.Module):
    """
    A simple TabTransformer-style model for tabular data.

    - Categorical features are embedded and treated as a token sequence.
    - Numeric features are projected into the same embedding space and
      concatenated as an extra token.
    - A TransformerEncoder processes the sequence.
    - The final representation is pooled and fed into a small MLP head
      to produce a single fraud logit per row.
    """

    def __init__(
        self,
        num_numeric: int,
        num_categories: int,
        num_categorical: int,
        dim: int = 64,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.dim = dim

        # Shared embedding table for all categorical tokens
        self.category_emb = nn.Embedding(num_categories, dim)

        # Project all numeric features into the same embedding dimension
        self.numeric_proj = nn.Linear(num_numeric, dim)

        # Transformer encoder over sequence: [numeric_token] + [categorical_tokens]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,
            batch_first=True,  # so we can use [batch, seq, dim]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        # MLP head: takes pooled transformer output and outputs 1 logit
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: [batch_size, num_numeric]       (float)
        x_cat: [batch_size, num_categorical]   (long / int64)

        Returns:
            logits: [batch_size] (float) — unnormalized fraud scores
        """

        # -----------------------------
        # 1) Embed categorical features
        # -----------------------------
        # x_cat: [B, num_categorical] -> [B, num_categorical, dim]
        cat_emb = self.category_emb(x_cat)

        # -----------------------------
        # 2) Project numeric features
        # -----------------------------
        # x_num: [B, num_numeric] -> [B, 1, dim]
        num_emb = self.numeric_proj(x_num).unsqueeze(1)

        # -----------------------------
        # 3) Build sequence and run Transformer
        # -----------------------------
        # Sequence: [numeric_token] + [categorical_tokens]
        # Shape: [B, 1 + num_categorical, dim]
        seq = torch.cat([num_emb, cat_emb], dim=1)

        # Transformer expects [B, seq_len, dim] when batch_first=True
        seq_out = self.transformer(seq)

        # -----------------------------
        # 4) Pool sequence representation
        # -----------------------------
        # Simple mean pooling over sequence length
        pooled = seq_out.mean(dim=1)  # [B, dim]

        # -----------------------------
        # 5) MLP head -> logit
        # -----------------------------
        logits = self.mlp_head(pooled).squeeze(-1)  # [B]

        return logits