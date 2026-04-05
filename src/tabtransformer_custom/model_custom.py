import torch
import torch.nn as nn


class CustomTabTransformer(nn.Module):
    """
    A from-scratch TabTransformer-style model.

    - Embeds each categorical column into a learned vector.
    - Projects all numeric features into a single “numeric token”.
    - Concatenates numeric token + categorical tokens into a sequence.
    - Passes this sequence through a TransformerEncoder.
    - Flattens and feeds to a classification head → fraud probability.
    """

    def __init__(
        self,
        vocab_sizes,
        num_numeric_features,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_numeric_features = num_numeric_features
        self.vocab_sizes = vocab_sizes
        self.d_token = d_token

        # 1) Embedding layers for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat, d_token) for n_cat in vocab_sizes
        ])

        # 2) One token per numeric feature — stored as a single weight matrix
        #    for efficient batched projection (avoids a Python loop in forward)
        self.num_weight = nn.Parameter(torch.randn(num_numeric_features, d_token) * 0.02)
        self.num_bias   = nn.Parameter(torch.zeros(num_numeric_features, d_token))

        # Total token count = num_numeric tokens + N categorical tokens
        self.n_tokens = num_numeric_features + len(vocab_sizes)

        # 3) Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4) Classification head
        # LayerNorm stabilises the flattened transformer output before the MLP
        self.head = nn.Sequential(
            nn.LayerNorm(self.n_tokens * d_token),
            nn.Linear(self.n_tokens * d_token, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x_num, x_cat):
        """
        x_num: (batch_size, num_numeric_features)
        x_cat: (batch_size, num_categorical_features)
        """
        x_num = x_num.float()
        x_cat = x_cat.long()

        # One token per numeric feature — single batched matmul, no Python loop
        # x_num: (batch, n_num) -> unsqueeze -> (batch, n_num, 1)
        # num_weight: (n_num, d_token) -> broadcast over batch
        num_tokens = x_num.unsqueeze(2) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        # result: (batch, n_num, d_token)

        # Categorical tokens
        cat_tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
            dim=1
        )

        # Sequence = [numeric tokens] + [categorical tokens]
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)

        # Transformer encoder
        encoded = self.transformer(tokens)

        # Flatten
        flat = encoded.reshape(encoded.size(0), -1)

        # Classification head
        logits = self.head(flat).squeeze(-1)
        return logits