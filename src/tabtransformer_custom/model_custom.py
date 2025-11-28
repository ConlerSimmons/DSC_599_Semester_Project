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
        d_token: int = 32,
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

        # 2) One numeric token
        self.num_linear = nn.Linear(num_numeric_features, d_token)

        # Total token count = 1 numeric token + N categorical tokens
        self.n_tokens = 1 + len(vocab_sizes)

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
        self.head = nn.Sequential(
            nn.Linear(self.n_tokens * d_token, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_cat):
        """
        x_num: (batch_size, num_numeric_features)
        x_cat: (batch_size, num_categorical_features)
        """
        x_num = x_num.float()
        x_cat = x_cat.long()
        # Numeric token
        num_token = self.num_linear(x_num).unsqueeze(1)

        # Categorical tokens
        cat_tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
            dim=1
        )

        # Sequence = [numeric_token] + [each categorical token]
        tokens = torch.cat([num_token, cat_tokens], dim=1)

        # Transformer encoder
        encoded = self.transformer(tokens)

        # Flatten
        flat = encoded.reshape(encoded.size(0), -1)

        # Classification head
        logits = self.head(flat).squeeze(-1)
        return logits