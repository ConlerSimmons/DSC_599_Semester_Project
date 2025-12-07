import torch
import torch.nn as nn


class CustomTabTransformer(nn.Module):
    """
    Minimal TabTransformer-style model:
    - Embeds categorical columns
    - Applies TransformerEncoder
    - Concatenates numeric features
    - MLP head for classification
    """

    def __init__(self, num_numeric, num_categories, embed_dim=32, num_heads=4, hidden_dim=64):
        super().__init__()

        self.num_categories = num_categories
        self.embed_dim = embed_dim

        # one embedding per categorical column
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embed_dim) for num_cat in num_categories
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # final MLP classification
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + num_numeric, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num, x_cat):
        # embed each categorical column
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(x_cat[:, i]))

        x_cat_embed = torch.stack(embedded, dim=1)   # (batch, num_cat, embed_dim)

        x_transformed = self.transformer(x_cat_embed)
        x_cat_final = x_transformed.mean(dim=1)      # pooled transformer output

        # concat numeric + transformer output
        x = torch.cat([x_num, x_cat_final], dim=1)

        return self.mlp(x).squeeze(1)