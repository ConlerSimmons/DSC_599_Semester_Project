import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    """
    A 2-layer GNN with:
      - Linear projection for numeric features
      - Embeddings for categorical columns
      - Concatenation of numeric + categorical embeddings
      - Two GCN-style message passing layers
      - Residual connections
      - Layer normalization
      - Dropout
      - Output MLP → fraud logit
    """

    def __init__(
        self,
        num_numeric: int,
        num_categories_per_col,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categories_per_col = list(num_categories_per_col)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Project numeric features into embedding space
        self.num_linear = nn.Linear(num_numeric, embed_dim)

        # Embeddings for categorical columns
        self.cat_embeddings = nn.ModuleList()
        for size in self.num_categories_per_col:
            self.cat_embeddings.append(
                nn.Embedding(num_embeddings=size + 1, embedding_dim=embed_dim)
            )

        # Total input dimension = numeric embed + each categorical embed
        total_in_dim = embed_dim * (1 + len(self.cat_embeddings))

        # Input projection before GNN layers
        self.input_linear = nn.Linear(total_in_dim, hidden_dim)

        # ---------- TWO GNN LAYERS ----------
        self.gcn_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_linear2 = nn.Linear(hidden_dim, hidden_dim)

        # ---------- Normalization + dropout ----------
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # Output classifier
        self.out_linear = nn.Linear(hidden_dim, 1)

    # -----------------------------
    # GCN message passing step
    # -----------------------------
    def gcn_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        One mean-aggregation GNN step:

            h_i = mean( x_j for j in N(i) )

        Residual connections are handled outside this function.
        """
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbor messages
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        # Degree normalization
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)

        return agg / deg

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, edge_index: torch.Tensor):
        """
        x_num: (N, num_numeric)
        x_cat: (N, num_categorical)
        edge_index: (2, E)
        """

        # ----- Numeric path -----
        num_embed = self.activation(self.num_linear(x_num))

        # ----- Categorical embeddings -----
        cat_embeds = []
        if x_cat.numel() > 0:
            for col_idx, emb_layer in enumerate(self.cat_embeddings):
                col_ids = x_cat[:, col_idx].clamp(0, emb_layer.num_embeddings - 1)
                cat_embeds.append(emb_layer(col_ids))

        if cat_embeds:
            cat_embed = torch.cat(cat_embeds, dim=1)
            h = torch.cat([num_embed, cat_embed], dim=1)
        else:
            h = num_embed

        # ----- Input projection -----
        h = self.activation(self.input_linear(h))

        # ================================================
        #      GNN LAYER 1  (residual + norm)
        # ================================================
        h_res = h  # residual
        h_mp = self.gcn_step(h, edge_index)  # message passing
        h = self.gcn_linear1(h_mp)           # linear transform
        h = self.activation(h)
        h = self.dropout(h)
        h = h + h_res                        # residual add
        h = self.norm1(h)                    # layer norm

        # ================================================
        #      GNN LAYER 2  (residual + norm)
        # ================================================
        h_res = h
        h_mp = self.gcn_step(h, edge_index)
        h = self.gcn_linear2(h_mp)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + h_res
        h = self.norm2(h)

        # ----- Output -----
        out = self.out_linear(h).squeeze(-1)
        return out