import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    """
    A 2-layer residual GNN with:
      - Numeric projection
      - Categorical embeddings (scaled)
      - Input fusion MLP
      - TWO message-passing layers
      - Residual connections
      - LayerNorm for stability
      - Dropout
      - Output classifier
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

        # ----------------------------------------------------------------------
        # Numeric projection
        # ----------------------------------------------------------------------
        self.num_linear = nn.Linear(num_numeric, embed_dim)

        # ----------------------------------------------------------------------
        # Categorical embeddings
        # Add 1 to avoid OoB indexing
        # ----------------------------------------------------------------------
        self.cat_embeddings = nn.ModuleList()
        for size in self.num_categories_per_col:
            self.cat_embeddings.append(
                nn.Embedding(size + 1, embed_dim)
            )

        total_in_dim = embed_dim * (1 + len(self.cat_embeddings))
        self.input_linear = nn.Linear(total_in_dim, hidden_dim)

        # ----------------------------------------------------------------------
        # Two-layer residual GNN
        # ----------------------------------------------------------------------
        self.gcn_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Output classifier
        self.out_linear = nn.Linear(hidden_dim, 1)

    # ----------------------------------------------------------------------
    # Message Passing (mean aggregation)
    # ----------------------------------------------------------------------
    def gcn_step(self, x, edge_index):
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        num_nodes = x.size(0)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)

        return agg / deg

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, x_num, x_cat, edge_index):
        # ----- numeric -----
        num_embed = self.activation(self.num_linear(x_num))

        # ----- categorical -----
        cat_embeds = []
        if x_cat.numel() > 0:
            for i, emb in enumerate(self.cat_embeddings):
                col = x_cat[:, i].clamp(0, emb.num_embeddings - 1)
                cat_embeds.append(emb(col))

        if cat_embeds:
            cat_embed = torch.cat(cat_embeds, dim=1)
            h = torch.cat([num_embed, cat_embed], dim=1)
        else:
            h = num_embed

        # ----- fused projection -----
        h = self.activation(self.input_linear(h))

        # =======================
        # LAYER 1 (Residual GNN)
        # =======================
        h_res = h
        h_mp = self.gcn_step(h, edge_index)
        h = self.activation(self.gcn_linear1(h_mp))
        h = self.dropout(h)
        h = self.norm1(h + h_res)

        # =======================
        # LAYER 2 (Residual GNN)
        # =======================
        h_res = h
        h_mp = self.gcn_step(h, edge_index)
        h = self.activation(self.gcn_linear2(h_mp))
        h = self.dropout(h)
        h = self.norm2(h + h_res)

        # ----- output -----
        return self.out_linear(h).squeeze(-1)