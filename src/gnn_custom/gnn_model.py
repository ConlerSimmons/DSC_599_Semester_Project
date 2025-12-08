import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    """
    A minimal GNN:
      - Linear projection for numeric features
      - Embeddings for each categorical column
      - Concatenate numeric + categorical embeddings
      - One GCN-like aggregation step over the transaction graph
      - Small MLP -> fraud logit
    """

    def __init__(
        self,
        num_numeric: int,
        num_categories_per_col,
        embed_dim: int = 8,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categories_per_col = list(num_categories_per_col)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Project numeric features -> embed_dim
        self.num_linear = nn.Linear(num_numeric, embed_dim)

        # One embedding per categorical column
        self.cat_embeddings = nn.ModuleList()
        for size in self.num_categories_per_col:
            # add 1 to size for safety in case of unseen index (should not happen, but safer)
            self.cat_embeddings.append(nn.Embedding(num_embeddings=size + 1, embedding_dim=embed_dim))

        # Input dimension after concatenation:
        #   numeric_embed (embed_dim) + each categorical embed (embed_dim)
        total_in_dim = embed_dim * (1 + len(self.cat_embeddings))

        self.input_linear = nn.Linear(total_in_dim, hidden_dim)
        self.gcn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()

    def gcn_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        One simple GCN-like step with mean aggregation:

          h_i = mean_{j in N(i)} x_j

        with safe degree handling to avoid divide-by-zero.
        """
        if edge_index.numel() == 0:
            # No edges: just return x unchanged
            return x

        src, dst = edge_index  # each shape (E,)
        num_nodes = x.size(0)
        feat_dim = x.size(1)

        # Aggregate neighbour messages
        agg = torch.zeros_like(x)  # (N, F)
        agg.index_add_(0, dst, x[src])  # sum messages from src -> dst

        # Degree of each node = number of incoming messages
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))

        # Avoid division by zero
        deg = deg.clamp_min(1.0).unsqueeze(1)  # (N, 1)

        # Mean aggregation
        h = agg / deg
        return h

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_num : (N, num_numeric)
        x_cat : (N, num_categorical)
        edge_index : (2, E)
        """
        # Numeric path
        num_embed = self.num_linear(x_num)  # (N, embed_dim)
        num_embed = self.activation(num_embed)

        # Categorical path
        cat_embeds = []
        if x_cat.numel() > 0 and len(self.cat_embeddings) > 0:
            # One column per embedding
            for col_idx, emb_layer in enumerate(self.cat_embeddings):
                # x_cat[:, col_idx] is (N,)
                col_indices = x_cat[:, col_idx]
                # Clamp in range [0, num_embeddings-1] to be safe
                col_indices = col_indices.clamp(min=0, max=emb_layer.num_embeddings - 1)
                cat_embeds.append(emb_layer(col_indices))  # (N, embed_dim)

        if cat_embeds:
            cat_embed = torch.cat(cat_embeds, dim=1)  # (N, embed_dim * num_categorical)
            h = torch.cat([num_embed, cat_embed], dim=1)
        else:
            h = num_embed

        # Initial MLP
        h = self.input_linear(h)
        h = self.activation(h)

        # Graph step
        h = self.gcn_step(h, edge_index)
        h = self.gcn_linear(h)
        h = self.activation(h)

        # Output logit
        out = self.out_linear(h).squeeze(-1)  # (N,)
        return out