import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGNN(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        num_categories_per_col,
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categorical = len(num_categories_per_col)

        # One embedding per categorical column
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embed_dim)
                for num_categories in num_categories_per_col
            ]
        )

        in_dim = num_numeric + self.num_categorical * embed_dim

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def gcn_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Very simple GCN-like aggregation:
        - Sum neighbor features
        - Divide by degree
        - ReLU
        """
        if edge_index.numel() == 0:
            # No edges: just return x passed through
            return x

        src, dst = edge_index  # each shape (E,)

        num_nodes, feat_dim = x.shape
        device = x.device

        # aggregate neighbor features into each destination node
        agg = torch.zeros_like(x, device=device)
        agg.index_add_(0, dst, x[src])

        # degree for each node
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)

        agg = agg / deg
        return F.relu(agg)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, edge_index: torch.Tensor):
        """
        x_num: (N, num_numeric)
        x_cat: (N, num_categorical)
        edge_index: (2, E)
        """
        # Embed categorical columns
        embeds = []
        if x_cat.numel() > 0:
            for i in range(x_cat.shape[1]):
                embeds.append(self.embeddings[i](x_cat[:, i]))
            cat_feats = torch.cat(embeds, dim=1)
            h = torch.cat([x_num, cat_feats], dim=1)
        else:
            h = x_num

        h = F.relu(self.fc_in(h))
        h = self.gcn_step(h, edge_index)
        logits = self.fc_out(h).squeeze(-1)  # (N,)

        return logits