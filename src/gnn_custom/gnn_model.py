import torch
import torch.nn as nn
from typing import List


class CustomGNN(nn.Module):
    """
    Simple GNN for fraud detection.

    - Uses embeddings for each categorical column
    - Projects numeric features to the same embedding dimension
    - Concatenates numeric + categorical embeddings for each node
    - Applies a few GCN-style message-passing steps implemented in plain PyTorch
    - Outputs a fraud logit per transaction (node)
    """

    def __init__(
        self,
        num_numeric: int,
        num_categories: List[int],
        emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categories = num_categories
        self.emb_dim = emb_dim

        # One embedding table per categorical column
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality in num_categories]
        )

        # Project numeric features into the same embedding space
        self.numeric_proj = nn.Linear(num_numeric, emb_dim)

        # GNN hidden stack
        in_dim = emb_dim * (len(num_categories) + 1)
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        prev_dim = in_dim
        for _ in range(num_layers):
            layer = nn.Linear(prev_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
            self.activations.append(nn.ReLU())
            prev_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev_dim, 1)

    def gcn_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Very simple GCN-style message passing:
        - For each destination node, sum neighbor features
        - Divide by degree to get a mean
        """
        # x: [N, d], edge_index: [2, E]
        src, dst = edge_index
        N = x.size(0)

        # Aggregate neighbor messages
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        # Degree-normalize
        deg = torch.bincount(dst, minlength=N).unsqueeze(1).clamp(min=1)
        agg = agg / deg
        return agg

    def forward(
        self,
        x_num: torch.Tensor,      # [N, num_numeric]
        x_cat: torch.Tensor,      # [N, num_categorical]
        edge_index: torch.Tensor  # [2, E]
    ) -> torch.Tensor:
        # Build initial node features
        num_emb = self.numeric_proj(x_num)  # [N, emb_dim]

        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(x_cat[:, i]))  # each [N, emb_dim]

        x = torch.cat([num_emb] + cat_embs, dim=1)  # [N, (num_cats+1)*emb_dim]

        # Add self-loops so every node keeps some of its own info
        N = x.size(0)
        device = x.device
        self_loops = torch.arange(N, device=device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)  # [2, N]
        edge_index = torch.cat([edge_index.to(device), self_loops], dim=1)

        h = x
        for layer, act in zip(self.layers, self.activations):
            h = self.gcn_step(h, edge_index)
            h = layer(h)
            h = act(h)
            h = self.dropout(h)

        logits = self.out(h).squeeze(1)  # [N]
        return logits