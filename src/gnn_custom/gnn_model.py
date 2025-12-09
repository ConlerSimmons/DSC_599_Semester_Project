import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    I’m writing a simple multi-head GAT layer from scratch.

    - It takes node features h of shape (N, in_dim)
    - It uses edge_index (2, E) to know which nodes are connected
    - For each head, it learns attention weights over neighbors and aggregates
      them into new node features of shape (N, out_dim * num_heads)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        # I project input features into num_heads * out_dim in one shot
        self.linear = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # For each head, I keep separate attention vectors for source and target
        self.attn_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))

        # I add a small dropout on attention weights for regularization
        self.attn_dropout = nn.Dropout(dropout)

        # I like to init things with xavier for stability
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h         : (N, in_dim)
        edge_index: (2, E) with [src, dst] node indices

        Returns:
        out       : (N, num_heads * out_dim)
        """
        if edge_index.numel() == 0:
            # If for some reason I had no edges, I just project and return
            N = h.size(0)
            projected = self.linear(h)
            return projected.view(N, self.num_heads * self.out_dim)

        src, dst = edge_index  # each is (E,)

        N = h.size(0)

        # Project features and reshape into (N, num_heads, out_dim)
        Wh = self.linear(h)  # (N, num_heads * out_dim)
        Wh = Wh.view(N, self.num_heads, self.out_dim)  # (N, H, D)

        # Gather source and destination node features per edge
        Wh_src = Wh[src]  # (E, H, D)
        Wh_dst = Wh[dst]  # (E, H, D)

        # I compute unnormalized attention scores:
        # e_ij^h = LeakyReLU( a_src^h · Wh_i^h + a_dst^h · Wh_j^h )
        # where "·" is a dot product along the last dimension.
        e_src = (Wh_src * self.attn_src).sum(dim=-1)  # (E, H)
        e_dst = (Wh_dst * self.attn_dst).sum(dim=-1)  # (E, H)
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)  # (E, H)

        # I exponentiate to get positive attention weights
        # alpha_ij^h ∝ exp(e_ij^h)
        alpha = torch.exp(e)  # (E, H)

        # Now I need to normalize per destination node and per head:
        # alpha_ij^h = exp(e_ij^h) / sum_{k in N(j)} exp(e_kj^h)
        # I do this by aggregating sums over dst with index_add.
        # First I sum alpha per destination node for each head.
        # denom has shape (N, H)
        denom = torch.zeros(N, self.num_heads, device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, alpha)

        # I avoid division by zero
        denom = denom.clamp_min(1e-8)

        # Normalize
        alpha = alpha / denom[dst]  # (E, H)

        # I can optionally dropout attention coefficients
        alpha = self.attn_dropout(alpha)

        # Now I aggregate neighbor messages:
        # m_j^h = sum_{i in N(j)} alpha_ij^h * Wh_i^h
        # Wh_src is (E, H, D), alpha is (E, H)
        alpha_expanded = alpha.unsqueeze(-1)  # (E, H, 1)
        messages = Wh_src * alpha_expanded    # (E, H, D)

        # Sum messages per destination node
        out = torch.zeros(N, self.num_heads, self.out_dim, device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, messages)  # (N, H, D)

        # Finally I reshape back to (N, H * D)
        out = out.view(N, self.num_heads * self.out_dim)  # (N, H*D)
        return out


class SimpleGNN(nn.Module):
    """
    I’m turning SimpleGNN into a 2-layer GAT-based model:

      1) I embed numeric + categorical features into a shared vector space
      2) I run two GAT layers with multi-head attention
      3) I apply LayerNorm, residual connections, and dropout
      4) I pass the final node representations through a small MLP to get a logit

    This keeps the interface the same as before, so train_gnn.py can stay as-is.
    """

    def __init__(
        self,
        num_numeric: int,
        num_categories_per_col,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_categories_per_col = list(num_categories_per_col)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # I project numeric features to embed_dim
        self.num_linear = nn.Linear(num_numeric, embed_dim)

        # I keep one embedding table per categorical column
        self.cat_embeddings = nn.ModuleList()
        for size in self.num_categories_per_col:
            # I add +1 to size to be safe in case of unseen index
            self.cat_embeddings.append(
                nn.Embedding(num_embeddings=size + 1, embedding_dim=embed_dim)
            )

        # After embeddings, I concatenate:
        #   numeric_embed (embed_dim) +
        #   each categorical embed (embed_dim)
        total_in_dim = embed_dim * (1 + len(self.cat_embeddings))

        # I first map this big concatenated vector into a hidden_dim space
        self.input_linear = nn.Linear(total_in_dim, hidden_dim)

        # GAT layers: I keep the output dim per head such that H * D = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_out_dim = hidden_dim // num_heads

        self.gat1 = GATLayer(
            in_dim=hidden_dim,
            out_dim=head_out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gat2 = GATLayer(
            in_dim=hidden_dim,
            out_dim=head_out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # I add LayerNorm + Dropout for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Final MLP head
        self.out_linear = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_num : (N, num_numeric)
        x_cat : (N, num_categorical)   # each col is an integer code
        edge_index : (2, E)
        """
        # ----- Numeric branch -----
        num_embed = self.num_linear(x_num)  # (N, embed_dim)
        num_embed = self.activation(num_embed)

        # ----- Categorical branch -----
        cat_embeds = []
        if x_cat.numel() > 0 and len(self.cat_embeddings) > 0:
            for col_idx, emb_layer in enumerate(self.cat_embeddings):
                # I take the column indices for this categorical feature
                col_indices = x_cat[:, col_idx]
                # I clamp to a valid range just to be safe
                col_indices = col_indices.clamp(
                    min=0, max=emb_layer.num_embeddings - 1
                )
                cat_embeds.append(emb_layer(col_indices))  # (N, embed_dim)

        if cat_embeds:
            # I concatenate all categorical embeddings along the feature dimension
            cat_embed = torch.cat(cat_embeds, dim=1)  # (N, embed_dim * num_cats)
            h = torch.cat([num_embed, cat_embed], dim=1)  # (N, total_in_dim)
        else:
            h = num_embed  # (N, embed_dim) in the degenerate case

        # ----- Initial projection into hidden_dim -----
        h = self.input_linear(h)  # (N, hidden_dim)
        h = self.activation(h)

        # ----- GAT layer 1 with residual + norm -----
        h1 = self.gat1(h, edge_index)  # (N, hidden_dim)
        h1 = self.dropout_layer(h1)
        h = self.norm1(h + h1)
        h = self.activation(h)

        # ----- GAT layer 2 with residual + norm -----
        h2 = self.gat2(h, edge_index)  # (N, hidden_dim)
        h2 = self.dropout_layer(h2)
        h = self.norm2(h + h2)
        h = self.activation(h)

        # ----- Output head -----
        out = self.out_linear(h).squeeze(-1)  # (N,)
        return out