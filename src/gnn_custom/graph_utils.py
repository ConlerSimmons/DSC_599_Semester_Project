from typing import Optional
import pandas as pd
import torch


def build_edge_index_from_key(
    df: pd.DataFrame,
    key_col: str = "card1",
    add_reverse: bool = True,
) -> torch.Tensor:
    """
    I build a simple transaction graph by connecting rows
    that share the same value in `key_col` (e.g., same card1).

    For each group with at least 2 rows, I chain them:
    (i -> i+1) and optionally (i+1 -> i) to make it undirected.

    Returns:
        edge_index: LongTensor of shape [2, E]
    """
    groups = df.groupby(key_col).indices
    edges = []

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs) - 1):
            a = idxs[i]
            b = idxs[i + 1]
            edges.append((a, b))
            if add_reverse:
                edges.append((b, a))

    if not edges:
        # Fallback: only self-loops so the model still runs
        n = len(df)
        idx = torch.arange(n, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def add_self_loops(edge_index: torch.Tensor, num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    I add self-loops to an existing edge_index.
    This makes sure every node keeps some of its own info each layer.
    """
    if device is None:
        device = edge_index.device

    self_nodes = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([self_nodes, self_nodes], dim=0)  # [2, N]
    return torch.cat([edge_index.to(device), self_loops], dim=1)