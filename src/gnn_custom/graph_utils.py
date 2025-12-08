from typing import Optional
import pandas as pd
import torch


def build_edge_index_from_key(
    df: pd.DataFrame,
    key_col: str = "card1",
    add_reverse: bool = True,
    max_nodes: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a simple transaction graph by connecting rows that share
    the same value in `key_col` (e.g., same card1).

    For each group with at least 2 rows, I chain them:
        (i -> i+1) and optionally (i+1 -> i) to make it undirected.

    Args:
        df: Pandas DataFrame with at least the key_col present.
        key_col: Column name used to group transactions into nodes that
                 should be connected (e.g., "card1").
        add_reverse: If True, for every edge (a, b) also add (b, a).
        max_nodes: If provided, I only allow node indices in
                   range [0, max_nodes). This is important when we
                   only use a prefix of df as nodes (e.g., train split)
                   but df has more rows.

    Returns:
        edge_index: LongTensor of shape [2, E] with valid node indices.
    """
    # Ensure a clean, 0-based index
    df = df.reset_index(drop=True)

    # Effective upper bound for valid node indices
    if max_nodes is None:
        valid_len = len(df)
    else:
        valid_len = min(max_nodes, len(df))

    # Group rows by the key_col; .indices maps key -> array of row indices
    groups = df.groupby(key_col).indices
    edges = []

    for idxs in groups.values():
        # Ignore groups with fewer than 2 nodes
        if len(idxs) < 2:
            continue

        # Convert to list and enforce bounds [0, valid_len)
        idxs = [int(i) for i in idxs if 0 <= int(i) < valid_len]
        if len(idxs) < 2:
            continue

        # Chain consecutive indices in this group
        for i in range(len(idxs) - 1):
            a = idxs[i]
            b = idxs[i + 1]

            # Double-check bounds to be safe
            if 0 <= a < valid_len and 0 <= b < valid_len:
                edges.append((a, b))
                if add_reverse:
                    edges.append((b, a))

    if not edges:
        # Fallback: only self-loops so the model still runs,
        # even if no edges were found.
        n = valid_len
        idx = torch.arange(n, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def add_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Add self-loops to an existing edge_index so that every node
    preserves some of its own signal each GNN layer.

    Args:
        edge_index: LongTensor of shape [2, E].
        num_nodes: Total number of nodes.
        device: Optional device override; if None, I use edge_index.device.

    Returns:
        edge_index_with_loops: LongTensor of shape [2, E + num_nodes].
    """
    if device is None:
        device = edge_index.device

    self_nodes = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([self_nodes, self_nodes], dim=0)  # [2, N]

    return torch.cat([edge_index.to(device), self_loops], dim=1)