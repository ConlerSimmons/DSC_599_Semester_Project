import torch
import pandas as pd
from typing import List

# Columns we will use to create edges between transactions.
# We only use the ones that actually exist in df.columns.
EDGE_COLUMNS: List[str] = [
    "card1",
    "addr1",
    "P_emaildomain",
    "id_30",
    "DeviceInfo",
    "id_31",
]


def build_transaction_graph(df, numeric_cols=None, k_neighbors: int = 5):
    """
    Build a graph over transactions using *categorical identity/device features*.

    Nodes:
        - Each transaction (row in df), with index 0..N-1

    Edges:
        - For each column in EDGE_COLUMNS (that exists in df):
            - Group rows by that column's value
            - For each group with size >= 2, connect the rows in a "star":
                  center node <-> each other node
              This gives us an undirected edge pattern, but avoids
              O(group_size^2) fully-connected cliques.

    Parameters
    ----------
    df : pandas.DataFrame
        The (already merged and filtered) transaction dataframe.
        Its index will be reset inside this function so that row positions
        match node IDs 0..N-1.
    numeric_cols : list or None
        Kept for API compatibility. Not used in this graph builder.
    k_neighbors : int
        Kept for API compatibility. Ignored in this implementation.

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, E)
        edge_index[0] = source node indices
        edge_index[1] = destination node indices
    """
    # Ensure index is 0..N-1 so row positions are stable node IDs
    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    src_list = []
    dst_list = []

    # We only use edge columns that are actually present in the dataframe
    used_columns = [col for col in EDGE_COLUMNS if col in df.columns]

    if not used_columns:
        # No usable columns -> return an empty graph
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index

    for col in used_columns:
        col_series = df[col]

        # Build groups of indices by category value, skipping NaNs
        # (NaN would otherwise make one giant, meaningless group).
        # We use a dict: value -> list of row indices.
        groups = {}
        for idx, val in col_series.items():
            if pd.isna(val):
                continue
            groups.setdefault(val, []).append(idx)

        # For each group, build a star: center <-> each other node
        for _, indices in groups.items():
            if len(indices) < 2:
                continue  # no edges from singleton groups

            center = indices[0]
            for other in indices[1:]:
                # center -> other
                src_list.append(center)
                dst_list.append(other)
                # other -> center (make graph undirected
                src_list.append(other)
                dst_list.append(center)

    if not src_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return edge_index