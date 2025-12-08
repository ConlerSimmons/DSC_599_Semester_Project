import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transaction_graph(
    df,
    numeric_cols,
    k_neighbors: int = 5,
    min_group_size: int = 2,
    max_group_size: int = 1000,
):
    """
    Build a hybrid transaction graph.

    - Nodes: each transaction (one row in df after reset_index)
    - Edges:
        1) k-NN edges in numeric feature space
        2) Star-style edges for shared identity/device-type categorical fields

    I’m trying to keep this simple enough to reason about, but richer than
    pure k-NN so the GNN can actually exploit identity-like patterns.
    """

    # I reset the index so row positions match node IDs 0..N-1
    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    # ------------------------------------------------------------------
    # 1) k-NN edges on numeric features
    # ------------------------------------------------------------------
    X = df[numeric_cols].fillna(0.0).values.astype("float32")

    # I keep k reasonably small so the graph doesn't become too dense
    k = min(k_neighbors + 1, num_nodes)  # +1 to account for self

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                # skip self-loop; we could add them later if we wanted
                continue
            # edge i -> j
            src_list.append(i)
            dst_list.append(j)
            # and j -> i to keep it undirected
            src_list.append(j)
            dst_list.append(i)

    # ------------------------------------------------------------------
    # 2) Identity / device style edges (star pattern)
    # ------------------------------------------------------------------
    candidate_id_cols = [
        "card1",
        "addr1",
        "P_emaildomain",
        "id_30",
        "id_31",
        "DeviceInfo",
    ]

    # Only use columns that actually exist in df
    active_id_cols = [c for c in candidate_id_cols if c in df.columns]

    for col in active_id_cols:
        # I cast to string so everything is comparable (including NaNs)
        values = df[col].astype(str)

        # Group rows by the shared value in this column
        groups = values.groupby(values).groups  # dict: value -> Int64Index

        for val, idxs in groups.items():
            group_idx = list(idxs)
            group_size = len(group_idx)

            # I only connect groups that are "moderate" in size
            if group_size < min_group_size or group_size > max_group_size:
                continue

            # Star pattern: choose first node as hub
            hub = group_idx[0]
            for other in group_idx[1:]:
                # hub -> other
                src_list.append(hub)
                dst_list.append(other)
                # other -> hub
                src_list.append(other)
                dst_list.append(hub)

    # ------------------------------------------------------------------
    # 3) Final edge_index tensor
    # ------------------------------------------------------------------
    if not src_list:
        # Degenerate case: no edges at all
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return edge_index