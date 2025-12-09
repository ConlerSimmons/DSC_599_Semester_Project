import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transaction_graph(
    df,
    numeric_cols,
    k_neighbors: int = 5,
    min_group_size: int = 2,
    max_group_size: int = 1000,
    small_group_full_connect: int = 10,
):
    """
    Build a hybrid transaction graph.

    Nodes
    -----
    - One node per transaction (row in df after reset_index).

    Edges
    -----
    1 k-NN edges in numeric feature space
       - Uses scaled numeric features passed in via `numeric_cols`.
       - Undirected: for each i->j we also add j->i.

    2 Identity/device-style edges for:
           card1, addr1, P_emaildomain, id_30, id_31, DeviceInfo
       - Groups rows that share the same value in one of these columns.
       - For each group with size in [min_group_size, max_group_size]:
           * If group_size <= small_group_full_connect:
                 - Fully connect the group (undirected clique).
           * Else:
                 - "Cluster" pattern:
                     - Pick a hub node (first index).
                     - Connect hub <-> everyone.
                     - Also connect neighbors along a chain:
                         idx[0] <-> idx[1], idx[1] <-> idx[2], ...
       - This is richer than a simple star, but we still avoid
         fully dense cliques for very large groups.

    Returns
    -------
    edge_index : LongTensor of shape (2, E)
        edge_index[0] = source node indices
        edge_index[1] = destination node indices
    """
    # Make sure row positions match node ids 0..N-1
    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    # ------------------------------------------------------------------
    # 1 k-NN edges on numeric features
    # ------------------------------------------------------------------
    X = df[numeric_cols].fillna(0.0).values.astype("float32")

    # Keep k reasonably small so the graph doesn't blow up
    k = min(k_neighbors + 1, num_nodes)  # +1 to include the point itself

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                # Skip self-loop here; we'll handle self-loops separately if needed
                continue
            # i -> j
            src_list.append(i)
            dst_list.append(j)
            # j -> i (undirected)
            src_list.append(j)
            dst_list.append(i)

    # ------------------------------------------------------------------
    # 2 Identity / device / browser-style edges (cluster pattern)
    # ------------------------------------------------------------------
    candidate_id_cols = [
        "card1",
        "addr1",
        "P_emaildomain",
        "id_30",
        "id_31",
        "DeviceInfo",
    ]

    # Only use columns that actually exist
    active_id_cols = [c for c in candidate_id_cols if c in df.columns]

    for col in active_id_cols:
        # Cast to string so everything is comparable (including NaNs)
        values = df[col].astype(str)

        # Group rows by shared value in this column
        # groups: dict[value] -> Int64Index of row positions
        groups = values.groupby(values).groups

        for val, idxs in groups.items():
            group_idx = list(idxs)
            group_size = len(group_idx)

            # Skip tiny or gigantic groups
            if group_size < min_group_size or group_size > max_group_size:
                continue

            group_idx_sorted = sorted(group_idx)

            if group_size <= small_group_full_connect:
                # Small group → full clique (everyone connected to everyone)
                for i_pos in range(group_size):
                    for j_pos in range(i_pos + 1, group_size):
                        u = group_idx_sorted[i_pos]
                        v = group_idx_sorted[j_pos]
                        # u <-> v
                        src_list.append(u)
                        dst_list.append(v)
                        src_list.append(v)
                        dst_list.append(u)
            else:
                # Medium-sized group → hub + chain pattern
                hub = group_idx_sorted[0]

                # Hub connections
                for other in group_idx_sorted[1:]:
                    src_list.append(hub)
                    dst_list.append(other)
                    src_list.append(other)
                    dst_list.append(hub)

                # Local chain connections: idx[0] <-> idx[1], idx[1] <-> idx[2], ...
                for i_pos in range(group_size - 1):
                    u = group_idx_sorted[i_pos]
                    v = group_idx_sorted[i_pos + 1]
                    src_list.append(u)
                    dst_list.append(v)
                    src_list.append(v)
                    dst_list.append(u)

    # ------------------------------------------------------------------
    # 3 Build final edge_index tensor + deduplicate + optional self-loops
    # ------------------------------------------------------------------
    if not src_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Deduplicate edges (they can appear multiple times from different mechanisms)
    # edge_index: (2, E) -> (E, 2)
    edges_2d = edge_index.t()
    # Unique rows
    unique_edges = torch.unique(edges_2d, dim=0)
    edge_index = unique_edges.t().contiguous()  # back to (2, E_unique)

    # Optional: add self-loops for stability
    # (this can help GNN layers have a "self" contribution in the aggregation)
    num_nodes_tensor = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([num_nodes_tensor, num_nodes_tensor], dim=0)  # (2, N)

    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return edge_index