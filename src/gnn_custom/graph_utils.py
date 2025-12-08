import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transaction_graph(df, numeric_cols, k_neighbors=5):
    """
    Build a k-NN graph using ONLY the rows present in df.
    This ensures edge_index always aligns with x_num/x_cat tensors.

    Nodes: transactions (0..N-1)
    Edges: k nearest neighbors in numeric space (undirected)
    """

    # Reset index so node IDs = row numbers
    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    # Numeric feature matrix
    X = df[numeric_cols].fillna(0.0).values.astype("float32")

    # If dataset is very small, shrink k
    k = min(k_neighbors + 1, num_nodes)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue  # skip self-loop

            # Add edge i → j
            src_list.append(i)
            dst_list.append(j)

            # Add reverse edge j → i
            src_list.append(j)
            dst_list.append(i)

    # No edges?
    if len(src_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # -------- SAFETY CHECK -------- #
    # Remove edges that refer to out-of-range nodes
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, mask]

    return edge_index