import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transaction_graph(df, numeric_cols, k_neighbors: int = 5) -> torch.Tensor:
    """
    Build a simple k-NN graph over transactions using numeric features.

    Nodes: each transaction (row in df)
    Edges: undirected edges between k nearest neighbours in numeric space

    Returns
    -------
    edge_index : LongTensor of shape (2, E)
        edge_index[0] = source node indices
        edge_index[1] = destination node indices
    """
    # Make sure index is 0..N-1 so row positions match node ids
    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    # Use numeric features (already scaled in train_gnn) to define similarity
    X = df[numeric_cols].values.astype("float32")

    # Request k+1 neighbors (the closest neighbor is the point itself)
    k = min(k_neighbors + 1, num_nodes)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                # skip self-loop here; model can still handle self-information
                continue

            # i -> j
            src_list.append(i)
            dst_list.append(j)

            # j -> i to make it undirected
            src_list.append(j)
            dst_list.append(i)

    if not src_list:
        # fallback: no neighbors found (degenerate tiny case)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return edge_index