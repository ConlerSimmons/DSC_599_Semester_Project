import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transaction_graph(
    df,
    identity_cols=None,
):
    """
    Pure Option A: exact-match identity edges (no kNN, no distances).

    Nodes: each transaction
    Edges: connect all transactions that share the same value
           for identity-like categorical columns:
             - card1
             - addr1
             - P_emaildomain
             - id_30
             - id_31

    Returned:
        edge_index : LongTensor of shape (2, E)
    """

    df = df.reset_index(drop=True)
    num_nodes = len(df)

    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    # Default identity columns
    if identity_cols is None:
        identity_cols = [
            "card1",
            "addr1",
            "P_emaildomain",
            "id_30",
            "id_31",
        ]

    identity_cols = [c for c in identity_cols if c in df.columns]

    src_list = []
    dst_list = []

    # For each identity column, connect nodes that share the same value
    for col in identity_cols:
        values = df[col].astype(str)
        groups = values.groupby(values).groups  # value -> row indices

        for val, idxs in groups.items():
            idxs = list(idxs)
            n = len(idxs)

            # No arbitrary min/max cutoffs — we preserve groups exactly
            if n < 2:
                continue

            # Fully connect the group (undirected)
            # but use a simple chain instead of full clique to keep edge count reasonable:
            #   i0 <-> i1 <-> i2 <-> ... <-> i(n-1)
            for i in range(n - 1):
                a = idxs[i]
                b = idxs[i + 1]

                src_list.append(a)
                dst_list.append(b)
                src_list.append(b)
                dst_list.append(a)

    # Build final tensor
    if not src_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return edge_index