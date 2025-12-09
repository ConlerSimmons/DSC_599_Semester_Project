import torch

def build_transaction_graph(
    df,
    numeric_cols=None,
    min_group_size: int = 2,
    max_group_size: int = 2000,
):
    """
    Build a pure identity-based graph:
      - Nodes = transactions (rows)
      - Edges = connect rows sharing strong identity attributes:
            card1, addr1, P_emaildomain, id_30, id_31, DeviceInfo

    No kNN edges anymore (Option A).
    """

    df = df.reset_index(drop=True)
    num_nodes = len(df)
    if num_nodes == 0:
        raise ValueError("build_transaction_graph: dataframe is empty")

    candidate_id_cols = [
        "card1",
        "addr1",
        "P_emaildomain",
        "id_30",
        "id_31",
        "DeviceInfo",
    ]

    active_cols = [c for c in candidate_id_cols if c in df.columns]

    src = []
    dst = []

    for col in active_cols:
        values = df[col].astype(str)
        groups = values.groupby(values).groups

        for _, idxs in groups.items():
            idxs = list(idxs)
            size = len(idxs)

            if size < min_group_size or size > max_group_size:
                continue

            hub = idxs[0]
            for other in idxs[1:]:
                src.append(hub)
                dst.append(other)
                src.append(other)
                dst.append(hub)

    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return edge_index