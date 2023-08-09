import torch
from commonroad_geometric.common.torch_utils.pygeo import transitive_edges


def test_transitive_edges() -> None:
    e1 = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ], dtype=torch.long)
    e2 = torch.tensor([
        [5, 3, 4, 2, 10],
        [6, 4, 5, 6, 11],
    ], dtype=torch.long)
    e3 = torch.tensor([
        [11, 5, 6, 4],
        [12, 6, 7, 5],
    ], dtype=torch.long)

    assert torch.all(e1 == transitive_edges(e1))

    edge_index = transitive_edges(e1, e2, e3)
    # print(edge_index)
    assert torch.all(edge_index == torch.tensor([
        [1, 2, 3, 4],
        [7, 5, 6, 7],
    ], dtype=torch.long))
