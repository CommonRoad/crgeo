from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from commonroad_geometric.common.config import Config
from commonroad_geometric.learning.geometric import BaseGeometric
from projects.geometric_models.drivable_area.models.modules.gcn import GCN


class OccupancyPredictionModel(BaseGeometric):

    def __init__(self, cfg: Config):
        super().__init__()
        self._cfg = cfg

    def _build(self, batch: Union[Batch, BaseData]) -> None:
        self.vehicle_feature_dim = sum(batch["vehicle"][attr].shape[1] for attr in ["x", "length", "lanelet_arclength"])
        self.vehicle_embedding_dim = 32
        self.vehicle_net = GCN(
            layers=3,
            input_dim=self.vehicle_feature_dim,
            hidden_dim=32,
            output_dim=self.vehicle_embedding_dim,
            p_dropout=0.2,
        )

        # TODO add Jumping Knowledge skip connections to stabilize training with more graph convolutional layers
        self.lanelet_feature_dim = sum(batch["lanelet"][attr].shape[1]
                                       for attr in ["length", "start_pos", "center_pos", "end_pos"])
        self.lanelet_feature_dim += self.vehicle_embedding_dim
        self.lanelet_net = GCN(
            layers=5,
            input_dim=self.lanelet_feature_dim,
            hidden_dim=128,
            output_dim=batch["lanelet"].occupancy.size(1),
            p_dropout=0.2,
        )

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )

    def _forward(self, batch: Union[Batch, BaseData], **kwargs) -> Tensor:
        vehicle_features = torch.cat((
            batch["vehicle"].x,
            batch["vehicle"].length,
            batch["vehicle"].lanelet_arclength,
            # batch["vehicle"].pos,
        ), dim=-1)
        assert vehicle_features.size(-1) == self.vehicle_feature_dim
        vehicle_emb = self.vehicle_net(x=vehicle_features, edge_index=batch["vehicle", "lanelet", "vehicle"].edge_index)

        # aggregate vehicle embeddings into one embedding per lanelet
        # vehicle_emb_graph = Data(batch=batch.batch, x=vehicle_emb, edge_index=batch["vehicle", "lanelet", "vehicle"].edge_index)
        # vehicle_emd_pooled = pyg_nn.avg_pool(cluster=batch["vehicle", "lanelet"].edge_index, data=vehicle_emb_graph)

        vehicle_emb_lanelet = torch.zeros(
            (batch["lanelet"].num_nodes,
             self.vehicle_embedding_dim),
            dtype=vehicle_features.dtype)
        vehicle_emb_lanelet_count = torch.zeros((batch["lanelet"].num_nodes,), dtype=torch.int)
        v2l_lanelet = batch["vehicle", "lanelet"].edge_index[1]
        num_vehicles = batch["vehicle"].num_nodes
        # the vehicle -> lanelet edge_index tensor is sorted by the vehicle node index
        # assert batch["vehicle", "lanelet"].edge_index[0, i] == i
        # v2l_lanelet[i] is the lanelet node index corresponding to vehicle node i
        # vehicle_emb_lanelet[v2l_lanelet] += vehicle_emb  # add up all vehicle embeddings for vehicles which share a lanelet
        # vehicle_emb_lanelet_count[v2l_lanelet] += 1
        for i in range(num_vehicles):
            vehicle_emb_lanelet[v2l_lanelet[i]] += vehicle_emb[i]
            vehicle_emb_lanelet_count[v2l_lanelet[i]] += 1
        vehicle_emb_lanelet /= vehicle_emb_lanelet_count.view(-1, 1)  # compute mean embedding

        lanelet_features = torch.cat((
            batch["lanelet"].length,
            batch["lanelet"].start_pos,
            batch["lanelet"].center_pos,
            batch["lanelet"].end_pos,
            vehicle_emb_lanelet,
        ), dim=-1)
        assert lanelet_features.size(-1) == self.lanelet_feature_dim
        lanelet_occupancies_unnorm: Tensor = self.lanelet_net(
            x=lanelet_features, edge_index=batch["lanelet", "lanelet"].edge_index)
        lanelet_occupancies = torch.sigmoid(lanelet_occupancies_unnorm)

        return lanelet_occupancies

    def _compute_loss(
            self,
            y: Tensor,
            batch: Union[Batch, BaseData],
            **kwargs
    ) -> Tuple[Tensor, Dict[str, Any], Dict[str, Tensor]]:
        loss = F.binary_cross_entropy(input=y, target=batch["lanelet"].occupancy)
        return loss, {}, {}
