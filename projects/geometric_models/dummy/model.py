from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GCNConv

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.implementations.render_traffic_graph_plugin import RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin


class DummyModel(BaseGeometric):
    """Example implementation of a Graph Neural Network extending the BaseGeometric class.
    The dummy training objective is to simply push the
    aggregate node embeddings towards zero.
    """

    def compute_loss(
        self,
        out: Tensor,
        batch: CommonRoadData,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any], Dict[str, Tensor]]:
        """Learning to simply minimize the mean squared output"""
        loss = (out**2).mean()

        info = dict(
            out_max=out.max().item(),
            out_min=out.min().item(),
            out_std=out.std().item(),
        )

        return loss, info

    def forward(
        self,
        batch: CommonRoadData,
        **kwargs
    ) -> Tensor:
        # Extracting graph attributes
        #  Reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        x = batch.vehicle.x
        edge_index = batch.vehicle_to_vehicle.edge_index
        if 'batch' in batch.v:
            batch_indices = batch.v.batch
        else:
            batch_indices = torch.zeros(x.shape[0], dtype=int, device=x.device)

        # Some potentially useful utilities
        # Insert breakpoint below and use the debugging console to inspect them :-)
        # try:
        #     batch_masks = get_batch_masks(batch_indices, max_num_nodes=20)
        #     batch_internal_indices = get_batch_internal_indices(batch_indices)
        #     dense_x = to_dense_batch(x, batch_indices, fill_value=1e10, max_num_nodes=batch_indices.shape[0])[0]
        #     dense_adj_matrix = to_dense_adj(
        #         edge_index,
        #         batch=batch_indices,
        #         max_num_nodes=20
        #     )
        # except Exception as e:
        #     # TODO: fix error when batch includes empty graphs
        #     print(f"ERROR: TODO: fix error when batch includes empty graphs\n{e}")
        #     raise

        # Scaling up vehicle node features with a linear transformation
        z = self.upscaler(x)

        # Applying non-linear activation function
        z = torch.relu(z)

        # Feeding upscaled features through GCNConv layer
        #  Reference: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        phi = self.conv.forward(x=z, edge_index=edge_index)

        # Graph-level pooling
        # Reference:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=global_max_pool#global-pooling-layers
        y = global_add_pool(phi, batch_indices)

        return y

    def configure_optimizer(
        self,
        trial: Any = None,
        optimizer_service = None
    ) -> Optimizer:
        # Just creating an optimizer for our model
        # For reference, see https://pytorch.org/docs/stable/optim.html
        if optimizer_service is not None:
            return optimizer_service.suggest_optimizer(self.parameters(), trial)
        return torch.optim.Adam(
            self.parameters()
        )

    def _build(self, batch: CommonRoadData, trial = None) -> None:
        # Building the network architecture,
        # with the help of a dummy batch for extracting appropriate dimensions.
        x = batch.vehicle.x

        # Demonstrating parameter fetching via optimize service
        in_features = x.shape[1]
        out_features = 20
        n_layers = 0
        if self._optimizer_service is not None:
            out_features = self._optimizer_service.suggest_param(
                "suggest_int", "out_features", trial, **{'low': 1, 'high': 10})
            n_layers = self._optimizer_service.suggest_param("suggest_int", "n_layers", trial, **{'low': 1, 'high': 3})

        layers = []
        for i in range(n_layers):
            hidden_channels = self._optimizer_service.suggest_param(
                "suggest_int", "hidden_channels_{}".format(i), trial, **{'low': 10, 'high': 20})
            layers.append(nn.Linear(in_features, hidden_channels))
            layers.append(nn.ReLU())
            p = 0.3
            if self._optimizer_service is not None:
                p = self._optimizer_service.suggest_param(
                    "suggest_float", "dropout_{}".format(i), trial, **{'low': 0.2, 'high': 0.5})
            layers.append(nn.Dropout(p))
            in_features = hidden_channels

        layers.append(nn.Linear(in_features, out_features))
        self.upscaler = nn.Sequential(*layers)

        self.conv = GCNConv(
            in_channels=out_features,
            out_channels=10
        )

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRenderPlugin]]:
        return [
            RenderLaneletNetworkPlugin(),
            RenderTrafficGraphPlugin(),
            RenderObstaclePlugin(),
        ]
