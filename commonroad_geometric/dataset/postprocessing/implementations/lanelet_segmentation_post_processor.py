from typing import List, Optional, Union

import torch
from torch_geometric.utils import add_remaining_self_loops, subgraph

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class LaneletSegmentationPostProcessor(BaseDataPostprocessor):
    """
    Segments data from road network into individual lanelet instances.
    """

    def __init__(
        self,
        add_self_loops: bool = True,
        ensure_multiple_nodes_per_sample: bool = True
    ) -> None:
        """
        Args:
            add_self_loops (bool): Adds self loops num nodes for a lanelet is one.
        """
        self._add_self_loops = add_self_loops
        self._ensure_multiple_nodes_per_sample = ensure_multiple_nodes_per_sample
        super().__init__()

    def __call__(
        self,
        samples: Union[CommonRoadData, List[CommonRoadData]],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:

        if isinstance(samples, CommonRoadData):
            samples = [samples]
        # require all data to be from the same scenario
        assert all(sample.scenario_id == samples[0].scenario_id for sample in samples[1:])

        for data in samples:
            assert data.vehicle, f"Sample {data.scenario_id} does not have vehicle nodes"
            assert data.lanelet, f"Sample {data.scenario_id} does not have lanelet nodes"
            assert data.vehicle_to_lanelet, f"Sample {data.scenario_id} does not have v2l edges"
            assert len(data.l.lanelet_id.numpy()) > 1, f"Sample {data.scenario_id} only has one lanelet!"

        segmented_samples = []

        for i_sample, sample in enumerate(samples):
            data = {i: [] for i in range(len(sample.l.id.numpy()))}
            for j in sample.v.indices:
                data[int(sample.v2l.edge_index[:, j][1])].append(
                    int((sample.v.indices == sample.v2l.edge_index[:, j][0]).nonzero(as_tuple=True)[0]))
            for k in data.keys():
                # sample.__dict__['_node_store_dict']['vehicle'].clone()
                segmented_sample = sample.clone()
                # Reorganize vehicle parameters
                segmented_sample.vertices.x = segmented_sample.vertices.x[data[k]]
                segmented_sample.vertices.is_ego_mask = segmented_sample.vertices.is_ego_mask[data[k]]
                segmented_sample.vertices.pos = segmented_sample.vertices.pos[data[k]]
                segmented_sample.vertices.id = segmented_sample.vertices.id[data[k]]
                # Remove lanelets without vehicles
                if segmented_sample.vertices.id.nelement() != 0:
                    segmented_sample.vertices.indices = segmented_sample.vertices.indices[data[k]]
                    segmented_sample.vertices.num_nodes = len(data[k])
                    # Reorganize vehicle to lanelet connections
                    segmented_sample.v2l.edge_index = segmented_sample.v2l.edge_index[:, data[k]]
                    segmented_sample.v2l.edge_attr = segmented_sample.v2l.edge_attr[data[k]]

                    # Reorganize lanelet connections
                    for key in segmented_sample.l:
                        segmented_sample.l[key] = segmented_sample.l[key][k].unsqueeze(0)
                    # Segmenting by lanelet so only one lanelet node
                    segmented_sample.l.num_nodes = 1

                    # Reorganize vehicle to vehicle connections
                    # Filter out edge nodes that have no v2v connections
                    edge_nodes = [x for x in data[k] if x in sample.v2v.edge_index.flatten().numpy()]
                    edge_index = sample.v2v.edge_index
                    edge_attr = sample.v2v.edge_attr
                    if len(edge_nodes) > 0 or self._add_self_loops:
                        if len(edge_nodes) == 0 and self._add_self_loops:
                            edge_index, edge_attr = add_remaining_self_loops(
                                edge_index, edge_attr, num_nodes=sample.v.id.shape[0])
                            edge_nodes = data[k]
                        segmented_sample.v2v.edge_index, segmented_sample.v2v.edge_attr = subgraph(
                            subset=edge_nodes, edge_index=edge_index, edge_attr=edge_attr, relabel_nodes=True)
                    else:
                        segmented_sample.v2v.edge_index = torch.empty(2, 0)
                        segmented_sample.v2v.edge_attr = torch.empty(2, 0)
                    # Reorganize lanelet to lanelet connections
                    l2l_indices = [j for j in range(sample.l2l.edge_index.shape[1]) if k in sample.l2l.edge_index[:, j]]
                    segmented_sample.l2l.edge_index = segmented_sample.l2l.edge_index[:, l2l_indices]
                    segmented_sample.l2l.edge_attr = segmented_sample.l2l.edge_attr[l2l_indices]

                    if self._ensure_multiple_nodes_per_sample:
                        if segmented_sample.vertices.num_nodes > 1:
                            segmented_samples.append(segmented_sample)
                    else:
                        segmented_samples.append(segmented_sample)

        return segmented_samples
