from __future__ import annotations
import os

from typing import List, Dict, Tuple, TypeVar, Any


from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import  TrafficExtractorFactory
from commonroad_geometric.learning.geometric.training.experiment import  GeometricExperimentConfig
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal, CommonRoadDataTemporalBatch

from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.data.collate import collate

from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import separate
from typing import Any, List, Optional

from tutorials.collect_commonroad_temporal_dataset import TemporalGeometricExperiment
DATA_COLLECTOR_CLS = ScenarioDatasetCollector
SCENARIO_DIR = 'data/highd-test' if DATA_COLLECTOR_CLS is ScenarioDatasetCollector else 'data/highway_test'
DATASET_DIR = 'tutorials/output/dataset_t40'
T_Data = TypeVar("T_Data", Data, HeteroData, CommonRoadData, CommonRoadDataTemporal)
T_Batch = TypeVar("T_Batch",Batch,CommonRoadDataTemporalBatch)

class CommonRoadDataTemporalBatch(Batch):
    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        """
        modified from torch_geometric.data.Batch
        additionally maintaining slice_dict, inc_dict, num_graphs of CommonroadDataTemporal
        """

        temporal_data_num_graphs = [data._num_graphs for data in data_list]
        temporal_data_slice_dict = [data._slice_dict for data in data_list]
        temporal_data_inc_dict = [data._inc_dict for data in data_list]
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=True,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict
        batch._temporal_data_num_graphs = temporal_data_num_graphs
        batch._temporal_data_slice_dict = temporal_data_slice_dict
        batch._temporal_data_inc_dict = temporal_data_inc_dict

        return batch
    def get_example(self, idx: int) -> BaseData:

        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))
        
        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )
        data._slice_dict = self._temporal_data_slice_dict[idx]
        data._inc_dict = self._temporal_data_inc_dict[idx]
        data._num_graphs = self._temporal_data_num_graphs[idx]
        data.batch_size = self._temporal_data_num_graphs[idx]

        return data

def test_data_loader_collate_fn():
    # To test CommonRoadDataTemporalBatch, first collect a dataset composed of CommonRoadDataTemporal instances

    experiment_config = GeometricExperimentConfig(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50.0),
            )
        ),
        data_collector_cls=ScenarioDatasetCollector,
        preprocessors=[],
        postprocessors=[]
    )
    experiment = TemporalGeometricExperiment(experiment_config)

    #collect CommonRoadDataset, which contains collected Iterable[CommonRoadDataTemporal]
    dataset = experiment.get_dataset(
        scenario_dir=SCENARIO_DIR,
        dataset_dir=DATASET_DIR,
        overwrite=True,
        pre_transform_workers=4,
        max_scenarios=1,
        cache_data=True
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    i = 0
    for batch in loader:
        print(f"{i} / {len(dataset)}")
        assert isinstance(batch[0],CommonRoadDataTemporal)
        assert isinstance(batch[0][0],CommonRoadData)
        n = batch.batch_size
        assert sum( dataset[k].num_nodes for k in range(i, i + n) ) == batch.num_nodes
        for k in range(n):
            temporal_data = dataset[i + k]
            assert (batch[k].v.x == temporal_data.v.x).all()
            assert (batch[k].v2v.edge_attr == temporal_data.v2v.edge_attr).all()
        i += n
    return True
def main():
    print(test_data_loader_collate_fn())
if __name__ == '__main__':
    main()