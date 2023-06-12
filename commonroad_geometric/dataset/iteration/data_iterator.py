from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.utils.filesystem import list_files
from commonroad_geometric.dataset.commonroad_data import CommonRoadData


# TODO delete?

class DataIterator(AutoReprMixin):
    """Class for iterating over single data instances found in folder.
    
    Example usage:
    .. code-block:: python

        data_iterator = DataIterator(
            directory='output/data'
        )
        for data in data_iterator:
            pred = model.forward(data)
            print(pred)
    """

    def __init__(
        self,
        directory: str,
        device: str = 'cpu',
        loop: bool = False,
        shuffle: bool = False
    ) -> None:
        """Init new DataIterator instance 

        Args:
            directory (str): Base directory to load data from.
            device (str, optional): PyTorch device. Defaults to 'cpu'.
            loop (bool, optional): Whether to recycle data. Defaults to False.
            shuffle (bool, optional): Whether to shuffle batches. Defaults to False.
        """

        self._loader_paths = list_files(
            directory,
            file_type='pth',
            join_paths=True
        )
        self._data_counter = 0
        self._cycle = -1
        self._shuffle = shuffle
        self._batch_counter = -1
        self._device = device
        self._loop = loop
        self._load_next_batch()

    @property
    def cycle(self) -> int:
        """Current iteration cycle"""
        return self._cycle

    def __iter__(self) -> DataIterator:
        return self

    def __next__(self) -> CommonRoadData:
        """Yields next data instance"""
        self._data_counter += 1
        try:
            data = next(self._iterator)
        except StopIteration:
            if not self._loop and self._batch_counter >= len(self._loader_paths) - 1:
                raise StopIteration()
            self._load_next_batch()
            data = next(self._iterator)
        return data

    def _load_next_batch(self) -> None:
        self._batch_counter = (self._batch_counter + 1) % len(self._loader_paths)
        self._cycle += 1
        path = self._loader_paths[self._batch_counter]
        original_loader = torch.load(path, map_location=self._device)
        loader = DataLoader(
            original_loader.dataset,
            batch_size=1,
            shuffle=self._shuffle
        )
        self._iterator = iter(loader)
