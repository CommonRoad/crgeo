from enum import Enum
from typing import TypeVar

from commonroad_geometric.dataset.commonroad_data import CommonRoadData


class Train_Features(Enum):
    Best = 'best'
    Avg = 'average'
    Current = 'current'


class Train_Categories(Enum):
    Train = 'train'
    Test = 'test'
    Validation = 'validation'


T_CommonRoadDataInput = TypeVar("T_CommonRoadDataInput", bound=CommonRoadData)
