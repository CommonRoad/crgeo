from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

_T_ArrayBufferMetric = TypeVar("_T_ArrayBufferMetric", float, np.ndarray)

@dataclass
class _ArrayBufferMetrics(Generic[_T_ArrayBufferMetric]):
    max: _T_ArrayBufferMetric
    min: _T_ArrayBufferMetric
    mean: _T_ArrayBufferMetric
    absmean: _T_ArrayBufferMetric
    std: _T_ArrayBufferMetric