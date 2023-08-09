from typing import Dict

import numpy as np

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin


class AggregatedRunningStats(AutoReprMixin):
    def __init__(self) -> None:
        self._max: float
        self._min: float
        self._mean: float
        self._absmean: float
        self._old_mean: float
        self._std: float
        self._s: float
        self._old_s: float
        self._n: int
        self.reset()

    def reset(self) -> None:
        self._max = -float('inf')
        self._min = float('inf')
        self._absmean = np.nan
        self._mean = np.nan
        self._old_mean = np.nan
        self._std = np.nan
        self._s = 0.0
        self._old_s = 0.0
        self._n = 0

    def update(self, value: float) -> None:
        import math

        abs_value = abs(value)

        self._n += 1
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        
        if self._n == 1:
            self._old_mean = self._mean = value
            self._old_s = 0.0
            self._absmean = abs_value
        else:
            self._mean = self._old_mean + (value - self._old_mean) / self._n
            self._absmean = self._absmean + (abs_value - self._absmean) / self._n
            self._s = self._old_s + (value - self._old_mean) * (value - self._mean)

            self._old_mean = self._mean
            self._old_s = self._s

        variance = self._s / (self._n - 1) if self._n > 1 else 0.0
        self._std = math.sqrt(variance)

    @property
    def max(self) -> float:
        return self._max

    @property
    def min(self) -> float:
        return self._min

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def absmean(self) -> float:
        return self._absmean

    @property
    def std(self) -> float:
        return self._std

    def asdict(self) -> Dict[str, float]:
        return dict(
            max=self.max,
            min=self.min,
            mean=self.mean,
            absmean=self.absmean,
            std=self.std
        )
