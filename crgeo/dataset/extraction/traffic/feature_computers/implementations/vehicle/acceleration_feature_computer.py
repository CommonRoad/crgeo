from collections import defaultdict
from typing import Dict

from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from crgeo.simulation.base_simulation import BaseSimulation


class AccelerationFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    def __init__(
        self,
        from_velocity: bool = False
    ) -> None:
        self.from_velocity = from_velocity
        self._last_velocity_cache: Dict[int, float] = defaultdict(float)
        super().__init__()

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        
        if not self.from_velocity and hasattr(params.state, 'acceleration'):
            return {V_Feature.Acceleration.value: float(params.state.acceleration)}

        obstacle_id = params.obstacle.obstacle_id
        velocity = params.state.velocity
        if obstacle_id in self._last_velocity_cache:
            last_velocity = self._last_velocity_cache.get(obstacle_id)
            velocity_difference = velocity - last_velocity
            acceleration = velocity_difference / params.dt
        else:
            acceleration = 0.0

        self._last_velocity_cache[obstacle_id] = velocity

        features = {
            V_Feature.Acceleration.value: acceleration
        }

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        self._last_velocity_cache = defaultdict(float)
