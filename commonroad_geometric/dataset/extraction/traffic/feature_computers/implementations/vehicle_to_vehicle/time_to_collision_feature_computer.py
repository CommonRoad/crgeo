import math

import numpy as np

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.common.geometry.helpers import relative_orientation, rotate_2d_matrix
from commonroad_geometric.dataset.extraction.traffic.feature_computers import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, V2VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class TimeToCollisionFeatureComputer(BaseFeatureComputer[V2VFeatureParams]):
    """
    Feature computer for time-to-collision between vehicles.

    Derivation:
        - https://www.geogebra.org/m/fqsdaptc
        - https://www.wolframalpha.com/input?i=minimize+%28x_1+%2B+v_1+t%29%C2%B2+%2B+%28v_0+t-y_1+-+v_2++t%29%C2%B2+wrt+t

    """

    # TODO: Generalize to curvilinear coordinate frame?
    # TODO: Should probably be unit-tested
    # TODO: Numba @jit?

    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __call__(
        self,
        params: V2VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        source_pos = params.source_state.position
        source_orientation = params.source_state.orientation
        source_velocity = params.source_state.velocity
        target_pos = params.target_state.position
        target_orientation = params.target_state.orientation
        target_velocity = params.target_state.velocity

        # Computing source vehicle pose in target vehicle's body coordinate frame
        pos_rel = source_pos - target_pos
        rotation = rotate_2d_matrix(-target_orientation)
        pos_target_frame = pos_rel @ rotation.T
        orientation_target_frame = relative_orientation(source_orientation, target_orientation)

        # Declaring variables
        x_1: float = -pos_target_frame[1]
        y_1: float = pos_target_frame[0]
        v_0: float = target_velocity
        v_x: float = np.sin(orientation_target_frame) * source_velocity
        v_y: float = np.cos(orientation_target_frame) * source_velocity
        v_0_sq: float = v_0 ** 2
        v_x_sq: float = v_x ** 2
        v_y_sq: float = v_y ** 2

        # Computing critical time value and distance based on d/dt(distance²(p_source(t), p_target(t))) =: 0
        t_crid_den = (v_0_sq - 2 * v_0 * v_y + v_x_sq + v_y_sq)
        t_crit = ((v_0 - v_y) * y_1 - v_x * x_1) / t_crid_den if t_crid_den != 0.0 else 100
        dist_sq_crit_den = (v_x_sq + (v_0 - v_y) ** 2)
        dist_sq_crit = ((v_0 - v_y) * x_1 + v_x * y_1) ** 2 / dist_sq_crit_den if dist_sq_crit_den != 0.0 else 100
        dist_crit = np.sqrt(dist_sq_crit) if dist_sq_crit_den != 0.0 else 100

        # Deciding whether collision will occur
        collision_threshold: float = 2 * params.source_obstacle.obstacle_shape.width  # type: ignore
        expects_collision = math.isfinite(dist_crit) and bool(dist_crit <= collision_threshold)
        time_to_collision = t_crit if expects_collision else 100
        if time_to_collision <= 0:
            time_to_collision = 100
            expects_collision = False

        features = {
            V2V_Feature.TimeToClosest.value: t_crit,
            V2V_Feature.ClosestDistance.value: dist_crit,
            V2V_Feature.ExpectsCollision.value: expects_collision,
            V2V_Feature.TimeToCollision.value: time_to_collision,
            V2V_Feature.TimeToClosestCLT.value: self._closeness_transform(max(0.0, t_crit), 10.0),
            V2V_Feature.ClosestDistanceCLT.value: self._closeness_transform(dist_crit, 50.0),
            V2V_Feature.TimeToCollisionCLT.value: self._closeness_transform(time_to_collision, 10.0),
        }

        # print(f"x_1: {x_1:.2f}, y_1: {y_1:.2f}, v_0: {v_0:.2f}, v_x: {v_x:.2f}, v_y: {v_y:.2f}, ttc: {time_to_collision}")

        return features

    @staticmethod
    def _closeness_transform(x: float, threshold: float) -> float:
        if not math.isfinite(x):
            return 0.0
        return 1 - min(x, threshold) / threshold
