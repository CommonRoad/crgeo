import logging
from typing import Dict, List, Set, Union

from commonroad.geometry.shape import Shape
from commonroad.prediction.prediction import Occupancy, Prediction, TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory

logger = logging.getLogger(__name__)


def trajectory_prediction_init_fast(self: TrajectoryPrediction, trajectory: Trajectory, shape: Shape,
                 center_lanelet_assignment: Union[None, Dict[int, Set[int]]] = None,
                 shape_lanelet_assignment: Union[None, Dict[int, Set[int]]] = None):
    """
    :param trajectory: predicted trajectory of the obstacle
    :param shape: shape of the obstacle
    """
    self.shape = shape
    self.trajectory = trajectory
    self.shape_lanelet_assignment = shape_lanelet_assignment
    self.center_lanelet_assignment = center_lanelet_assignment
    Prediction.__init__(self, self._trajectory.initial_time_step, self._trajectory.state_list)
    self.final_time_step = max([state.time_step for state in self._trajectory.state_list])


def prediction_init_fast (self: Prediction, initial_time_step: int, occupancy_set: List[Occupancy]):
    self.initial_time_step = initial_time_step


TrajectoryPrediction.__init__ = trajectory_prediction_init_fast
Prediction.__init__ = prediction_init_fast

logger.info("Activated fast scenario loading mode (skipping occupancy computations)")