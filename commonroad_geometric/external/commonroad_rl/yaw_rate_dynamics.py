from typing import List, Tuple

import numpy as np
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.scenario.state import State
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from scipy.optimize import Bounds
from vehiclemodels.vehicle_parameters import VehicleParameters


# TODO: merge the change of feature_cbf here (limiting yaw rate with friction circle)
class YawParameters():
    def __init__(self):
        # constraints regarding yaw
        self.v_min = []  # minimum yaw velocity [rad/s]
        self.v_max = []  # maximum yaw velocity [rad/s]


def extend_vehicle_params(p: VehicleParameters) -> VehicleParameters:
    p.yaw = YawParameters()
    p.yaw.v_min = -2.  # minimum yaw velocity [rad/s]
    p.yaw.v_max = 2.  # maximum yaw velocity [rad/s]
    return p


class YawRateDynamics(VehicleDynamics):
    """
    Description:
        Class for the calculation of vehicle dynamics of YawRate vehicle model
    """

    def __init__(self, vehicle_type: VehicleType):
        super(YawRateDynamics, self).__init__(VehicleModel.YawRate, vehicle_type)
        self.parameters = extend_vehicle_params(self.parameters)
        self.l = self.parameters.a + self.parameters.b

        self.velocity = None

    def dynamics(self, t, x, u) -> List[float]:
        """
        Yaw Rate model dynamics function.

        :param x: state values, [position x, position y, steering angle, longitudinal velocity, orientation(yaw angle)]
        :param u: input values, [longitudinal acceleration, yaw rate]

        :return: system dynamics
        """
        velocity_x = x[3] * np.cos(x[4])
        velocity_y = x[3] * np.sin(x[4])
        self.velocity = x[3]

        # steering angle velocity depends on longitudinal velocity and yaw rate (as well as vehicle parameters)
        steering_ang_velocity = -u[1] * self.l / (x[3] ** 2 + u[1] * self.l ** 2)

        return [velocity_x, velocity_y, steering_ang_velocity, u[0], u[1]]

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of BaseVehicle Model in order to return bounds for the Yaw Rate Model inputs.

        Bounds are
            - -max longitudinal acc <= acceleration <= max longitudinal acc
            - mini yaw velocity <= yaw_rate <= max yaw velocity

        :return: Bounds
        """
        # self.parameters.yaw.v_max = np.abs(self.parameters.longitudinal.a_max / self.velocity)
        # self.parameters.yaw.v_min = -self.parameters.yaw.v_max
        return Bounds([-self.parameters.longitudinal.a_max, self.parameters.yaw.v_min - 1e-4],
                      [self.parameters.longitudinal.a_max, self.parameters.yaw.v_max + 1e-4])

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0],
            state.position[1],
            getattr(state, 'steering_angle', steering_angle_default),  # not defined in initial state
            state.velocity,
            state.orientation,
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
        }
        state = State(**values, time_step=time_step)
        return state

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """
        Actual conversion of input to array happens here. Vehicles can override this method to implement their own converter.
        """
        values = [
            input.acceleration,
            input.yaw_rate,
        ]
        time_step = input.time_step
        return np.array(values), time_step

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """
        Actual conversion of input array to input happens here. Vehicles can override this method to implement their
        own converter.
        """
        values = {
            'acceleration': u[0],
            'yaw_rate': u[1],
        }
        return State(**values, time_step=time_step)
