from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, NamedTuple, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commonroad_geometric.common.torch_utils.helpers import assert_size

T_VehicleStates = TypeVar("T_VehicleStates")


class VehicleModel(ABC, Generic[T_VehicleStates]):

    @property
    @abstractmethod
    def num_input_dims(self) -> int:
        ...

    @property
    @abstractmethod
    def num_state_dims(self) -> int:
        ...

    @abstractmethod
    def compute_next_state(self, states: T_VehicleStates, input: Tensor, dt: float) -> T_VehicleStates:
        ...

    @abstractmethod
    def compute_single_step_loss(self, prediction: T_VehicleStates, target: T_VehicleStates) -> Dict[str, Tensor]:
        ...


class RelativePositionVehicleStates(NamedTuple):
    position: Tensor

    @staticmethod
    def num_state_dims() -> int:
        return 2

    def to_tensor(self) -> Tensor:
        return torch.cat([
            self.position,
        ], dim=-1)

    @staticmethod
    def from_tensor(t: Tensor) -> RelativePositionVehicleStates:
        assert_size(t, (None, 2))
        return RelativePositionVehicleStates(
            position=t[:, 0:2],
        )


class RelativePositionVehicleModel(VehicleModel[RelativePositionVehicleStates]):

    def __init__(
        self,
        max_velocity: float,  # [m/s]
        max_orientation_delta: float,  # symmetric
    ):
        assert max_velocity > 0. and max_orientation_delta > 0.
        self.max_velocity = max_velocity
        self.max_orientation_delta = max_orientation_delta

    @property
    def num_input_dims(self) -> int:
        return 2

    @property
    def num_state_dims(self) -> int:
        return RelativePositionVehicleStates.num_state_dims()

    def compute_next_state(
        self,
        states: RelativePositionVehicleStates,
        input: Tensor,
        dt: float,
    ) -> RelativePositionVehicleStates:
        assert_size(input, (None, self.num_input_dims))

        position_delta_local_ = torch.tanh(input[:, 0:2])  # position_delta_local_ has a maximum max-norm of 1
        # scale position_delta_local_ such that its maximum 2-norm is 1
        position_delta_local = max_norm_to_2_norm(position_delta_local_)
        # then scale such that its maximum 2-norm is self.max_velocity
        position_delta_local = self.max_velocity * position_delta_local
        # rotate such that position_delta is relative to the new orientation
        # rotation_matrices = rotation_matrices_2d(states.orientation, dtype=input.dtype, device=input.device)
        # torch.mv does not support batching so use for-loop instead
        # position_delta = torch.empty_like(position_delta_local)
        # for i in range(input.size(0)):  # TODO vectorize like in KinematicSingleTrackVehicleModel
        #    position_delta[i] = rotation_matrices[i] @ position_delta_local[i]

        # orientation_delta = self.max_orientation_delta * torch.tanh(input[:, 2:3])
        # orientation = states.orientation + dt * orientation_delta

        return RelativePositionVehicleStates(
            position=states.position + dt * position_delta_local,
            # orientation=orientation,
        )

    def compute_single_step_loss(
        self,
        prediction: RelativePositionAndOrientationVehicleStates,
        target: RelativePositionAndOrientationVehicleStates,
    ) -> Dict[str, Tensor]:
        loss_fn = F.mse_loss
        # Huber loss for reduced sensitivity against outliers
        # loss_fn = F.huber_loss
        return {
            "primary": sum((
                loss_fn(prediction.position, target.position),
                # loss_fn(torch.zeros_like(prediction.orientation), relative_orientation(prediction.orientation, target.orientation)),
            )),
            "position_only": loss_fn(prediction.position.detach(), target.position.detach()),
        }


class RelativePositionAndOrientationVehicleStates(NamedTuple):
    position: Tensor
    orientation: Tensor

    @staticmethod
    def num_state_dims() -> int:
        return 3

    def to_tensor(self) -> Tensor:
        return torch.cat([
            self.position,
            self.orientation,
        ], dim=-1)

    @staticmethod
    def from_tensor(t: Tensor) -> RelativePositionAndOrientationVehicleStates:
        assert_size(t, (None, 3))
        return RelativePositionAndOrientationVehicleStates(
            position=t[:, 0:2],
            orientation=t[:, 2:3],
        )


class RelativePositionAndOrientationVehicleModel(VehicleModel[RelativePositionAndOrientationVehicleStates]):

    def __init__(
        self,
        max_velocity: float,  # [m/s]
        max_orientation_delta: float,  # symmetric
    ):
        assert max_velocity > 0. and max_orientation_delta > 0.
        self.max_velocity = max_velocity
        self.max_orientation_delta = max_orientation_delta

    @property
    def num_input_dims(self) -> int:
        return 3

    @property
    def num_state_dims(self) -> int:
        return RelativePositionAndOrientationVehicleStates.num_state_dims()

    def compute_next_state(
        self,
        states: RelativePositionAndOrientationVehicleStates,
        input: Tensor,
        dt: float,
    ) -> RelativePositionAndOrientationVehicleStates:
        assert_size(input, (None, self.num_input_dims))
        # input: vehicle x (position delta x, position delta y, orientation delta)
        # 1. update position, relative to previous position with previous orientation
        # 2. update orientation

        position_delta_local_ = torch.tanh(input[:, 0:2])  # position_delta_local_ has a maximum max-norm of 1
        # scale position_delta_local_ such that its maximum 2-norm is 1
        position_delta_local = max_norm_to_2_norm(position_delta_local_)
        # then scale such that its maximum 2-norm is self.max_velocity
        position_delta_local = self.max_velocity * position_delta_local
        # rotate such that position_delta is relative to the new orientation
        rotation_matrices = rotation_matrices_2d(states.orientation, dtype=input.dtype, device=input.device)
        # torch.mv does not support batching so use for-loop instead
        position_delta = torch.empty_like(position_delta_local)
        for i in range(input.size(0)):  # TODO vectorize like in KinematicSingleTrackVehicleModel
            position_delta[i] = rotation_matrices[i] @ position_delta_local[i]

        orientation_delta = self.max_orientation_delta * torch.tanh(input[:, 2:3])
        orientation = states.orientation + dt * orientation_delta

        return RelativePositionAndOrientationVehicleStates(
            position=states.position + dt * position_delta,
            orientation=orientation,
        )

    def compute_single_step_loss(
        self,
        prediction: RelativePositionAndOrientationVehicleStates,
        target: RelativePositionAndOrientationVehicleStates,
    ) -> Dict[str, Tensor]:
        loss_fn = F.mse_loss
        # Huber loss for reduced sensitivity against outliers
        # loss_fn = F.huber_loss
        return {
            "primary": sum((
                loss_fn(prediction.position, target.position),
                loss_fn(
                    torch.zeros_like(
                        prediction.orientation),
                    relative_orientation(
                        prediction.orientation,
                        target.orientation)),
            )),
            "position_only": loss_fn(prediction.position.detach(), target.position.detach()),
        }


def rotation_matrices_2d(phi: Tensor, dtype, device) -> Tensor:
    assert_size(phi, (None, 1))
    N = phi.size(0)
    rotation_matrices = torch.empty((N, 2, 2), dtype=dtype, device=device)
    s, c = torch.sin(phi.squeeze()), torch.cos(phi.squeeze())
    rotation_matrices[:, 0, 0] = c
    rotation_matrices[:, 0, 1] = -s
    rotation_matrices[:, 1, 0] = s
    rotation_matrices[:, 1, 1] = c
    return rotation_matrices


def relative_orientation(a1: Tensor, a2: Tensor) -> Tensor:
    two_pi = 2 * np.pi
    phi = (a2 - a1) % two_pi
    phi[..., phi > np.pi] -= two_pi
    return phi  # in (-pi, pi]


def max_norm_to_2_norm(x_max: Tensor) -> Tensor:
    # scale values with a maximum max-norm of 1 (x_max) to values with a maximum 2-norm of 1 (x_2)
    assert_size(x_max, (None, 2))
    x_max_2norm = torch.norm(x_max, p=2, dim=-1, keepdim=True)
    x_2 = torch.empty_like(x_max)
    norm_zero = x_max_2norm.view(-1) < 1e-6
    norm_non_zero = ~norm_zero
    x_larger_y = x_max[:, 0].abs() >= x_max[:, 1].abs()
    y_larger_x = ~x_larger_y
    # remove zero norm values from masks
    x_larger_y &= norm_non_zero
    y_larger_x &= norm_non_zero
    x_2[x_larger_y] = x_max[x_larger_y] * (x_max[x_larger_y][:, 0:1].abs() / x_max_2norm[x_larger_y])
    x_2[y_larger_x] = x_max[y_larger_x] * (x_max[y_larger_x][:, 1:2].abs() / x_max_2norm[y_larger_x])
    x_2[norm_zero] = x_max[norm_zero]  # copy zero values
    return x_2


class KinematicSingleTrackVehicleStates(NamedTuple):
    position: Tensor
    velocity_long: Tensor
    acceleration_long: Tensor
    orientation: Tensor
    length_wheel_base: Tensor

    @staticmethod
    def num_state_dims() -> int:
        return 5  # length_wheel_base not included

    def to_tensor(self) -> Tensor:
        return torch.cat([
            self.position,
            self.velocity_long,
            self.acceleration_long,
            self.orientation,
        ], dim=-1)

    @staticmethod
    def from_tensor(t: Tensor, length_wheel_base: Tensor) -> KinematicSingleTrackVehicleStates:
        assert_size(t, (None, 5))
        return KinematicSingleTrackVehicleStates(
            position=t[:, 0:2],
            velocity_long=t[:, 2:3],
            acceleration_long=t[:, 3:4],
            orientation=t[:, 4:5],
            length_wheel_base=length_wheel_base,
        )


class KinematicSingleTrackVehicleModel(VehicleModel[KinematicSingleTrackVehicleStates]):

    def __init__(
        self,
        velocity_bounds: Tuple[float, float],
        acceleration_bounds: Tuple[float, float],
        steering_angle_bound: float,  # symmetric bound
    ):
        assert velocity_bounds[0] <= 0. < velocity_bounds[1]
        assert acceleration_bounds[0] < 0. < acceleration_bounds[1]
        assert steering_angle_bound > 0.
        self.velocity_bounds = velocity_bounds
        self.acceleration_bounds = acceleration_bounds
        self.steering_angle_bound = steering_angle_bound

    @property
    def num_input_dims(self) -> int:
        return 2

    @property
    def num_state_dims(self) -> int:
        return KinematicSingleTrackVehicleStates.num_state_dims()

    def compute_next_state(
        self,
        states: KinematicSingleTrackVehicleStates,
        input: Tensor,
        dt: float,
    ) -> KinematicSingleTrackVehicleStates:
        # Input: vehicle x (acceleration, steering angle)
        assert_size(input, (None, self.num_input_dims))

        acceleration_ = torch.tanh(input[:, 0:1])
        positive_acc_mask = acceleration_ >= 0
        acceleration = torch.empty_like(acceleration_)
        acceleration[positive_acc_mask] = self.acceleration_bounds[1] * acceleration_[positive_acc_mask]
        acceleration[~positive_acc_mask] = -self.acceleration_bounds[0] * acceleration_[~positive_acc_mask]
        acceleration = torch.clamp(acceleration, min=self.acceleration_bounds[0], max=self.acceleration_bounds[1])

        velocity_delta = acceleration
        velocity = states.velocity_long + dt * velocity_delta
        velocity = torch.clamp(velocity, min=self.velocity_bounds[0], max=self.velocity_bounds[1])

        steering_angle = self.steering_angle_bound * torch.tanh(input[:, 1:2])

        orientation_delta = torch.tan(steering_angle) * velocity / states.length_wheel_base
        orientation = states.orientation + dt * orientation_delta

        position = states.position + dt * torch.cat([
            velocity * torch.cos(orientation),
            velocity * torch.sin(orientation),
        ], dim=-1)

        return KinematicSingleTrackVehicleStates(
            position=position,
            orientation=orientation,
            velocity_long=velocity,
            acceleration_long=acceleration,
            length_wheel_base=states.length_wheel_base,
        )

    def compute_single_step_loss(
        self,
        prediction: KinematicSingleTrackVehicleStates,
        target: KinematicSingleTrackVehicleStates,
    ) -> Dict[str, Tensor]:
        # TODO support different loss functions: mse, rmse, ade, fde (Notion -> Unimodal Trajectory)
        loss_fn = F.mse_loss
        # Huber loss for reduced sensitivity against outliers
        # loss_fn = F.huber_loss
        return {
            "primary": sum((
                loss_fn(prediction.position, target.position),
                loss_fn(prediction.velocity_long, target.velocity_long),
                loss_fn(prediction.acceleration_long, target.acceleration_long),
                loss_fn(
                    torch.zeros_like(
                        prediction.orientation),
                    relative_orientation(
                        prediction.orientation,
                        target.orientation)),
            )),
            "position_only": loss_fn(prediction.position.detach(), target.position.detach()),
        }


def test_vehicle_model():
    import numpy as np
    import matplotlib.pyplot as plt

    m = "rpo"
    print(f"Testing {m} vehicle model")
    if m == "kst":
        model = KinematicSingleTrackVehicleModel(
            velocity_bounds=(-13.6, 50.8),
            acceleration_bounds=(-11.5, 11.5),
            steering_angle_bound=1.006,
        )
    elif m == "rpo":
        model = RelativePositionAndOrientationVehicleModel(
            max_velocity=10.0,
            max_orientation_delta=torch.pi / 2,
        )

    dt = 0.2
    for T in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if m == "kst":
            states = [KinematicSingleTrackVehicleStates(
                position=torch.tensor([[0.0, 0.0]]),
                orientation=torch.tensor([[0.0]]),
                velocity_long=torch.tensor([[0.0]]),
                acceleration_long=torch.tensor([[0.0]]),
                length_wheel_base=torch.tensor([[4.5]]),
            )]
        elif m == "rpo":
            states = [
                RelativePositionAndOrientationVehicleStates(
                    position=torch.tensor([[0.0, 0.0]]),
                    orientation=torch.tensor([[0.0]]),
                ),
            ]

        for t in np.arange(0, T + dt, step=dt)[1:]:
            if m == "kst":
                acceleration = 1.0 if t < 2 else (-1.0 if t > 5 else 0.0)
                steering_angle = -0.1 if t <= 5 else 0.1
                input = torch.tensor([[acceleration, steering_angle]])
            elif m == "rpo":
                input = torch.tensor([[
                    1.0, 0.0,
                    -0.2 if t <= 5 else 0.2,
                ]])
            states.append(
                model.compute_next_state(states=states[-1], input=input, dt=dt)
            )

        x = dt * np.arange(len(states))
        pos = np.array([state.position.numpy() for state in states]).squeeze()
        ori = np.array([state.orientation.item() for state in states]).squeeze()

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_aspect("equal")
        ax.set_xlim(-10, 55)
        ax.set_ylim(-70, 5)
        ax.set_title(f"t = {T:.1f}s")

        ax.plot(pos[:, 0], pos[:, 1], marker="x")
        for state in states:
            ax.arrow(
                x=state.position[0, 0],
                y=state.position[0, 1],
                dx=torch.cos(state.orientation[0, 0]),
                dy=torch.sin(state.orientation[0, 0]),
                width=0.2,
                color="orange",
            )
        # ax.plot(x, ori)
        fig.show()


def test_max_norm_to_2_norm():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    x_max = torch.tensor([  # test vectors
        [1.0, 0.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [-0.8, 1.0],
        [1.0, -1.0],
        [0., -0.8],
        [-0.9, 0.6],
    ])
    x_2 = max_norm_to_2_norm(x_max)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.add_patch(Rectangle(xy=(-1., -1.), width=2., height=2., fill=False, edgecolor="gray"))  # max-norm = 1
    ax.add_patch(Circle(xy=(0., 0.), radius=1., fill=False, edgecolor="gray"))  # 2-norm = 1
    for x in x_max:
        ax.arrow(x=0, y=0, dx=x[0], dy=x[1], width=0.05, length_includes_head=True, color="orange")
    for x in x_2:
        ax.arrow(x=0, y=0, dx=x[0], dy=x[1], width=0.05, length_includes_head=True, color="blue")
    fig.show()


if __name__ == "__main__":
    test_vehicle_model()
    # test_max_norm_to_2_norm()
