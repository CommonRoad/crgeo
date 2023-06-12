import numpy as np

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin

class OrnsteinUhlenbeckProcess(AutoReprMixin):
    """
    A Ornstein Uhlenbeck noise process designed to approximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    """

    def __init__(self, mean: float, sigma: float, theta: float = 0.15, dt: float = 1e-2):
        super().__init__()
        self._theta = np.array([theta])
        self._mu = np.array([mean])
        self._sigma = sigma
        self._dt = dt
        self._noise_prev: np.ndarray
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = self._noise_prev + self._theta * (self._mu - self._noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self._noise_prev = noise
        return noise

    def reset(self) -> None:
        self._noise_prev = np.zeros_like(self._mu)