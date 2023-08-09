class PIDController:
    """PID Controller
    """

    def __init__(self, k_P: float, k_I: float, k_D: float, windup_guard: float = 10.0, d_threshold: float = 0.0):

        self._k_P = k_P
        self._k_I = k_I
        self._k_D = k_D
        self._windup_guard = windup_guard
        self._d_threshold = d_threshold

        self.clear()

    def clear(self) -> None:
        """Clears PID computations and coefficients"""

        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0
        self._last_error = 0.0
        self._int_error = 0.0
        self._output = 0.0

    def __call__(self, error: float, dt: float) -> float:
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
        """

        self._p_term = self._k_P * error

        if self._k_I > 0:
            self._i_term += error * dt
            if (self._i_term < -self._windup_guard):
                self._i_term = -self._windup_guard
            elif (self._i_term > self._windup_guard):
                self._i_term = self._windup_guard

        if self._k_D > 0:
            delta_error = error - self._last_error
            self._d_term = 0.0
            self._d_term = delta_error / dt
            if abs(self._d_term) <= self._d_threshold:
                self._d_term = 0.0
            # Remember last time and last error for next calculation
            self._last_error = error

        self._output = self._p_term + (self._k_I * self._i_term) + (self._k_D * self._d_term)

        return self._output
