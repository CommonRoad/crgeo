from time import sleep

class _Sleeper:
    def __init__(self, duration: float) -> None:
        self._duration = duration
        self._enabled = True

    def __call__(self) -> None:
        if self._enabled:
            sleep(self._duration)

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def duration(self) -> float:
        return self._duration

    def set_duration(self, duration: float) -> None:
        self._duration = duration
