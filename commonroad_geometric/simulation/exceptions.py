class SimulationRuntimeError(RuntimeError):
    pass


class SimulationNotYetStartedException(AttributeError):
    def __init__(self, attribute: str):
        message = f"BaseSimulation attribute '{attribute}' cannot be accessed as the simulation has not yet started (BaseSimulation.start was never called)."
        self.message = message
        super(SimulationNotYetStartedException, self).__init__(message)
