from dataclasses import dataclass


@dataclass
class ExperimentMetadata:
    epochs: int
    batch_size: int
