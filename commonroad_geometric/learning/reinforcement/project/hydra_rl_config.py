from pathlib import Path
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLTrainerConfig, RLModelConfig

@dataclass
class RLProjectConfig:
    project_dir: Path = MISSING
    scenario_dir: Path = MISSING
    checkpoint: Optional[str] = None
    seed: int = 0
    warmstart: bool = False
    profile: bool = False
    model: dict = field(default_factory=dict)
    training: RLTrainerConfig = RLTrainerConfig()
    feature_extractor: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)
    device: str = 'auto'
    cmd: str = ""


cs = ConfigStore.instance()
cs.store(name="base_rl_config", node=RLProjectConfig)
