from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from commonroad_geometric.learning.geometric.training.geometric_trainer import GeometricTrainerConfig


@dataclass
class GeometricDatasetConfig:
    train_scenario_dir: Path = MISSING
    test_scenario_dir: Path = MISSING
    val_scenario_dir: Path = MISSING
    overwrite: bool = False
    pre_transform_workers: int = 1
    cache_data: bool = False
    max_samples_per_scenario: Optional[int] = None
    max_scenarios: Optional[int] = None


@dataclass
class GeometricProjectConfig:
    training: GeometricTrainerConfig = GeometricTrainerConfig()
    dataset: GeometricDatasetConfig = GeometricDatasetConfig()
    project_dir: Path = MISSING
    checkpoint: Optional[str] = None
    seed: int = 0
    wandb_logging: bool = True
    warmstart: bool = False
    profile: bool = False
    logging_level: str = 'info'
    model: dict = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)
    cmd: str = ""
    device: str = 'auto'
    disable_postprocessing_inference: bool = False
    # Add a generic field for additional configurations
    additional_config: Dict[str, Any] = field(default_factory=dict)

cs = ConfigStore.instance()
cs.store(name="base_geometric_config", node=GeometricProjectConfig)
