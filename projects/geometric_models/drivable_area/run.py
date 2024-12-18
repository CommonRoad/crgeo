import sys
import os; sys.path.insert(0, os.getcwd())
import hydra
from omegaconf import OmegaConf
from commonroad_geometric.learning.geometric.project.hydra_geometric_config import GeometricProjectConfig
from projects.geometric_models.drivable_area.project import DrivableAreaProject


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: GeometricProjectConfig) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    project = DrivableAreaProject(cfg=cfg_obj)
    project.run(cfg_obj.cmd)


if __name__ == '__main__':
    main()
