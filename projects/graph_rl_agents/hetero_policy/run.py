import sys, os; sys.path.insert(0, os.getcwd())
import hydra
from omegaconf import OmegaConf
from commonroad_geometric.learning.reinforcement.project.hydra_rl_config import RLProjectConfig
from projects.graph_rl_agents.hetero_policy.project import HeteroPolicyProject


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    project = HeteroPolicyProject(cfg=cfg_obj)
    project.run(cfg_obj.cmd)


if __name__ == '__main__':
    main()
