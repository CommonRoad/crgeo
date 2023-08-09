import sys, os; sys.path.insert(0, os.getcwd())
import hydra
from omegaconf import OmegaConf
from commonroad_geometric.learning.geometric.project.hydra_geometric_config import GeometricProjectConfig
from projects.geometric_models.dummy.project import DummyProject


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: GeometricProjectConfig) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    project = DummyProject(cfg=cfg_obj)
    project.run('custom')
    project.run(cfg_obj.cmd)


if __name__ == '__main__':
    main()

    """
    Example of config overloading from command line (powered by hydra):

    python projects/geometric_models/dummy/run.py 
        cmd=collect 
        dataset.pre_transform_workers=8 
        dataset.max_scenarios=100 
        dataset.max_samples_per_scenario=1000 
        dataset.scenario_dir=../../data/Munich_2
    """
