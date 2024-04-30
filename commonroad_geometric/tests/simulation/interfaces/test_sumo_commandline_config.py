import os
from pathlib import Path

from commonroad_geometric.external.sumocr.sumo_config.pathConfig import \
    SUMO_BINARY
from commonroad_geometric.simulation.interfaces.interactive.sumo_commandline_config import \
    SumoCommandLineConfig


def test_generate_command_float():
    config = SumoCommandLineConfig(
        step_length=0.04
    )
    command = config.generate_command()
    idx = command.index('--step-length')
    assert command[idx + 1] == '0.04'


def test_generate_command_boolean():
    config = SumoCommandLineConfig(
        tls__all_off=True
    )
    command = config.generate_command()
    idx = command.index('--tls.all-off')
    assert command[idx + 1] == 'true'


def test_save_config():
    config = SumoCommandLineConfig(
        tls__all_off=True
    )
    config.save_config()
    config_path = os.path.join(Path(__file__).parents[3],
        'simulation/interfaces/interactive/config_files/sumo_saved_config.xml')
    assert os.path.exists(config_path)
