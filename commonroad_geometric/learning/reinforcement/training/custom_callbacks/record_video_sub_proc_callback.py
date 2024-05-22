import subprocess
import sys
import os
import logging
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

from commonroad_geometric.learning.reinforcement.training.utils.render_agent import PYTHON_PATH_RECORD_AGENT


logger = logging.getLogger(__name__)


class RecordVideoSubprocCallback(BaseCallback):
    """
    Records a video every ``video_frequency`` timestep.

    :param video_frequency: Number of timesteps between two recordings.
    """

    def __init__(
        self,
        scenario_dir: Path,
        experiment_file: Path,
        video_frequency: int,
        record_backoff: float,
        video_folder: Path,
        video_length: int,
        device: str
    ):
        super(RecordVideoSubprocCallback, self).__init__()
        self._scenario_dir = os.path.abspath(scenario_dir)
        self._experiment_file = os.path.abspath(experiment_file)
        self._video_frequency = video_frequency
        self._record_backoff = record_backoff
        self._video_folder = video_folder
        self._video_length = video_length
        self._last_time_trigger = 0
        self._device = device

    def _on_step(self) -> bool:
        if self._last_time_trigger == 0 or (self.num_timesteps - self._last_time_trigger) >= self._video_frequency:
            self._last_time_trigger = self.num_timesteps
            self._video_frequency = int(self._video_frequency * self._record_backoff)
            return self._on_event()
        return True

    def _on_event(self) -> bool:
        video_folder = Path(self._video_folder).resolve().joinpath(f"{self.num_timesteps}-steps").absolute()
        video_folder.mkdir(parents=True, exist_ok=True)
        model_path = self._video_folder.joinpath("model")

        self.model.save(model_path)
        cmd = f'{sys.executable} "{PYTHON_PATH_RECORD_AGENT}" ' + \
              f'--cwd "{os.getcwd()}" ' + \
              f"--model-cls {self.model.__class__.__name__} " + \
              f'--model-path "{model_path}" ' + \
              f'--experiment-path "{self._experiment_file}" ' + \
              f'--scenario-dir "{self._scenario_dir}" ' + \
              f'--video-folder "{video_folder}" ' + \
              f'--video-length {self._video_length} ' + \
              f'--device {self._device}'
        self.logger.info(f"Spawned rendering subprocess {PYTHON_PATH_RECORD_AGENT}")
        self.logger.debug(f"Executed command was: {cmd}")
        subprocess.Popen(cmd, shell=True)
        return True
