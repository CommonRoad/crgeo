import os
import sys
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, cast
import numpy as np

from gym.wrappers.record_video import RecordVideo
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from commonroad_geometric.common.logging import stdout
from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import get_most_recent_file, list_files
from commonroad_geometric.common.utils.seeding import set_system_time_seed
from commonroad_geometric.learning.reinforcement.commonroad_gym_env import CommonRoadGymStepInfo
from commonroad_geometric.learning.reinforcement.constants import COMMONROAD_GYM_ENV_ID
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment


PYTHON_PATH_RECORD_AGENT = os.path.realpath(__file__)


def render_agent(
    agent: BaseAlgorithm,
    experiment: Optional[RLExperiment] = None,
    env: Optional[VecEnv] = None,
    scenario_dir: Optional[str] = None,
    video_folder: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    override_step_id: Optional[int] = None,
    break_on_done: bool = False,
    deterministic: Union[bool, Callable[[int], bool]] = False,
    verbose: int = 0,
    seed: Optional[int] = None,
) -> Iterable[Tuple[Dict[str, np.ndarray], float, bool, CommonRoadGymStepInfo]]:
    assert total_timesteps is None or total_timesteps > 0
    assert experiment is not None or env is not None

    if env is None:
        env = experiment.make_env(scenario_dir, seed=seed)
        env.spec = None
        env.reward_range = (-float("inf"), float("inf"))
        seed = seed if seed is not None else set_system_time_seed()
        env.seed(seed)

    if video_folder is not None:
        total_timesteps = total_timesteps if total_timesteps is not None else np.inf
        venv = VecVideoRecorder(
            RecordVideo(env, os.path.join(video_folder, "gym-results")), 
            video_folder=video_folder, 
            record_video_trigger=lambda x: x == 0, 
            video_length=total_timesteps, 
            name_prefix=f"{COMMONROAD_GYM_ENV_ID}-{env.get_attr('current_scenario_id')[0]}-{get_timestamp_filename()}"
        )
    else:
        venv = env

    if override_step_id is not None:
        venv.step_id = override_step_id
    obs = venv.reset()
    
    if verbose > 0 and video_folder is not None:
        progress = ProgressReporter(total=total_timesteps, unit="epoch")

    t = 0
    total_reward = 0.0
    while True:
        deterministic_step = deterministic if isinstance(deterministic, bool) else deterministic(t)
        action, states = agent.predict(obs, deterministic=deterministic_step)
        obs, reward, done, info = venv.step(action)
        reward = float(reward)
        info = info[0] if isinstance(info, list) else info

        yield obs, reward, done, info

        total_reward += reward
        if video_folder is None and not venv.get_attr('options')[0].render_on_step:
            venv.render()
        if verbose > 0:
            if video_folder is not None:
                progress.update(t + 1)
            else:
                stdout(f"reward: {reward:.3f}, cum. reward: {total_reward:.3f}, t: {info['time_step']}, deterministic: {deterministic_step}")
        if done:
            if break_on_done:
                break
            elif video_folder is not None:
                venv.env.reset()
            else:
                venv.reset()
        t += 1
        if total_timesteps is not None and t >= total_timesteps:
            break

    venv.close()
    if verbose > 0 and video_folder is not None:
        progress.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Record RL agent"
    )
    parser.add_argument("--cwd", type=str, help="path to working directory")
    parser.add_argument("--model-cls", type=str, help="model class (e.g. 'PPO')")
    parser.add_argument("--model-path", type=str, help="path to RL agent")
    parser.add_argument("--experiment-path", type=str, help="path to RL experiment file")
    parser.add_argument("--scenario-dir", type=str, help="path to CommonRoad scenario dir")
    parser.add_argument("--video-folder", type=str, help="output folder. if not specified, no video will be saved")
    parser.add_argument("--video-length", type=int, help="length of recordings", default=2000)
    parser.add_argument("--search-model", action="store_true", help="search for latest model")
    parser.add_argument("--break-on-done", action="store_true", help="stop recording on episode end")
    parser.add_argument("--n-videos", type=int, help="number of videos", default=1)
    parser.add_argument("--seed", type=int, help="seeding", default=0)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    if args.cwd is not None:
        sys.path.insert(0, args.cwd)

    if args.search_model:
        # will return most recently saved model
        model_path = get_most_recent_file(list_files(args.model_path, file_type='zip', sub_directories=True))
        experiment_path = get_most_recent_file(list_files(args.model_path, file_name='experiment_config', file_type='pkl', sub_directories=True))
    else:
        model_path = args.model_path
        experiment_path = args.experiment_path

    import stable_baselines3
    model_cls = cast(BaseAlgorithm, getattr(stable_baselines3, args.model_cls))
    agent = model_cls.load(model_path, device=args.device)

    experiment = RLExperiment.load(experiment_path)

    env = experiment.make_env(args.scenario_dir, seed=args.seed)
    env.spec = None
    env.reward_range = (-float("inf"), float("inf"))
    seed = args.seed if args.seed is not None else set_system_time_seed()
    env.seed(seed)

    for i in range(args.n_videos):
        list(render_agent(
            experiment=experiment,
            env=env,
            scenario_dir=args.scenario_dir,
            video_folder=args.video_folder,
            total_timesteps=args.video_length,
            agent=agent,
            deterministic=True,
            break_on_done=args.break_on_done
        ))
