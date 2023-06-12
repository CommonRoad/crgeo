import os
from pathlib import Path
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Sequence, Tuple, Union
import humanize
import torch

from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import get_file_last_modified_datetime, get_most_recent_file, list_files, load_pickle
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations.voronoi import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.iteration import ScenarioIterator
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric, MODEL_FILE
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment, GeometricExperimentConfig
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions, TrafficSceneRenderer
from commonroad_geometric.rendering.types import RenderParams, T_RendererPlugin
from commonroad_geometric.rendering.video_recording import save_video_from_frames
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

PYTHON_PATH_RENDER_MODEL = os.path.realpath(__file__)


logger = logging.getLogger(__name__)


def render_model(
    model: Union[str, BaseGeometric],
    *,
    experiment: Union[str, GeometricExperiment],
    scenario_path: str,
    renderer_plugins: Optional[Union[str, Sequence[T_RendererPlugin]]] = None,
    video_dir: Optional[str] = None,
    video_length: int = 400,
    video_freq: int = 1000,
    screenshot_freq: Optional[int] = None,
    video_record_backoff: float = 1.0,
    video_size_multiplier: float = 1.0,
    load_freq: Optional[int] = 1,
    loop_scenarios: bool = True,
    disable_overlays: bool = False,
    window_caption: Optional[str] = None,
    shuffle: bool = True,
    rendering_options: Optional[TrafficSceneRendererOptions] = None,
    disable_postprocessing: bool = True,
    disable_compute_loss: bool = True,
    disable_inference: bool = False,
    verbose: bool = True,
    device: Optional[Union[str, torch.device]] = 'cpu'
):
    if verbose:
        logger.info(f"Rendering model '{model}' in scenarios '{scenario_path}'")

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
    def load_model(model_path: str, after_datetime: Optional[datetime]) -> Optional[Tuple[BaseGeometric, datetime]]:
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, MODEL_FILE)
        # logger.debug(f"Loading model from '{model_path}'")
        while 1:
            try:
                last_modified_ts = get_file_last_modified_datetime(model_path)
                if after_datetime is not None and last_modified_ts < after_datetime:
                    return None
                model = BaseGeometric.load(model_path, device=device)
                model.eval()
                break
            except Exception as e:
                time.sleep(0.1)
        return model, last_modified_ts

    model_path_provided = isinstance(model, (Path, str))
    if model_path_provided:
        model_path = model
    model_reloads = 0
    if model_path_provided:
        # Loading model
        model, last_modified_ts = load_model(model_path, None)

    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
    recording_enabled = video_dir is not None
    last_reload_ts = datetime.now()

    if isinstance(experiment, str) and os.path.isfile(experiment):
        experiment = GeometricExperiment.load(experiment, None)
    elif isinstance(experiment, GeometricExperiment):
        pass
    else:
        config = GeometricExperimentConfig(
            extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
                )
            ),
            data_collector_cls=ScenarioDatasetCollector
        )
        experiment = GeometricExperiment(config)

    scenario_iterator = ScenarioIterator(
        scenario_path,
        loop=True,
        save_scenario_pickles=False,
        load_scenario_pickles=False,
        preprocessors=experiment.config.preprocessors,
        shuffle=shuffle
    )

    scenario_bundle = next(scenario_iterator)
    scenario, planning_problem_set = scenario_bundle.preprocessed_scenario, scenario_bundle.planning_problem_set

    def setup_scenario(scenario: Scenario) -> Tuple[BaseSimulation, TrafficExtractor]:
        # Creating simulation instance
        if isinstance(experiment._collector, ScenarioDatasetCollector):
            simulation_cls = ScenarioSimulation
        else:
            from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation
            simulation_cls = SumoSimulation

        simulation = simulation_cls(
            initial_scenario=scenario,
            options=experiment.config.simulation_options
        )

        # Creating traffic extractor
        extractor = experiment.config.extractor_factory.create(
            simulation=simulation
        )

        # Starting simulation loop
        simulation.start()

        return simulation, extractor

    simulation, extractor = setup_scenario(scenario)

    # Defining renderers
    if isinstance(renderer_plugins, str):
        renderer_plugins = load_pickle(renderer_plugins)
    elif renderer_plugins is None:
        renderer_plugins = model.configure_renderer_plugins()
    renderer_options = rendering_options if rendering_options is not None else TrafficSceneRendererOptions(
        window_size_multiplier=video_size_multiplier,
        camera_follow=False,
        skip_redundant=False
    )
    if video_size_multiplier is not None:
        renderer_options.window_size_multiplier = video_size_multiplier # TODO
    if window_caption is not None:
        renderer_options.caption = window_caption

    renderer_options.plugins = renderer_plugins
    video_renderer = TrafficSceneRenderer(renderer_options)
    if screenshot_freq is not None:
        screenshot_renderer = TrafficSceneRenderer(TrafficSceneRendererOptions(
            plugins=renderer_plugins,
            window_size_multiplier=6.0,
            minimize_window=True,
            skip_redundant=False
        ))

    recording: bool = recording_enabled
    video_frames = []
    i = 0
    compute_loss = not disable_compute_loss
    while 1:
        for time_step, scenario in simulation():
            now = datetime.now()
            if model_path_provided and load_freq is not None and i % load_freq == 0:
                model_reload_tuple = load_model(model_path, last_reload_ts)
                if model_reload_tuple is not None:
                    model, last_modified_ts = model_reload_tuple
                    last_reload_ts = now
                    model_reloads += 1
                    # print(f"Reloaded model (reload {model_reloads})")
            
            data = extractor.extract(
                TrafficExtractionParams(
                    index=time_step,
                    disable_postprocessing=disable_postprocessing,
                    device=device,
                    no_skip=True
                )
            )
            if data is None:
                continue
            data = experiment.transform_data(data)

            if not disable_inference:
                output = model.forward(data)
            else:
                output = None
            if compute_loss:
                try:
                    model.compute_loss(output, data, 0)
                except Exception as e:
                    print(f"Exception encountered during model.compute_loss: {repr(e)} ")
                    compute_loss = False

            if disable_overlays:
                overlays = None
            else:
                overlays = dict(
                    reloads=model_reloads,
                    global_step=i,
                    time_step=time_step,
                    scenario=scenario.scenario_id
                )
                if model_path_provided:
                    overlays.update(dict(
                        last_modified=humanize.naturaltime(now - last_modified_ts),
                        last_reload=humanize.naturaltime(now - last_reload_ts),
                    ))
            
            video_frame = simulation.render(
                renderers=[video_renderer],
                return_rgb_array=recording_enabled,
                render_params=RenderParams(
                    render_kwargs=dict(
                        output=output,
                        overlays=overlays,
                    ),
                    data=data
                )
            )

            if screenshot_freq is not None and i % screenshot_freq == 0:
                screenshot_frame = simulation.render(
                    renderers=[screenshot_renderer],
                    return_rgb_array=True,
                    render_params=RenderParams(
                        render_kwargs=dict(
                            output=output,
                            overlays=overlays,
                        ),
                        data=data
                    )
                )
                from PIL import Image
                output_dir = os.path.join(video_dir, 'images')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'capture_{i}_{get_timestamp_filename()}_.png')
                im = Image.fromarray(screenshot_frame)
                im.save(output_file)

            if recording:
                video_frames.append(video_frame)

            elif recording_enabled and i % video_freq == 0:
                recording = True
                video_freq = int(video_record_backoff * video_freq)

            i += 1
            if recording and len(video_frames) == video_length:
                recording = False
                video_name = f"{simulation.current_scenario.scenario_id}-step-{i - video_length}-to-step-{i}.gif"
                output_file = os.path.join(video_dir, video_name)
                print(f"Saving video with {len(video_frames)} frames to {output_file}")
                save_video_from_frames(frames=video_frames, output_file=output_file)
                video_frames = []

        if loop_scenarios:
            scenario_bundle = next(scenario_iterator)
            scenario, planning_problem_set = scenario_bundle.preprocessed_scenario, scenario_bundle.planning_problem_set
            simulation, extractor = setup_scenario(scenario)
        else:
            simulation.reset()


def render_model_from_args(args) -> None:
    model_dir = args.model_dir

    if args.search_model:
        # will return most recently saved model
        model_path = get_most_recent_file(list_files(model_dir, file_name=MODEL_FILE.split('.')[0], file_type=MODEL_FILE.split('.')[-1], sub_directories=True))
        plugins_path = get_most_recent_file(list_files(model_dir, file_name='render_plugins', file_type=None, sub_directories=True))
        experiment_path = get_most_recent_file(list_files(model_dir, file_name='experiment_config', file_type='pkl', sub_directories=True))
    else:
        if os.path.isdir(model_dir):
            model_path = os.path.join(model_dir, MODEL_FILE)
        else:
            model_path = model_dir
        plugins_path = args.plugins
        experiment_path = args.experiment

    render_model(
        model=model_path,
        experiment=experiment_path,
        renderer_plugins=plugins_path,
        scenario_path=args.scenario,
        video_size_multiplier=args.video_size_multiplier,
        video_dir=args.video_dir,
        video_freq=args.video_freq,
        video_length=args.video_length,
        video_record_backoff=args.record_backoff,
        screenshot_freq=args.screenshot_freq,
        load_freq=args.load_freq,
        window_caption=args.window_caption,
        verbose=args.verbose
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Record model with custom rendering plugins"
    )
    parser.add_argument("--cwd", type=str, help="path to working directory")
    parser.add_argument("--scenario", type=str, help="path to scenario file")
    parser.add_argument("--experiment", type=str, help="path to experiment config file")
    parser.add_argument("--model-dir", type=str, help="path to model checkpoint directory")
    parser.add_argument("--search-model", action="store_true", help="search for latest model")
    parser.add_argument("--plugins", type=str, help="path to file containing video_renderer plugins")
    parser.add_argument("--load-freq", type=int, default=1, help="model refresh interval")
    parser.add_argument("--video-dir", type=str, help="output folder")
    parser.add_argument("--video-length", type=int, help="length of recordings", default=400)
    parser.add_argument("--video-freq", type=int, help="frequency of recordings", default=1000)
    parser.add_argument("--record-backoff", type=float, help="backoff factor for recording frequency", default=1.0)
    parser.add_argument("--screenshot-freq", type=int, help="frequency of screenshots")
    parser.add_argument("--window-caption", type=str, help="caption for video")
    parser.add_argument("--verbose", action="store_true", help="activates debug logging")
    parser.add_argument('--video-size-multiplier', type=float, help="optional size multiplier for rendering window")

    args = parser.parse_args()
    if args.cwd is not None:
        sys.path.insert(0, args.cwd)

    render_model_from_args(args)
