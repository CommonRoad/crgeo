import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
from commonroad_geometric.rendering.types import RenderParams, T_Frame


def save_video_from_frames(
    frames: Sequence[np.ndarray],
    output_file: str,
    fps: float = 25,
) -> None:
    # TODO: Needs cleanup
    if len(frames) == 0:
        warnings.warn("Trying to save empty video - ignoring")
        return

    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file_type = output_file.split('.')[-1].lower()
    if file_type == 'gif':
        imgs = [Image.fromarray(img) for img in frames]
        imgs[0].save(output_file, save_all=True, append_images=imgs[1:], duration=1 / fps, loop=0)
    else:
        warnings.warn("WARNING: Output video is most likely corrupt. Consider saving a GIF animation instead.")
        # TODO fix
        import cv2
        out = cv2.VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc(*'H264'),
            fps,
            tuple(frames[0].shape),
            False
        )
        for frame in frames:
            out.write(frame)
        out.release()
    print(f"Saved video to {output_file}")


def save_gif_from_images(
    frames: Union[Sequence[Image.Image], Sequence[np.ndarray]],
    output_file: Path,
    fps: float,
) -> None:
    if len(frames) == 0:
        raise ValueError("Frame sequence is empty")

    images = [
        img if isinstance(img, Image.Image) else Image.fromarray(img)
        for img in frames
    ]
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=1 / fps, loop=0)


def save_images_from_frames(
    frames: Sequence[np.ndarray],
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        output_file = os.path.join(output_dir, 'capture_' + str(i) + '.png')
        im = Image.fromarray(frame)
        im.save(output_file)


def render_scenario_movie(
    scenario: Union[str, Scenario, Sequence[str], Sequence[Scenario]],
    output_file: Optional[str] = None,
    save_pngs: bool = False,
    renderer: Optional[TrafficSceneRenderer] = None,
    **kwargs
):
    # TODO: Needs cleanup
    """Renders a video of the scenario.

    Args:
        scenario (Union[str, Scenario]): Scenario or path to scenario file.
        freeze (bool, optional): Freeze video after the scenario has played, prompt user to continue. Defaults to False.
        replay (bool, optional): Whether to replay the video. Defaults to False.
        output_file (str, optional): Optional output video path. Defaults to None.
        max_timesteps: (int, optional): Optional maximum number of timesteps to render. Defaults to None
    """
    if isinstance(scenario, str):
        scenario, _ = CommonRoadFileReader(filename=scenario).open()
        scenario_list = [scenario]
    elif isinstance(scenario, Scenario):
        scenario_list = [scenario]
    else:
        if isinstance(scenario[0], str):
            scenario_list = [CommonRoadFileReader(filename=s).open()[0] for s in scenario]
        else:
            scenario_list = [s for s in scenario]

    all_frames: List[np.ndarray] = []
    save_video = output_file is not None
    interrupt = None
    for scenario_idx, scenario in enumerate(scenario_list):
        try:
            print(f"Recording from {str(scenario.scenario_id)} (scenario {scenario_idx}/{len(scenario_list)})")
            frames, renderer = _render_scenario_movie(
                scenario,
                renderer=renderer,
                return_renderer=True,
                save_video=save_video,
                **kwargs
            )
            print(f"Recorded {len(frames)} from {str(scenario.scenario_id)}")
            all_frames.extend(frames)
        except KeyboardInterrupt as e:
            interrupt = e

    if save_video:
        save_video_from_frames(all_frames, output_file=output_file)
    if save_pngs:
        png_output_dir = os.path.join(os.path.dirname(output_file), 'pngs')
        os.makedirs(png_output_dir, exist_ok=True)
        save_images_from_frames(all_frames, png_output_dir)

    if renderer is not None:
        renderer.close()

    if interrupt:
        raise interrupt


def _render_scenario_movie(
    scenario: Scenario,
    freeze: bool = False,
    replay: Union[int, bool] = False,
    max_timesteps: Optional[int] = None,
    renderer: Optional[TrafficSceneRenderer] = None,
    return_renderer: bool = False,
    show_scenario_overlay: bool = True,
    save_video: bool = True
) -> List[T_Frame]:
    # TODO: Needs cleanup
    """Renders a video of the scenario.

    Args:
        scenario (Union[str, Scenario]): Scenario or path to scenario file.
        freeze (bool, optional): Freeze video after the scenario has played, prompt user to continue. Defaults to False.
        replay (bool, optional): Whether to replay the video. Defaults to False.
        output_file (str, optional): Optional output video path. Defaults to None.
        max_timesteps: (int, optional): Optional maximum number of timesteps to render.
        Supports negative indexing similar to array indexing. Defaults to None
    """

    if max_timesteps == 0:
        raise ValueError("max_timesteps cannot be zero")

    final_time_step = max(do.prediction.final_time_step for do in scenario.dynamic_obstacles)

    progress = ProgressReporter(total=final_time_step, unit="frame")

    frames = []
    if renderer is None:
        renderer = TrafficSceneRenderer()
    renderer.scenario = scenario
    replay_count = 0
    while 1:
        if max_timesteps is None:
            iter_bound = final_time_step
        elif max_timesteps > 0:
            iter_bound = min(final_time_step, max_timesteps)
        else:
            iter_bound = max(1, final_time_step + max_timesteps)
        for t in range(iter_bound):
            render_kwargs: Dict[str, Any] = dict()
            if show_scenario_overlay:
                render_kwargs['overlays'] = {
                    '': f"{scenario.scenario_id} (t={t})",
                }
            frame = renderer.render(
                return_rgb_array=True,
                render_params=RenderParams(time_step=t, render_kwargs=render_kwargs)
            )
            if save_video:
                frames.append(frame)
            progress.update(t)
            if t % 50 == 0:
                progress.display_memory_usage()
        replay_count += 1
        if freeze:
            input("Press a button to continue")
        if isinstance(replay, int):
            if replay_count >= replay:
                break
        if not replay:
            break

    if return_renderer:
        return frames, renderer
    else:
        renderer.close()
        return frames
