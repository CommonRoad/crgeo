import os
import pathlib
import warnings
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image

def save_video_from_frames(
    frames: Sequence[np.ndarray],
    output_file: Path,
    fps: float = 25,
) -> None:
    # TODO: Needs cleanup
    if len(frames) == 0:
        warnings.warn("Trying to save empty video - ignoring")
        return

    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    #!!! TODO fix splitting
    file_type = output_file.suffix.lower()
    if file_type == '.gif':
        imgs = []
        for img in frames:
            # Ensure img is a NumPy array
            array = np.array(img)
            # Check if the array is 4-dimensional and adjust
            if array.ndim == 4 and array.shape[:2] == (1, 1):
                # Reshape to 2D image with 3 color channels (RGB)
                array = array.reshape((array.shape[2], array.shape[3], 3))
            imgs.append(Image.fromarray(array))

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
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        output_file = output_dir.joinpath('capture_' + str(i) + '.png')
        im = Image.fromarray(frame)
        im.save(output_file)