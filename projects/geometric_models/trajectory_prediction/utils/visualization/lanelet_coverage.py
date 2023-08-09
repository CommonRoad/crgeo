import numpy as np
from PIL import Image


def debug_visualization_road_orientation(self, road_orientation: np.ndarray) -> Image.Image:
    buckets, size, _ = road_orientation.shape

    buffer = np.zeros((size, size, 3), dtype=np.uint8)
    num_directions = (road_orientation > 0).sum(axis=0)
    buffer[:] = (num_directions.astype(float) / buckets * 200).astype(np.uint8).reshape(size, size, 1)
    buffer[buffer > 0] += 55

    return Image.fromarray(buffer, mode="RGB")

    # image = ImageData(
    #     width=size,
    #     height=size,
    #     format="RGB",
    #     pitch=-3 * size,
    #     data=buffer.ravel().tobytes(),
    # )
    #
    # self._clear_window()
    # self._set_up_projection_matrix(scale=1, rotation_deg=0, translation=np.array([0, round(size / 2)]))
    # gl.glColor3f(1.0, 1.0, 1.0)
    # image.blit(0, 0)
    # gl.glFlush()


# Image.fromarray((road_coverage * 255).astype(dtype=np.uint8), mode="L")\
#     .save(f"/tmp/tmp.1HyGe4YED0/{random.randrange(0, 100000000)}.png")
