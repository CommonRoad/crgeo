
from typing import Callable
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallbackParams

epoch_callback: Callable[[BaseCallbackParams], None] = lambda params: print(f"Epoch {params.ctx.epoch}")
