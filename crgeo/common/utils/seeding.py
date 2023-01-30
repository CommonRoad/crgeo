import numpy as np
import random
from datetime import datetime


def set_global_seed(seed: int) -> None:
    """Sets random seeds."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_random_seed() -> int:
    """
    _summary_

    Returns:
        int: _description_
    """
    seed = random.getrandbits(32)
    return seed


def set_system_time_seed() -> int:
    import torch
    seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed