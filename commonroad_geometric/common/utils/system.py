import sys
import multiprocessing
import torch
import numpy as np


def debugger_is_active() -> bool:
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


def get_cpu_count() -> int:
    return multiprocessing.cpu_count()


def get_gpu_count() -> int:
    return torch.cuda.device_count()


def get_gpu_usage() -> float:
    # t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    # f = r-a  # free inside reserved
    return a / r if r > 0 else np.nan
