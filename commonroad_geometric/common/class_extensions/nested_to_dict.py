from enum import Enum
from typing import Any, Dict
from copy import deepcopy
import logging


logger = logging.getLogger(__name__)


def _recursive_to_dict(instance: Any) -> Any:
    if isinstance(instance, list):
        return str(instance)
    if not hasattr(instance, "__dict__"):
        return instance
    if issubclass(type(instance), type):
        return instance.__name__
    if isinstance(instance, Enum):
        return str(instance)
    new_subdic = vars(instance)
    for key, value in new_subdic.items():
        try:
            new_subdic[key] = _recursive_to_dict(value)
        except Exception as e:
            logger.error(e, exc_info=True)
            new_subdic[key] = None
    return new_subdic


def nested_to_dict(instance: Any) -> Dict:
    try:
        obj_copy = deepcopy(instance)
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}
    return _recursive_to_dict(obj_copy)
