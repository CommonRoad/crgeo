from datetime import datetime
from typing import Optional, Union


def get_timestamp(ts: Optional[Union[int, float]] = None) -> str:
    dt = datetime.now() if ts is None else datetime.fromtimestamp(ts)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def get_timestamp_filename(ts: Optional[Union[int, float]] = None, include_time: bool = True) -> str:
    dt = datetime.now() if ts is None else datetime.fromtimestamp(ts)
    fmt = '%Y-%m-%d-%H%M%S' if include_time else '%Y-%m-%d'
    return dt.strftime(fmt)