import numpy as np
from typing import Any
import json
import ast
import re


def resolve_string(s: str) -> Any:
    try:
        return json.loads(s.lower())
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except ValueError:
            return s


def rchop(s: str, sub: str) -> str:
    return s[:-len(sub)] if s.endswith(sub) else s


def lchop(s: str, sub: str) -> str:
    return s[len(sub):] if s.startswith(sub) else s


def lstrip_n(s: str, n: int, sub: str = ' ') -> str:
    tmp = s
    for idx, char in enumerate(s):
        if idx >= n:
            return tmp
        if char == sub:
            tmp = tmp[1:]
        else:
            return tmp
    return tmp
    
    
def numpy_prettify(x: Any, precision: int = 3) -> str:
    # TODO: Make less hacky
    precision_fmt = '{:.' + str(precision) + 'f}'
    if isinstance(x, (int, str)):
        return f"{x}"
    x = x.squeeze() if isinstance(x, np.ndarray) else np.array(x).squeeze()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        item: float = x.item()
        try:
            if item.is_integer():
                return str(item)
            return precision_fmt.format(item)
        except Exception:
            return str(item)
    if isinstance(x, np.ndarray) and x.ndim == 1 and len(x) == 1:
        item = x[0]
        try:
            if item.is_integer():
                return str(item)
            return precision_fmt.format(item)
        except Exception:
            return str(item)
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    return "[" + ', '.join([precision_fmt.format(i) for i in x]) + "]"


def get_indent(astr: str) -> int:

    if len(astr) == 0:
        return 0

    """Return index of first non-space character of a sequence else False."""

    try:
        iter(astr)
    except:
        raise

    # OR for not raising exceptions at all
    # if hasattr(astr,'__getitem__): return False

    idx = 0
    while idx < len(astr) and astr[idx] == ' ':
        idx += 1
    if astr[0] != ' ':
        return False
    return idx


def filter_alpha(s: str) -> str:
    return re.sub('[^a-zA-Z]+', '', s)