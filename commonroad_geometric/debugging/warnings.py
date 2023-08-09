import warnings
import linecache
from typing import Any, Callable


def debug_warnings(fn: Callable[[], Any]) -> None:
    from commonroad_geometric.debugging.activate_warn_with_traceback import activate_warn_with_traceback
    activate_warn_with_traceback()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("default")

        fn()

        for wi in w:
            if wi.line is None:
                wi.line = linecache.getline(wi.filename, wi.lineno)
            print(wi.category, wi.message)
            print(wi.filename, 'line number {}:'.format(wi.lineno))
            print('line: {}'.format(wi.line))
