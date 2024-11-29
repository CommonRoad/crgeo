import inspect
import logging
from typing import Any, Dict, List

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.types import AnyCallable, T_AnyReturn
from commonroad_geometric.common.utils.sleep import _Sleeper
from commonroad_geometric.common.utils.string import get_indent, lstrip_n

logger = logging.getLogger(__name__)


class DelayedExecutor(AutoReprMixin):
    """
    The DelayedExecutor facilitates function executions with injected sleep delays, implicitely letting
    the user control the priority of a thread in a multithreaded program execution.

    The motivation for this was to alleviate the bottleneck caused by how rollout data is collected
    from parallel environments in Stable Baselines 3, where at each step, the root process waits for each worker
    to return the current observations. Since our gym environment's reset() call was significantly slower than the step() calls
    (by many orders of magnitude), this introduced a bottleneck where each process was regulary forced to wait for a single process to
    complete a time-expensive reset call. By instead spreading the reset compute workload over the preceding step() calls, this problem
    is avoided. However, as one cannot easily set the thread scheduling priorities in Python, the reset thread will naturally be allocated too much CPU time,
    the bottleneck remains present (albeit to a smaller extend) even if the reset is prepared in advance in an asynchronous manner.

    By introducing sleep statements everywhere within the subroutines involved in the reset procedure, however,
    we are able to lower the priority of the reset thread.
    """

    def __init__(self, delay: float, disabled: bool = False) -> None:
        self._enabled = not disabled
        self._delay = delay
        self._sleeper = _Sleeper(delay)

    def __call__(self, fn: AnyCallable[T_AnyReturn], *args: Any, **kwargs: Any) -> T_AnyReturn:
        """
        Calls the specified function with injected delays, inserting sleep statements between each statement.

        Args:
            fn (AnyCallable[T_AnyReturn]): The function to call.
            delay (float): The injected delay length in seconds.

        Returns:
            AnyCallable[T_AnyReturn]: The function output.
        """
        if not self._enabled or self._delay == 0.0:
            return fn(*args, **kwargs)

        logger.debug(f"Created delay-injected version of {fn.__name__}")
        delayed_fn = self.inject_delays(fn=fn)
        result = delayed_fn(*args, **kwargs)
        return result

    def disable(self) -> None:
        """
        Disables all the injected sleep statements, making the thread hurry up.
        """
        self._enabled = False
        self._sleeper.disable()
        logger.debug("DelayedExecutor disabled delay injection")

    def enable(self) -> None:
        """
        Enables all the sleep statements
        """
        self._enabled = True
        self._sleeper.enable()
        logger.debug("DelayedExecutor enabled delay injection")

    def set_sleep_duration(self, duration: float) -> None:
        """
        Sets the injected delay duration.

        Args:
            duration (float): New sleep duration in seconds.
        """
        self._sleeper.set_duration(duration)
        logger.debug(f"DelayedExecutor updated sleep duration to {duration:.4f} seconds")

    def multiply_sleep_duration(self, multiplier: float) -> None:
        """
        Updates the injected delay duration by multiplying the current one with the specified number.

        Args:
            multiplier (float): Multiplier to apply.
        """
        self.set_sleep_duration(self._sleeper.duration * multiplier)

    def inject_delays(
        self,
        fn: AnyCallable[T_AnyReturn]
    ) -> AnyCallable[T_AnyReturn]:
        """
            Creates a new function with injected delays.
        """
        if not self._enabled or self._delay == 0.0:
            return fn

        if inspect.isgeneratorfunction(fn):
            # Handle generator functions
            def delayed_gen(*args, **kwargs):
                gen = fn(*args, **kwargs)
                for value in gen:
                    self._sleeper()
                    yield value
            return delayed_gen
        else:
            # Existing code for normal functions
            try:
                source_code = inspect.getsourcelines(fn)[0]
            except (TypeError, OSError) as e:
                logger.debug(f"Cannot inject delays into {fn}: {e}")
                return fn  # Return the original function without modification

        # Proceed with delay injection as before
        sleep_statement = '_Sleeper()'

        # Extract the length (LOC) of the function header
        header_end_line_idx = next((i for i in range(len(source_code))
                                    if source_code[i].rstrip('\n').rstrip(' ').endswith(':')), None)
        if header_end_line_idx is None:
            logger.warning(f"Cannot find function header in {fn}")
            return fn

        # Merge the LOC to a single-line function header
        delayed_fn_name = f"delayed_{fn.__name__}"
        header = ''.join([source_code[i].rstrip('\n').lstrip()
                        for i in range(header_end_line_idx + 1) if len(source_code[i].strip('\n')) > 0])

        base_indent = get_indent(source_code[0])

        # Create new function params, injecting the _Sleeper as a function parameter
        params = ['_Sleeper'] + [k for k in inspect.signature(fn).parameters.keys()]
        if inspect.ismethod(fn):
            params = ['self'] + params

        # Initialize the delay-injected code
        delayed_code: List[str] = [
            lstrip_n(header.replace(fn.__name__, delayed_fn_name, 1).replace('self', 'self, _Sleeper'), base_indent),
        ]

        # Keep track of current indentation level
        prev_indent = base_indent
        skipping = False
        for line_idx in range(header_end_line_idx + 1, len(source_code)):
            line = source_code[line_idx].rstrip('\n')
            skip_next = skipping
            if line.endswith('('):
                skip_next = True
            elif line.endswith(')'):
                skip_next = False
            stripped_line = line.strip()
            if not skipping and len(line.strip()) > 0 and stripped_line[0] != '#':
                this_indent = get_indent(line)
                sleep_indent = max(prev_indent, this_indent)
                sleep_statement_line = lstrip_n(' ' * sleep_indent + sleep_statement, base_indent) + '\n'
                delayed_code.append(sleep_statement_line)
                prev_indent = this_indent
            delayed_code.append(lstrip_n(line, base_indent) + '\n')
            skipping = skip_next

        # Compile new source code as string
        delayed_source_code = '\n'.join(delayed_code)

        # Update context for exec call
        locals().update(inspect.getmodule(fn).__dict__)
        globals().update(fn.__globals__)

        # Create the new function
        exec(delayed_source_code)

        # Obtain the function that was created
        delayed_fn_ = locals().get(delayed_fn_name)

        # Create a wrapper function that injects the _Sleeper
        if inspect.ismethod(fn):
            # For methods, we also inject the self argument
            def delayed_fn_wrapper(*args: Any, **kwargs: Any) -> Any:
                return delayed_fn_(fn.__self__, self._sleeper, *args, **kwargs)
        else:
            def delayed_fn_wrapper(*args: Any, **kwargs: Any) -> Any:
                return delayed_fn_(self._sleeper, *args, **kwargs)
        delayed_fn = delayed_fn_wrapper

        # Update metadata for the new function
        delayed_fn.__dir__ = fn.__dir__
        delayed_fn.__name__ = fn.__name__
        delayed_fn.__qualname__ = fn.__qualname__
        delayed_fn.__module__ = fn.__module__
        delayed_fn.__str__ = fn.__str__
        delayed_fn.__repr__ = fn.__repr__

        return delayed_fn
