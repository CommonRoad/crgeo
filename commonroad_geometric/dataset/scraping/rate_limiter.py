import asyncio
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type


log = logging.getLogger(__name__)


class AsyncRateLimiter:

    def __init__(
        self,
        func: Callable,
        *,
        min_delay_seconds: float,
        max_retries: int,
        exception_wait_seconds: Optional[float] = None,
        swallow_exceptions: bool,
        retry_exceptions: Tuple[Type[Exception], ...],
        exception_callback: Optional[Callable] = None,
        return_value_on_swallowed_exception: Any = None,
    ) -> None:
        assert min_delay_seconds >= 0.0
        assert max_retries > 0
        assert exception_wait_seconds is None or exception_wait_seconds >= 0.0
        self._func = func
        self._min_delay_seconds = min_delay_seconds
        self._max_retries = max_retries
        self._exception_wait_seconds = exception_wait_seconds if exception_wait_seconds is not None else min_delay_seconds
        self._swallow_exceptions = swallow_exceptions
        self._retry_exceptions = retry_exceptions
        self._exception_callback = exception_callback
        self._return_value_on_swallowed_exception = return_value_on_swallowed_exception

        self._lock = asyncio.Lock()
        self._last_call = time.perf_counter() - self._min_delay_seconds - 1

    async def __call__(self, *args, **kwargs):
        for i in range(1 + self._max_retries):
            is_last_retry = i == self._max_retries - 1
            async with self._lock:
                delay = self._min_delay_seconds - (time.perf_counter() - self._last_call)
                if delay > 0:
                    await asyncio.sleep(delay)

                try:
                    return await self._func(*args, **kwargs)

                except self._retry_exceptions as e:
                    raise_exception = is_last_retry
                    if self._exception_callback is not None:
                        raise_exception |= self._exception_callback(e)

                    if raise_exception:
                        if self._swallow_exceptions:
                            log.exception("AsyncRateLimiter swallowed exception")
                            return self._return_value_on_swallowed_exception
                        else:
                            raise

                finally:
                    self._last_call = time.perf_counter()

            await asyncio.sleep(self._exception_wait_seconds)
