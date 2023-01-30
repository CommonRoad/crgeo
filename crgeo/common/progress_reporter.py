from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Literal, Optional, Type, Union, cast

from tqdm.autonotebook import tqdm
from typing_extensions import Literal

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin


class BaseProgressReporter(ABC, AutoReprMixin):

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        total: Union[int, float] = 0,
        initial: Union[int, float] = 0,
        unit: Optional[str] = None
    ) -> None:
        pass

    def __enter__(self) -> BaseProgressReporter:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> Literal[False]:
        self.close()
        return False

    @abstractmethod
    def nested_progress(self) -> Type[BaseProgressReporter]:
        ...

    @abstractmethod
    def set_postfix_str(self, s: str, *, refresh: bool = True) -> None:
        ...

    @abstractmethod
    def display_memory_usage(self) -> None:
        ...

    @abstractmethod
    def update(self, progress: Union[int, float]) -> None:
        ...

    @abstractmethod
    def write(self, message: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ProgressReporter(BaseProgressReporter):

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        total: Union[int, float] = 0,
        initial: Union[int, float] = 0,
        unit: Optional[str] = None,
        lazy: bool = False,
        parent_reporter: Union[None, int, ProgressReporter] = None,
        report_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._depth: int
        if parent_reporter is None:
            self._depth = 0
        elif isinstance(parent_reporter, int):
            self._depth = parent_reporter
        else:
            self._depth = parent_reporter._depth + 1
        self._percent = total <= 0

        if unit is None:
            unit = "%" if self._percent else "it"

        self._tqdm_kwargs = dict(
            desc=name,
            leave=False,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            position=self._depth,
            total=100.0 if self._percent else total,
            initial=initial,
            unit=unit,
            **kwargs,
        )
        self._tqdm: Optional[tqdm] = None if lazy else tqdm(**self._tqdm_kwargs)
        self._tqdm_memory: Optional[tqdm] = tqdm(position=self._depth + 1, leave=False) if report_memory else None

    def _init_tqdm(self) -> tqdm:
        if self._tqdm is None:
            self._tqdm = tqdm(**self._tqdm_kwargs)
        return self._tqdm

    def nested_progress(self) -> Type[ProgressReporter]:
        return cast(Type[ProgressReporter], functools.partial(ProgressReporter, parent_reporter=self))

    def set_postfix_str(self, s: str, *, refresh: bool = True, append: bool = False) -> None:
        tqdm_inst = self._init_tqdm()
        if append and tqdm_inst.postfix:
            s = tqdm_inst.postfix + ' | ' + s
        tqdm_inst.set_postfix_str(s, refresh=refresh)

    def display_memory_usage(self) -> None:
        import psutil
        ram_percentage = psutil.virtual_memory().percent
        cpu_percentage = psutil.cpu_percent()
        msg = f"ram: {ram_percentage}%, cpu: {cpu_percentage}%"
        self.set_postfix_str(msg, refresh=True, append=False)

    def update(self, progress: Union[int, float]) -> None:
        tqdm_inst = self._init_tqdm()
        tqdm_inst.n = 0
        if self._percent:
            progress = round(100 * progress, ndigits=1)
        tqdm_inst.update(n=progress)

        if self._tqdm_memory is not None:
            import psutil
            self._tqdm_memory.n = psutil.virtual_memory().percent
            self._tqdm_memory.refresh()

    def write(self, message: str) -> None:
        tqdm_inst = self._init_tqdm()
        tqdm_inst.write(message)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


class NoOpProgressReporter(BaseProgressReporter):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def nested_progress(self) -> Type[NoOpProgressReporter]:
        return NoOpProgressReporter

    def set_postfix_str(self, s: str, *, refresh: bool = True) -> None:
        pass

    def display_memory_usage(self) -> None:
        pass

    def update(self, progress: Union[int, float]) -> None:
        pass

    def write(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass
