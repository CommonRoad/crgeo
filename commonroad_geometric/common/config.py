from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, Optional, Tuple, Dict, Sequence, Iterable

unset = object()


class ImmutableError(Exception):
    pass


class Config:

    __slots__ = ("_data", "_root_cfg")

    def __init__(self, data: Dict[str, Any], *, _root: Optional[Config] = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_root_cfg", _root)

    @classmethod
    def load_from_json_file(cls, path: Path) -> Config:
        with path.open("rt", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def load_from_yaml_file(cls, path: Path) -> Config:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("load_from_yaml_file requires PyYAML package") from e

        with path.open("rt", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return cls(config)

    def as_dict(self) -> Dict[str, Any]:
        return {
            k: v.as_dict() if isinstance(v, Config) else v
            for k, v in self._data.items()
        }

    def overlay(self, data: Union[Dict[str, Any], Config]) -> Config:
        if isinstance(data, Config):
            data = data._data
        c = self.__class__(self._data, _root=self._root_cfg)
        for k, v in data.items():
            if isinstance(v, (dict, Config)) and k in c._data and isinstance(c._data[k], (dict, Config)):
                c._data[k] = c[k].overlay(v)
            else:
                c._data[k] = v
        return c

    def mutable(self) -> MutableConfig:
        return MutableConfig(self._data, _root=self._root_cfg)

    def value_by_path(self, path: Sequence[str]) -> Any:
        if len(path) == 0:
            raise ValueError("path cannot be empty")
        elif len(path) == 1:
            return self[path[0]]
        return self[path[0]].value_by_path(path[1:])

    def set_key_value(self, key: str, value: Any) -> None:
        key_split = key.split(".", maxsplit=1)
        our_key = key_split[0]
        if len(our_key) == 0:
            raise KeyError("empty key")
        elif len(key_split) == 1:
            self._data[our_key] = value
        else:
            rest_key = key_split[1]
            if our_key not in self:
                self._data[our_key] = {}
            child = self[our_key]
            if isinstance(child, Config):
                child.set_key_value(rest_key, value)
            else:
                raise ValueError(f"Cannot set {key} because self.{our_key} is not a Config instance")

    def get(self, key: str, *, default: Any = unset) -> Any:
        if default is not unset and key not in self._data:
            return default
        return self.__getattr__(key)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def values(self) -> Iterable[Any]:
        for key in self._data.keys():
            yield self.__getattr__(key)

    def items(self) -> Iterable[Tuple[str, Any]]:
        for key in self._data.keys():
            yield key, self.__getattr__(key)

    def __getattr__(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(key)
        val = self._data[key]
        if isinstance(val, dict):
            val = self._data[key] = Config(val, _root=self._root_cfg or self)
            return val

        elif isinstance(val, str) and val.startswith("$"):
            val = self._data[key] = ConfigReference(path=tuple(val[1:].split(".")))

        if isinstance(val, ConfigReference):
            root_cfg = self if self._root_cfg is None else self._root_cfg
            return root_cfg.value_by_path(path=val.path)

        return val

    def __setattr__(self, key: str, value: Any) -> None:
        raise ImmutableError("Config objects are immutable")

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key=key, value=value)

    def __contains__(self, key: str):
        return key in self._data

    def __getstate__(self) -> Tuple[Dict[str, Any], Optional[Config]]:
        return self._data, self._root_cfg

    def __setstate__(self, state: Tuple[Dict[str, Any], Optional[Config]]) -> None:
        data, root_cfg = state
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_root_cfg", root_cfg)

    def __deepcopy__(self, memo: Optional[dict] = None):
        root = copy.deepcopy(self._root_cfg, memo=memo)
        data = copy.deepcopy(self._data, memo=memo)
        return self.__class__(data, _root=root)

    def __repr__(self) -> str:
        return repr(self.as_dict())

    def __str__(self) -> str:
        from pprint import pformat
        return pformat(self.as_dict())

class MutableConfig(Config):

    def immutable(self) -> Config:
        return Config(self._data)

    def __getattr__(self, key: str) -> Any:
        val = super().__getattr__(key)
        if isinstance(val, Config) and not isinstance(val, MutableConfig):
            return val.mutable()
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value


@dataclass(frozen=True)
class ConfigReference:
    path: Tuple[str]
