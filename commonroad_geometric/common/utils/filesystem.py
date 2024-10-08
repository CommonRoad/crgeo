import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, cast

import dill
import numpy as np

from commonroad_geometric.common.utils.datetime import get_timestamp


class FileFormatNotSupportedError(RuntimeError):
    """
    Used to indicate that we do not support this file format.
    """
    pass


def list_files(
    directory: Path,
    file_name: Optional[str] = None,
    file_type: Optional[str] = None,
    join_paths: bool = False,
    max_results: int = -1,
    sub_directories: bool = False,
    path_search_term: Optional[str] = None
) -> List[Path]:
    return_files = []
    if file_type is not None and file_type[0] == '.':
        file_type = file_type[1:]
    join_paths = join_paths or sub_directories

    def process_file(root: Path, name: str) -> bool:
        nonlocal return_files
        root = Path(root)
        type_ok = file_type is None or name.endswith('.' + file_type)
        name_ok = file_name is None or name.startswith(file_name)
        if type_ok and name_ok:
            full_path = root.joinpath(name)
            if path_search_term is not None:
                if not path_search_term in str(full_path):
                    return False
            if join_paths:
                return_files.append(full_path)
            else:
                return_files.append(name)
            if max_results > 0 and len(return_files) >= max_results:
                return True
        return False

    if sub_directories:
        for root, subdirs, files in os.walk(directory):
            for name in files:
                if process_file(root, name):
                    return return_files
    else:
        for name in sorted([f.name for f in Path(directory).iterdir()]):
            if process_file(directory, name):
                return return_files
    return return_files


def search_file(
    base_dir: Path,
    search_term: str,
    file_name: Optional[str] = None,
    file_type: Optional[str] = None
) -> Path:
    try:
        return list_files(
            directory=base_dir,
            file_name=file_name,
            file_type=file_type,
            join_paths=True,
            max_results=1,
            sub_directories=True,
            path_search_term=search_term
        )[0]
    except IndexError as e:
        raise FileNotFoundError(
            f"Failed to search for file: base_dir={base_dir}, search_term={search_term}, file_name={file_name}") from e


def is_pickleable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


# TODO: Remove all references and just use dill
def save_pickle(obj: object, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    exception = None
    warn(
        'This method does not allow pickling of lambda functions. To pickle lambda functions, use the save_dill method instead',
        DeprecationWarning,
        stacklevel=2)
    while True:
        try:
            with open(file_path, 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            break
        except KeyboardInterrupt as e:
            exception = e
    if exception is not None:
        raise exception


def load_pickle(file_path: Path) -> object:
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_dill(obj: object, file_path: Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    exception = None
    while True:
        try:
            with open(file_path, 'wb') as handle:
                dill.dump(obj, handle)
            break
        except KeyboardInterrupt as e:
            exception = e
    if exception is not None:
        raise exception


def load_dill(file_path: Path) -> Optional[object]:
    if file_path.exists():
        with open(file_path, 'rb') as handle:
            return dill.load(handle)
    return None


def slugify(value: Any, allow_unicode: bool = False, replacement: str = '-', title: bool = False) -> str:
    import unicodedata
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    if title:
        value = re.sub(r'[^\w\s-]', '', value.title())
    else:
        value = re.sub(r'[^\w\s-]', '', value.lower())
    result = cast(str, re.sub(r'[-\s]+', replacement, value).strip(replacement + '_'))
    return result


def get_most_recent_file(
    input: Union[Path, Sequence[Path]],
    extension: Optional[str] = None
) -> Path:
    if isinstance(input, Path):
        if extension is None:
            list_of_files = input.glob('*')
        else:
            list_of_files = input.glob(f'*.{extension}')
    else:
        list_of_files = [str(f) for f in input]

    if not list_of_files:
        raise FileNotFoundError(f"No files in directory: '{input}' ({extension=})")

    latest_file = max(list_of_files, key=lambda i: Path(i).stat().st_mtime)
    return latest_file


def get_file_last_modified_timestamp(filepath: Path) -> str:
    ts = filepath.stat().st_mtime
    return get_timestamp(ts)


def get_file_last_modified_datetime(filepath: Path) -> datetime:
    ts = filepath.stat().st_mtime
    return datetime.fromtimestamp(ts)


class SafeJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)
