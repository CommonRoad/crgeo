from collections import defaultdict
from time import sleep
from types import ModuleType
from typing import Any, List, Optional, Set, Type
import ast
import inspect
import json
import logging
import sys
import threading
import warnings

from commonroad_geometric.common.type_checking import is_mutable
from commonroad_geometric.common.utils.string import resolve_string


logger = logging.getLogger(__name__)


FRAMEWORK_NAME = __name__.split('.')[0]


def is_internal_module(
    module: ModuleType,
    base_module: Optional[ModuleType] = None
) -> bool:
    if base_module is None:
        root_name = FRAMEWORK_NAME
    else:
        root_name = base_module.__name__.split('.')[0]
    return module.__name__.split('.')[0] == root_name


def get_stack_objects(only_mutable: bool = False, ignore_modules: bool = False) -> List[Any]:
    stack_objs: List[Any] = sum([list(f[0].f_locals.values()) for f in inspect.stack()], [])

    if only_mutable:
        stack_objs = [o for o in stack_objs if is_mutable(o)]

    if ignore_modules:
        stack_objs = [o for o in stack_objs if not inspect.ismodule(o)]

    return stack_objs


def get_stack_types() -> List[Type]:
    stack_objs = get_stack_objects(only_mutable=True, ignore_modules=True)
    stack_types = [type(obj) for obj in stack_objs]
    return stack_types


def get_submodules(
    module: ModuleType,
    recursive: bool = False,
    depth: int = 0,
    max_depth: int = 3,
    visited: Optional[Set[int]] = None,
    only_internal: bool = False,
    base_module: Optional[ModuleType] = None
) -> Set[ModuleType]:
    if base_module is None:
        base_module = module
    if depth > max_depth:
        return set()

    visited = visited if visited is not None else set()
    visited.add(id(module))

    submodules = set((child for child in module.__dict__.values() if inspect.ismodule(child)))
    if only_internal:
        submodules = set(s for s in submodules if is_internal_module(s, base_module=base_module))

    if recursive:
        submodules_new: Set[ModuleType] = set(s for s in submodules)
        for s in submodules:
            if id(s) in visited:
                continue
            submodules_new.update(get_submodules(
                module=s,
                recursive=True,
                depth=depth + 1,
                visited=visited,
                only_internal=only_internal,
                base_module=base_module
            ))
        submodules = submodules_new
            
    return submodules


def get_stack_modules(recursive: bool = True) -> Set[ModuleType]:
    stack_objs = get_stack_objects(only_mutable=True, ignore_modules=False)
    stack_modules_list = [x if inspect.ismodule(x) else inspect.getmodule(x) for x in stack_objs]
    stack_modules = set((m for m in stack_modules_list if m is not None))

    if recursive:
        submodules = set.union(*(get_submodules(module, recursive=True) for module in stack_modules))
        stack_modules.update(submodules)

    return stack_modules
    
    
def resolve_string_eval(s: str) -> Any:
    modules = get_stack_modules()
    for m in modules:
        locals().update(vars(m))
    try:
        return eval(s)
    except Exception as e:
        return resolve_string(s)


def magic_stack_reassignment(
    search_term: str,
    replace: str,
    skip_external: bool = True,
    replace_all: bool = True
) -> int:
    search_term = search_term.strip()
    replace = resolve_string_eval(replace.strip())

    num_reassignments = 0
    visited: Set[int] = set()

    stack_objs = get_stack_objects(only_mutable=True)

    for obj in stack_objs:
        num_reassignments += _recursive_search_reassignment(
            obj=obj, 
            search_term=search_term,
            replace=replace,
            visited=visited,
            skip_external=skip_external,
            replace_all=replace_all
        )

    return num_reassignments


def _recursive_search_reassignment(
    obj: Any, 
    search_term: str, 
    replace: Any,  
    visited: Set[int],
    recursion_level: int = 0,
    skip_external: bool = True,
    replace_all: bool = True,
    max_depth: int = 10
) -> int:
    if recursion_level > max_depth:
        return 0

    num_reassignments = 0
    if isinstance(obj, list):
        for v in obj:
            if is_mutable(v):
                num_reassignments += _recursive_search_reassignment(
                    obj=v, 
                    search_term=search_term,
                    replace=replace,
                    recursion_level=recursion_level + 1,
                    visited=visited
                )
        visited.add(id(obj))
        return num_reassignments

    if isinstance(obj, dict):
        for v in obj.values():
            if is_mutable(v):
                num_reassignments += _recursive_search_reassignment(
                    obj=v, 
                    search_term=search_term,
                    replace=replace,
                    recursion_level=recursion_level + 1,
                    visited=visited
                )
        visited.add(id(obj))
        return num_reassignments

    try:
        if skip_external and not (hasattr(obj, '__module__') and obj.__module__.startswith('commonroad_geometric')): # TODO
            return 0
    except Exception as e:
        return 0
    if id(obj) in visited:
        return 0

    visited.add(id(obj)) 

    # searching and replacing matching values
    for k, v in obj.__dict__.items():
        if k == search_term:
            setattr(obj, k, replace)
            logger.info(f"Reassigned attribute '{k}' of {repr(obj)} from {v} to {replace}")
            if not replace_all:
                return 1
            num_reassignments += 1

    # processing mutable child attributes recursively
    for k, v in obj.__dict__.items():
        if id(v) not in visited:
            if is_mutable(v):
                num_reassignments_inner = _recursive_search_reassignment(
                    obj=v, 
                    search_term=search_term,
                    replace=replace,
                    recursion_level=recursion_level + 1,
                    visited=visited
                )
                if num_reassignments_inner:
                    if not replace_all:
                        return num_reassignments_inner
                    num_reassignments += num_reassignments_inner

    return num_reassignments