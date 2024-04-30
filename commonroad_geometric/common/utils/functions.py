import functools
import inspect
import re


def partial_satisfiable(partial_fn: functools.partial) -> bool:
    signature = inspect.signature(partial_fn.func)
    try:
        signature.bind_partial(*partial_fn.args, **partial_fn.keywords)
        return True
    except TypeError:
        return False


def get_function_return_variable_name(fn) -> str:
    variable_name = inspect.getsource(fn).replace(' ', '').rstrip('\n').split()[-1].split('.')[-1]
    variable_name = re.sub(r'\W+', '', variable_name)
    return variable_name
