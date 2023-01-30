from typing import Callable, List, Optional, Dict, Any, Sequence, Tuple


def profile(
    func: Callable,
    args: Optional[Sequence] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    print_stdout: bool = True,
    save_to: Optional[str] = 'main.profile',
    disable: bool = False,
    max_location_str_length: int = 80
):

    """Prints profiling summary of function."""
    import io
    import pstats
    import cProfile
    import timeit
    import pandas as pd
    import os
    from pstats import SortKey

    kwargs = kwargs if kwargs is not None else {}
    args = args if args is not None else []
    if disable:
        return func(*args, **kwargs)
    pr = cProfile.Profile(builtins=False)
    pr.enable()
    start = timeit.default_timer()
    catched_exception = None
    try:
        func(*args, **kwargs)
    except (Exception, KeyboardInterrupt) as e:
        catched_exception = e
    stop = timeit.default_timer()
    elapsed_time = stop - start
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CALLS
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby) #.strip_dirs()
    ps.print_stats()

    if save_to is not None:
        path = os.path.join('profiles', save_to)
        if not os.path.isdir('profiles'):
            os.makedirs('profiles')
        ps.dump_stats(path)

    # Parse the stdout text and split it into a table
    data = []
    started = False
    for l in s.getvalue().split("\n"):
        if not started:
            if l == "   ncalls  tottime  percall  cumtime  percall filename:lineno(function)":
                started = True
                data.append(l)
        else:
            data.append(l)
    content: List[Tuple[Any, ...]] = []
    for i, l in enumerate(data):
        fs = l.find(" ", 8)
        if i == 0:
            content.append(tuple(
                [l[0:fs], l[fs:fs+9], l[fs+9:fs+18], l[fs+18:fs+27], l[fs+27:fs+36], l[fs+36:]]))
        else:
            try:
                content.append(tuple([int(l[0:fs].split('/')[0]), float(l[fs:fs+9]), float(
                    l[fs+9:fs+18]), float(l[fs+18:fs+27]), float(l[fs+27:fs+36]), l[fs+36:]]))
            except ValueError:
                pass
    prof_df = pd.DataFrame(
        content[1:], columns=list(map(str.strip, content[0])))
    prof_df.insert(0, 'reltime', prof_df['tottime']/elapsed_time)
    prof_df = prof_df[prof_df['reltime'] > 0.001]
    prof_df = prof_df.sort_values(by=['reltime'], ascending=False)
    #prof_df['reltime'] = (prof_df['reltime']*100).astype(str) + '%'
    #prof_df = prof_df[prof_df['ncalls'] > 1000]

    if print_stdout:
        print_df = prof_df.copy()
        print_df['reltime'] = pd.Series(["{0:.2f}%".format(
            val * 100) for val in prof_df['reltime']], index=prof_df.index)
        location_str = print_df['filename:lineno(function)'].str.slice(-max_location_str_length)
        print_df['filename:lineno(function)'] = location_str
        print(print_df.to_string())

    if catched_exception is not None:
        raise catched_exception

    return prof_df


def profile_decorator(func):
    def _decorator(self, *args, **kwargs):
        return profile(func, args=args, kwargs=kwargs)
    return _decorator