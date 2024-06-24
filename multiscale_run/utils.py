import ast
import contextlib
import functools
import inspect
import json
import logging
import math
import operator
import os
import pickle
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psutil
import simpleeval
from scipy import sparse


@functools.cache
def mpi():
    """Returns
    The `mpi4py.MPI` module
    """
    from mpi4py import MPI

    return MPI


@functools.cache
def comm():
    """Returns
    The MPI communicator (mpi4py.MPI.Comm)
    """
    return mpi().COMM_WORLD


@functools.cache
def rank():
    """Returns
    MPI rank of the current process (int)
    """
    return comm().Get_rank()


@functools.cache
def rank0():
    """Returns
    True if the current process has MPI rank 0, False otherwise
    """
    return rank() == 0


@functools.cache
def size():
    """Returns
    The size of the MPI communicator (int)
    """
    return comm().Get_size()


class MsrException(Exception):
    """Custom exception class"""


def timesteps(end: float, step: float):
    """Timestep generator

    Includes the end point

    Example::

       >>> timesteps(1, 0.2)
       [0.2, 0.4, ..., 1.0]
    """
    l = np.arange(*[step, end + step, step])
    if len(l) and l[-1] > end:
        l = l[:-1]
    return l


def print_once(*args, **kwargs):
    """Print only once among ranks"""
    if rank0():
        print(*args, *kwargs)


def describe_obj(v, affix: str = ""):
    """Inspect the structure and statistics of a variable and its contents.

    This function provides a detailed view of the variable and its subcomponents, including lists,
    dictionaries, and NumPy arrays, along with their statistics (mean, min, max).

    Args:
        v: The variable to inspect.
        affix: A prefix to add to the printed output for formatting.

    Example::

        inspect(my_data, "  ")
    """
    skip = "  "

    def _base(v, affix):
        rank_print(f"{affix}{v}")

    def _list(v, affix):
        s = f"{affix}list ({len(v)}): "
        if len(v) == 0:
            rank_print(s)
            return

        if type(v[0]) in [int, float]:
            ss = [f"mean: {np.mean(v)}", f"min: {np.min(v)}", f"max: {np.max(v)}"]
            rank_print(s + ", ".join(ss))
            return

        rank_print(s)
        for idx, i in enumerate(v):
            rank_print(f"{affix}{idx}:")
            describe_obj(i, affix + skip)

    def _dict(v, affix):
        s = f"{affix}dict ({len(v)}): "
        rank_print(s)
        if len(v) == 0:
            return

        for k, v in v.items():
            rank_print(f"{affix}{k}:")
            describe_obj(v, f"{affix}{skip}")

    def _nparray(v, affix):
        if len(v.shape) == 1:
            _list(v.tolist(), affix)
            return
        rank_print(f"{affix}nparray {v.shape}")

    s = {list: _list, dict: _dict, np.ndarray: _nparray}
    s.get(type(v), _base)(v, affix)


def delete_rows(mat, indices):
    """Remove the rows denoted by ``indices`` from the matrix ``mat``.

    Args:
        mat: The matrix (NumPy array or CSR sparse matrix).
        indices: The indices of rows to be deleted.

    Returns:
        The modified matrix with specified rows removed.

    Raises:
        ValueError: If the input matrix is not a NumPy array or CSR sparse matrix.

    Example::

        new_matrix = delete_rows(matrix_or_sparse, [2, 4, 6])
    """
    if not isinstance(indices, list):
        indices = list(indices)

    if isinstance(mat, np.ndarray):
        return np.delete(mat, indices, axis=0)
    if isinstance(mat, sparse.csr_matrix):
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[indices] = False
        return mat[mask]

    raise ValueError(
        f"Input matrix must be a NumPy array or CSR sparse matrix. Current type: {type(mat)}"
    )


def delete_cols(mat, indices):
    """Remove the columns denoted by ``indices`` from the matrix ``mat``.

    Args:
        mat: The matrix (NumPy array or CSR sparse matrix).
        indices: The indices of columns to be deleted.

    Returns:
        The modified matrix with specified columns removed.

    Raises:
        ValueError: If the input matrix is not a NumPy array or CSR sparse matrix.

    Example::

        new_matrix = delete_cols(matrix_or_sparse, [2, 4, 6])
    """
    if not isinstance(indices, list):
        indices = list(indices)

    if isinstance(mat, np.ndarray):
        return np.delete(mat, indices, axis=1)
    if isinstance(mat, sparse.csr_matrix):
        all_cols = np.arange(mat.shape[1])
        cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, indices)))[0]
        return mat[:, cols_to_keep]

    raise ValueError(
        f"Input matrix must be a NumPy array or CSR sparse matrix. Current type: {type(mat)}"
    )


def rank_print(*args, **kwargs):
    """Print with rank information.

    Args:
        *args: Variable length positional arguments for print.
        **kwargs: Variable length keyword arguments for print.

    Note:
        This function appends the rank of the process to the output.

    Example::

        rank_print("Hello, World!")
    """
    print(f"rank {rank()}:", *args, **kwargs, flush=True)


def cache_decorator(
    field_names,
    path=None,
    is_save: bool = None,
    is_load: bool = None,
    only_rank0: bool = False,
):
    """Caching system for parts of a class.

    This decorator must be applied to a function that adds at least 1 field to a class.
    The specified field is cached in memory.

    The function should not return any values at the moment.

    Args:
        field_names (str or list of str): The name(s) of the field(s) to be cached.
        path (str or Path, optional): The path to the cache directory. Defaults to None.
        is_save: If True, data is saved to the cache. Defaults to None.
        is_load: If True, data is loaded from the cache. Defaults to None.
        only_rank0: If True, cache is only used on rank 0. Defaults to False.

    Returns:
        function: The wrapped method.

    Note:
        This decorator facilitates the caching of data in memory.

    Example::

        @cache_decorator("my_field", path="/cache", is_save=True, is_load=True)
        def my_method(self, *args, **kwargs):
            # Your method implementation here
    """
    if isinstance(field_names, str):
        field_names = [field_names]

    file_names = (
        field_names if only_rank0 else [f"{i}_rank{rank()}" for i in field_names]
    )

    def decorator_add_field_method(method):
        @functools.wraps(method)
        def wrapper(self, *args, path=path, is_save=is_save, is_load=is_load, **kwargs):
            if hasattr(self, "config"):
                path = self.config.multiscale_run.cache_path
                is_save = self.config.multiscale_run.cache_save
                is_load = self.config.multiscale_run.cache_load

            path = Path(path)
            if not only_rank0:
                path = path / f"n{size()}"

            np.testing.assert_equal(len(field_names), len(file_names))
            np.testing.assert_array_less([0], [len(field_names)])

            fn_npz = [
                (path / i).with_suffix(".npz")
                for i in file_names
                if (path / i).with_suffix(".npz").is_file()
            ]
            fn_pickle = [
                (path / i).with_suffix(".pickle")
                for i in file_names
                if (path / i).with_suffix(".pickle").is_file()
            ]

            fn = [*fn_npz, *fn_pickle]

            if len(fn) > len(file_names):
                raise FileNotFoundError(
                    "some files appear as pickle and npz, it is ambiguous"
                )

            all_files_are_present = len(fn) == len(file_names)

            if is_load and all_files_are_present:
                for field, full_path in zip(field_names, fn):
                    if not (rank0() or "rank" in str(full_path)):
                        setattr(self, field, None)
                        continue

                    logging.info(f"load {field} from {full_path}")
                    if full_path.suffix == ".npz":
                        obj = sparse.load_npz(full_path)
                    else:
                        with open(full_path, "rb") as f:
                            obj = pickle.load(f)
                    setattr(self, field, obj)

                return

            logging.info(f"no cache found. Compute {str(field_names)}")
            method(self, *args, **kwargs)
            if is_save:
                path.mkdir(parents=True, exist_ok=True)
                for field, file in zip(field_names, file_names):
                    full_path = path / file
                    if rank0() or "rank" in file:
                        logging.info(f"save {field} to {full_path}")
                        obj = getattr(self, field)
                        if isinstance(obj, sparse.csr_matrix):
                            sparse.save_npz(full_path.with_suffix(".npz"), obj)
                        else:
                            with open(full_path.with_suffix(".pickle"), "wb") as f:
                                pickle.dump(obj, f)

        return wrapper

    return decorator_add_field_method


def append_suffix(path: Path, suffix: str):
    """Append to path a suffix respecting file extensions"""
    path = Path(path)
    stem = path.stem
    extension = path.suffix
    new_stem = stem + "_" + suffix
    # Reconstruct the Path with modified file name
    return path.with_name(new_stem).with_suffix(extension)


def remove_elems(v: list, to_be_removed: set) -> list:
    """Convenience function: removes elements from a list based on their indices.

    Args:
        v: The list from which elements will be removed.
        to_be_removed: an iterable containing the indices of elements to be removed.

    Returns:
        A list with elements removed based on the provided indices.
    """
    return [i for idx, i in enumerate(v) if idx not in to_be_removed]


def logs_decorator(wrapped):
    """Decorator for logging function execution details.

    This decorator logs the start and end of a function's execution, its memory usage, and elapsed time.

    Args:
        wrapped (callable): The function to be wrapped by the decorator.

    Returns:
        callable: The wrapped function.

    Example::

        @logs_decorator
        def my_function(arg1, arg2):
            # Your function implementation here
            return result
    """

    @functools.wraps(wrapped)
    def logs(*args, **kwargs):
        function_name = wrapped.__name__
        logging.info(f"   {function_name}")
        start = time.perf_counter()
        res = wrapped(*args, **kwargs)
        mem = psutil.Process().memory_info().rss / 1024**2
        stop = time.perf_counter()
        logging.info(f"   /{function_name}: mem: {mem}, time: {stop - start}")
        return res

    return logs


def ppf(n):
    """Pretty Print of float

    Args:
        n (float): float

    Returns:
        str: str
    """
    return f"{n:.3}"


def merge_dicts(parent: dict, child: dict):
    """Merge dictionaries recursively (in case of nested dicts) giving priority to child over parent
    for ties. Values of matching keys must match or a TypeError is raised.

    Args:
        parent: parent dict
        child: child dict (priority)

    Returns:
        dict: merged dict following the rules listed before

    Example::

        >>> parent = {"A":1, "B":{"a":1, "b":2}, "C": 2}
        >>> child = {"A":2, "B":{"a":2, "c":3}, "D": 3}
        >>> merge_dicts(parent, child)
        {"A":2, "B":{"a":2, "b":2, "c":3}, "C": 2, "D": 3}
    """

    def merge_vals(k, parent: dict, child: dict):
        """Merging logic.

        Args:
            k (key type): the key can be in either parent, child or both.
            parent: parent dict.
            child: child dict (priority).

        Raises:
            TypeError: in case the key is present in both parent and child and the type missmatches.

        Returns:
            value type: merged version of the values possibly found in child and/or parent.
        """
        if k not in parent:
            return child[k]
        if k not in child:
            return parent[k]
        if type(parent[k]) is not type(child[k]):
            raise TypeError(
                f"Field type missmatch for the values of key {k}: {parent[k]} ({type(parent[k])}) != {child[k]} ({type(child[k])})"
            )
        if isinstance(parent[k], dict):
            return merge_dicts(parent[k], child[k])
        return child[k]

    return {k: merge_vals(k, parent, child) for k in set(parent) | set(child)}


def replace_values(obj: dict | list | str, replacements: dict):
    """Recursively replaces values in a dictionary, list, or string based on a replacement dictionary.

    Args:
        obj: object that needs the replacements
        replacements: dictionary of string replacements

    Returns:
        the modified object
    """
    if isinstance(obj, str):
        return replacements.get(obj, obj)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            obj[index] = replace_values(item, replacements)
        return obj
    elif isinstance(obj, dict):
        obj.update((k, replace_values(v, replacements)) for k, v in obj.items())
        return obj
    else:
        return obj  # In case obj is neither str, list, nor dict, return it as is


def get_dict_from_json(path) -> dict:
    """Convenience function to load json files.

    Args:
        path (Path or str): path that should be extracted.

    Returns:
        dict from the json
    """
    logging.info(f"reading: {str(path)}")
    with open(str(path), "r") as json_file:
        return json.load(json_file)


def heavy_duty_MPI_Gather(v: np.ndarray, root=0):
    """Optimized MPI gather wrapper for very big matrices and vectors

    In particular, MPI fails when the final vector is longer than an INT32.
    Here we avoid this problem without sacrificing performance by sending
    one custom object per rank.

    Args:
        np.ndarray: it can be a 1 or 2D array of ints or floats
        root (int, optional): MPI root

    Returns:
        np.ndarray: 1 or 2D array of ints or floats
    """
    dt = v.dtype

    # get the correct datatype for Create_contiguous

    T = mpi()._typedict[dt.char].Create_contiguous(v.size)
    T.Commit()
    ans = None
    if rank() == root:
        ans = np.zeros((size(), *v.shape), dtype=dt)

    comm().Gather(sendbuf=[v, 1, T], recvbuf=ans, root=root)

    T.Free()
    return ans


def stats(v):
    """Return some useful object stats if appropriate (used for debugging)

    Args:
        v (any): any object

    Returns:
        str: some useful stats (if appropriate)
    """
    t = type(v)
    if not isinstance(v, Iterable):
        v = [v]

    v = np.array(v)

    min0 = min(v) if len(v) else 0
    max0 = max(v) if len(v) else 0
    return f"stats: type={t}, shape={v.shape}, min={min0}, max={max0}"


def remove_path(path):
    """Remove a directory at the specified path (rank 0 only).

    Args:
        path (str or Path): The path to the directory to be removed.

    Note:
        This function is intended for use on rank 0 in a parallel or distributed computing context.
        It attempts to remove the specified directory and ignores `FileNotFoundError`
        if the directory does not exist.

    Example::

        remove_path("/path/to/directory")
    """
    if rank0():
        try:
            shutil.rmtree(path)
        except NotADirectoryError:
            os.remove(path)
        except FileNotFoundError:
            pass
    comm().Barrier()


def get_subs_d(d: dict) -> dict:
    """Recursively extracts and filters string key-value pairs from a nested dictionary.

    This function traverses the input dictionary recursively, retaining only the key-value pairs
    where both the key and the value are strings. It returns a new dictionary with these filtered pairs.

    Args:
    - d: The input dictionary to process.

    Returns:
    dict: A new dictionary containing only string key-value pairs.
    """
    ans = {k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str)}
    for k, v in d.items():
        if isinstance(k, str) and isinstance(v, dict):
            ans.update(get_subs_d(v))
    return ans


def get_resolved_value(d: dict, key: str, in_place: bool = False):
    """Get the value of a key, replacing ${token} placeholders with corresponding values in the same dictionary.

    This function retrieves the value associated with the specified key in the input dictionary (d),
    and recursively resolves ${token} placeholders in the value using other key-value pairs in the same dictionary.

    Args:
      d: The input dictionary containing key-value pairs.
      key: The key whose value needs to be retrieved and resolved.
      in_place: If True, performs in-place substitution of values in the input dictionary. Defaults to False.

    Returns:
      str: The resolved value associated with the specified key.
    """
    v = d[key]
    tokens = set(re.findall(r"\${(.*?)}", v))
    if not len(tokens):
        return v
    for token in tokens:
        v = v.replace(f"${{{token}}}", get_resolved_value(d, token, in_place))
    if in_place:
        d[key] = v
    return v


def resolve_replaces(d: dict, base_subs_d: dict = None) -> None:
    """Resolve ${token} placeholders in string values of a nested dictionary, using specified substitution values.

    This function processes a nested dictionary (d) and applies token substitution to string values.
    It first extracts and filters string key-value pairs from the dictionary, then resolves ${token}
    placeholders in those values using a combination of the original dictionary and additional base substitution values.

    Args:
        d: The input nested dictionary to process.
        base_subs_d: Additional base substitution values. Defaults to an empty dictionary.

    Returns:
        None: The function performs in-place substitution on the input dictionary (d).
    """
    if base_subs_d is None:
        base_subs_d = {}

    subs_d = get_subs_d(d)
    subs_d.update(base_subs_d)
    for k in subs_d.keys():
        get_resolved_value(subs_d, k, True)

    def _rep(obj, subs_d):
        if isinstance(obj, str):
            tokens = set(re.findall(r"\${(.*?)}", obj))
            for token in tokens:
                obj = obj.replace(f"${{{token}}}", subs_d[token])

        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                obj[idx] = _rep(item, subs_d)

        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = _rep(v, subs_d)

        return obj

    d = _rep(d, subs_d)


def bbox(pts: np.ndarray):
    """Calculate the bounding box of a set of 3D points.

    Args:
      pts: An array of 3D points with shape (n, 3).

    Returns:
        np.ndarray: An array containing the minimum and maximum coordinates of the bounding box.
            The first element is the minimum coordinates, and the second element is the maximum coordinates.

    Example::

       pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
       bbox(pts) # returns an array with the minimum and maximum coordinates of the bounding box.

    """
    return np.array([np.min(pts, axis=0), np.max(pts, axis=0)])


def generate_cube_corners(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Generate an array of n cube corner points between two given points a and b.

    Args:
        a: The lower corner of the cube.
        b: The upper corner of the cube.
        n: The number of corner points to generate.

    Returns:
      np.ndarray: An array of n corner points.

    Example::

        generate_cube_corners([1, 1, 1], [2, 3, 3], 8)
        # returns an array of 8 corner points within the specified cube.

    """
    ab = [a, b]
    ans = np.array(
        [
            np.array([ab[i % 2][0], ab[(i // 2) % 2][1], ab[(i // 4) % 2][2]])
            for i in range(n)
        ]
    )
    return ans


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def json_sanitize(obj):
    """Convenience function to convert recursively posix paths in a dict to make it json-able"""
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, list):
        return [json_sanitize(i) for i in obj]

    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}

    return obj


def get_var_name(lvl: int = 1, pos: int = 0) -> str:
    """Get the name of a variable from the caller's scope.

    Args:
        lvl: The number of levels up in the call stack to look for the variable name (default: 1).
        pos: The position of the variable in the calling function's argument list (default: 0).

    Returns:
        The name of the variable.

    """
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[lvl + 1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find("(") + 1 : -1].split(",")
    return str(args[pos]).strip()


def check_value(
    v: float,
    lb: float = -float("inf"),
    hb: float = float("inf"),
    leb: float = -float("inf"),
    heb: float = float("inf"),
    err: Exception = MsrException,
    msg: str = None,
):
    """Check if a value is within specified bounds and raising an exception if it's not.

    Args:
        v: The value to be checked.
        lb: The lower bound (default: negative infinity).
        hb: The upper bound (default: positive infinity).
        leb: The lower or equal bound (default: negative infinity).
        heb: The higher or equal bound (default: positive infinity).
        err: The exception class to be raised (default: MsrException).
        msg: Error affix. For when deduction fails.

    Raise:
        MsrException: If the value is None, not floatable, NaN, or outside the specified bounds. Custom exception otherwise.

    """
    msg = msg if msg is not None else f"{get_var_name(1, 0)}"

    if v is None:
        raise MsrException(msg + f" ({v}) is None")
    try:
        float(v)
    except ValueError:
        raise MsrException(msg + f" ({v}) is not floatable")
    if np.isnan(v):
        raise MsrException(msg + f" ({v}) is NaN")
    if np.isinf(v):
        raise MsrException(msg + f" ({v}) is Inf")

    if v < lb:
        raise err(f"{msg} ({v}) < {lb}")
    if v > hb:
        raise err(f"{msg} ({v}) > {hb}")
    if v <= leb:
        raise err(f"{msg} ({v}) <= {leb}")
    if v >= heb:
        raise err(f"{msg} ({v}) >= {heb}")


@contextlib.contextmanager
def pushd(path):
    """Change the current working directory within the scope of a Python `with` statement

    Args:
      path: the directory to jump into
    """
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(cwd)


def pushtempd(wrapped):
    """Callable decorator changing the current working directory to a
    temporary directory during the execution of the wrapped function.

    Args:
        wrapped (callable): the function wrapped by the decorator.

     Returns:
        callable: The wrapped function.
    """

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        if rank0():
            tempdir = tempfile.mkdtemp(prefix="msr-tests")
        else:
            tempdir = None
        tempdir = comm().bcast(tempdir, root=0)
        with pushd(tempdir):
            return wrapped(*args, **kwargs)

    return _wrap


def log_stats(
    vec: list[float],
    lb: float = -float("inf"),
    hb: float = float("inf"),
    leb: float = -float("inf"),
    heb: float = float("inf"),
    msg: str = "",
):
    min0, avg, max0, n = float("inf"), 0, -float("inf"), len(vec)
    if n:
        min0, avg, max0 = np.min(vec), np.mean(vec), np.max(vec)
    min0 = comm().gather(min0, root=0)
    avg = comm().gather(avg, root=0)
    max0 = comm().gather(max0, root=0)
    n = comm().gather(n, root=0)
    if rank0():
        min0 = np.min(min0)
        max0 = np.max(max0)
        avg = np.average(avg, weights=n)
        low_b = f"[{lb}" if lb > leb else f"({leb}"
        high_b = f"{hb}]" if hb < heb else f"{heb})"

        logging.info(
            f"{msg} (low_b/min/avg/max/high_b): {low_b}/{ppf(min0)}/{ppf(avg)}/{ppf(max0)}/{high_b}"
        )


class PyExprEval:
    """Parse and evaluate Python expressions in a controlled environment"""

    def __init__(self, config: dict):
        """
        Args:
          config: MultiscaleRun config. Can be a dict or an instance of `multiscale_run.MsrConfig`
        """
        operators = simpleeval.DEFAULT_OPERATORS.copy()
        operators.update(
            {
                ast.Add: operator.add,
                ast.Mult: operator.mul,
            }
        )
        # list of variables and modules available in the scope
        names = dict(
            config=config,
            math=math,
            np=np,
        )
        # list of builtins available in the scope
        functions = dict(
            abs=abs,
            min=min,
            max=max,
            pow=pow,
            round=round,
            sum=sum,
        )
        self._se = simpleeval.SimpleEval(
            functions=functions, names=names, operators=operators
        )

    def __call__(self, expr: str, **names):
        """Evaluate a restricted Python expression made of basic operations on builtin Python types and NumPy arrays.

        Additional symbols available:

        - `config`: the instance given in the constructor
        - `np` and `math`: the NumPy and standard `math` modules
        - computational Python builtins: abs, min, max, pow, round and sum

        Args:
          expr: the Python expression to evaluate
          names: additional symbols available in the expression

        Example::

          >>> config = MsrConfig._from_dict({"factor": 2})
          >>> pyeval = PyExprEval(config)
          >>> pyeval("data * config.factor + 1", data=np.array([1, 2, 3])))
          array([3, 5, 7])

        Note::

            This method is not reentrant since it modifies the state of `self._se` attribute.
        """
        tree = self._get_tree(expr)
        self.simple_eval.names.update(names)
        try:
            return self.simple_eval.eval(expr, previously_parsed=tree)
        except simpleeval.InvalidExpression as e:
            raise MsrException(
                f"Could not evaluate Python conversion expression: '{expr}'"
            ) from e

    @property
    def simple_eval(self):
        """Get internal instance of `simpleeval.SimpleEval`"""
        return self._se

    @functools.lru_cache(maxsize=None)
    def _get_tree(self, expr: str):
        """Get parsing result of a Python expression

        Args:
          expr: the Python expression to parse

        Note::

          The parsing is cached under the hood for efficiency.
        """
        try:
            return self.simple_eval.parse(expr)
        except simpleeval.InvalidExpression as e:
            raise MsrException(
                f"Could not parse Python conversion expression: '{expr}'"
            ) from e
