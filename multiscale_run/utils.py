import json
import logging
import os
import pickle
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psutil
from scipy import sparse

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrException(Exception):
    """Custom exception class"""

    pass


def timesteps(end: float, step: float):
    """Timestep generator

    Ex: timesteps(1, 0.2) -> [0.2, 0.4, ..., 1.0]
    """
    return ((i + 1) * step for i in range(int(end / step)))


def print_once(*args, **kwargs):
    """Print only once among ranks"""
    if rank == 0:
        print(*args, *kwargs)


def inspect(v, affix=""):
    """Inspect the structure and statistics of a variable and its contents.

    This function provides a detailed view of the variable and its subcomponents, including lists, dictionaries, and NumPy arrays, along with their statistics (mean, min, max).

    Parameters:
        v (any): The variable to inspect.
        affix (str): A prefix to add to the printed output for formatting.

    Example:
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
            inspect(i, affix + skip)

    def _dict(v, affix):
        s = f"{affix}dict ({len(v)}): "
        rank_print(s)
        if len(v) == 0:
            return

        for k, v in v.items():
            rank_print(f"{affix}{k}:")
            inspect(v, f"{affix}{skip}")

    def _nparray(v, affix):
        if len(v.shape) == 1:
            _list(v.tolist(), affix)
            return
        rank_print(f"{affix}nparray {v.shape}")

    s = {list: _list, dict: _dict, np.ndarray: _nparray}
    s.get(type(v), _base)(v, affix)


def delete_rows_csr(mat, indices):
    """Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.

    Parameters:
        mat (scipy.sparse.csr_matrix): The CSR sparse matrix.
        indices (iterable): The indices of rows to be deleted.

    Returns:
        scipy.sparse.csr_matrix: The modified CSR matrix with specified rows removed.

    Raises:
        ValueError: If the input matrix is not in CSR format.

    Example:
        new_csr_matrix = delete_rows_csr(sparse_csr_matrix, [2, 4, 6])
    """

    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError(
            f"works only for CSR format -- use .tocsr() first. Current type: {type(mat)}"
        )
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def delete_cols_csr(mat, indices):
    """Remove the columns denoted by ``indices`` from the CSR sparse matrix ``mat``.

    Parameters:
        mat (scipy.sparse.csr_matrix): The CSR sparse matrix.
        indices (iterable): The indices of columns to be deleted.

    Returns:
        scipy.sparse.csr_matrix: The modified CSR matrix with specified columns removed.

    Raises:
        ValueError: If the input matrix is not in CSR format.

    Example:
        new_csr_matrix = delete_cols_csr(sparse_csr_matrix, [2, 4, 6])
    """

    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError(
            f"works only for CSR format -- use .tocsr() first. Current type: {type(mat)}"
        )

    all_cols = np.arange(mat.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, indices)))[0]
    return mat[:, cols_to_keep]


def rank_print(*args, **kwargs):
    """Print with rank information.

    Parameters:
        *args: Variable length positional arguments for print.
        **kwargs: Variable length keyword arguments for print.

    Note:
        This function appends the rank of the process to the output.

    Example:
        rank_print("Hello, World!")
    """
    print(f"rank {comm.Get_rank()}:", *args, **kwargs, flush=True)


def cache_decorator(
    field_names, path=None, is_save=None, is_load=None, only_rank0=False
):
    """Caching system for parts of a class.

    This decorator must be applied to a function that adds at least 1 field to a class. The specified field is cached in memory.

    The function should not return any values at the moment.

    Parameters:
        field_names (str or list of str): The name(s) of the field(s) to be cached.
        path (str or Path, optional): The path to the cache directory. Defaults to None.
        is_save (bool, optional): If True, data is saved to the cache. Defaults to None.
        is_load (bool, optional): If True, data is loaded from the cache. Defaults to None.
        only_rank0 (bool, optional): If True, cache is only used on rank 0. Defaults to False.

    Returns:
        function: The wrapped method.

    Note:
        This decorator facilitates the caching of data in memory.

    Example:
        @cache_decorator("my_field", path="/cache", is_save=True, is_load=True)
        def my_method(self, *args, **kwargs):
            # Your method implementation here
    """

    if isinstance(field_names, str):
        field_names = [field_names]

    file_names = field_names if only_rank0 else [f"{i}_rank{rank}" for i in field_names]

    def decorator_add_field_method(method):
        def wrapper(self, *args, path=path, is_save=is_save, is_load=is_load, **kwargs):
            if hasattr(self, "config"):
                path = self.config.cache_path
                is_save = self.config.cache_save
                is_load = self.config.cache_load

            path = Path(path)
            if not only_rank0:
                path = path / f"n{size}"

            np.testing.assert_equal(len(field_names), len(file_names))
            np.testing.assert_array_less([0], [len(field_names)])

            fn_pnz = [
                (path / i).with_suffix(".npz")
                for i in file_names
                if (path / i).with_suffix(".npz").is_file()
            ]
            fn_pickle = [
                (path / i).with_suffix(".pickle")
                for i in file_names
                if (path / i).with_suffix(".pickle").is_file()
            ]

            fn = [*fn_pnz, *fn_pickle]

            if len(fn) > len(file_names):
                raise FileNotFoundError(
                    "some files appear as pikle and npz, it is ambiguous"
                )

            all_files_are_present = len(fn) == len(file_names)

            if is_load and all_files_are_present:
                for field, full_path in zip(field_names, fn):
                    if not (rank == 0 or "rank" in str(full_path)):
                        setattr(self, field, None)
                        continue

                    logging.info(f"load {field} from {full_path}")
                    try:
                        obj = sparse.load_npz(full_path)
                    except:
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
                    if rank == 0 or "rank" in file:
                        logging.info(f"save {field} to {full_path}")
                        obj = getattr(self, field)
                        if isinstance(obj, sparse.csr_matrix):
                            sparse.save_npz(full_path.with_suffix(".npz"), obj)
                        else:
                            with open(full_path.with_suffix(".pickle"), "wb") as f:
                                pickle.dump(obj, f)

        return wrapper

    return decorator_add_field_method


def clear_and_replace_files_decorator(paths):
    """
    A decorator that clears and replaces specified files before and after
    calling the decorated function.

    Args:
        paths (str or List[str]): A file path or a list of file paths to be cleared
            and replaced.

    Returns:
        Callable: A decorator that can be applied to functions.
    """
    logging.info("clear_and_replace_files_decorator")
    if not isinstance(paths, list):
        paths = [paths]
    paths = [Path(i) for i in paths]

    def remove_and_replace_path(path, to_tmp):
        """
        Remove and replace a file or path.

        Args:
            path (str or Path): The path to the file or directory.
            to_tmp (bool): True if replacing with a temporary file, False if restoring.
        """
        to_path = path.with_name(path.name + "_tmp")
        from_path = path.with_name(path.name)
        if not to_tmp:
            to_path, from_path = from_path, to_path

        rename_path(from_path, to_path)

    def decor(method):
        """
        The decorator function that wraps the original method.

        Args:
            method (callable): The function to be decorated.

        Returns:
            Callable: The decorated function.
        """

        def wrapper(*args, **kwargs):
            for path in paths:
                remove_and_replace_path(path, True)

            method(*args, **kwargs)

            for path in paths:
                remove_and_replace_path(path, False)

        return wrapper

    return decor


def logs_decorator(foo):
    """Decorator for logging function execution details.

    This decorator logs the start and end of a function's execution, its memory usage, and elapsed time.

    Parameters:
        foo (function): The function to be wrapped by the decorator.

    Returns:
        function: The wrapped function.

    Example:
        @logs_decorator
        def my_function(arg1, arg2):
            # Your function implementation here
            return result
    """

    def logs(*args, **kwargs):
        function_name = foo.__name__
        logging.info(f"   {function_name}")
        start = time.perf_counter()
        res = foo(*args, **kwargs)
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


def merge_dicts(parent, child):
    """Merge dictionaries recursively (in case of nested dicts) giving priority to child over parent
    for ties. Values of matching keys must match or a TypeError is raised.

    Example:
        parent: {"A":1, "B":{"a":1, "b":2}, "C": 2}
        child: {"A":2, "B":{"a":2, "c":3}, "D": 3}

        ans: {"A":2, "B":{"a":2, "b":2, "c":3}, "C": 2, "D": 3}

    Args:
        parent (dict): parent dict
        child (dict): child dict (priority)


    Returns:
        dict: merged dict following the rules listed before
    """

    def merge_vals(k, parent, child):
        """Merging logic

        Args:
            k (key type): the key can be in either parent, child or both
            parent (dict): parent dict
            child (dict): child dict (priority)

        Raises:
            TypeError: in case the key is present in both parent and child and the type missmatches

        Returns:
            value type: merged version of the values possibly found in child and/or parent
        """

        if k not in parent:
            return child[k]
        if k not in child:
            return parent[k]
        if type(parent[k]) != type(child[k]):
            raise TypeError(
                f"Field type missmatch for the values of key {k}: {parent[k]} ({type(parent[k])}) != {child[k]} ({type(child[k])})"
            )
        if isinstance(parent[k], dict):
            return merge_dicts(parent[k], child[k])
        return child[k]

    return {k: merge_vals(k, parent, child) for k in set(parent) | set(child)}


def load_from_env(env_name, default, t):
    """Either load from environment or set with default (env takes priority)"""
    var = os.getenv(env_name)

    return t(var) if var is not None else default


def get_dict_from_json(p):
    """Convenience function to load json files

    Args:
        path (Path or str): path that should be estracted

    Returns:
        dict
    """
    with open(str(p), "r") as json_file:
        logging.info(f"reading: {str(p)}")
        return json.load(json_file)


def heavy_duty_MPI_Gather(v: np.ndarray, root=0):
    """Optimized MPI gather wrapper for very big matrices and vectors

    In particular, MPI fails when the final vector is longer than an INT32.
    Here we avoid this problem without sacrificing performance by sending
    one custom object per rank.

    Args:
        v (np.ndarray): it can be a 1 or 2D array of ints or floats
        root (int, optional): MPI root

    Returns:
        np.ndarray: 1 or 2D array of ints or floats
    """
    dt = v.dtype

    # get the correct datatype for Create_contiguous
    T = MPI4PY._typedict[dt.char].Create_contiguous(v.size)
    T.Commit()
    ans = None
    if rank == root:
        ans = np.zeros((size, *v.shape), dtype=dt)

    comm.Gather(sendbuf=[v, 1, T], recvbuf=ans, root=root)

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

    l = v.shape
    min0 = min(v) if len(v) else 0
    max0 = max(v) if len(v) else 0
    return f"stats: type={t}, shape={l}, min={min0}, max={max0}"


def remove_path(path):
    """Remove a directory at the specified path (rank 0 only).

    Parameters:
        path (str or Path): The path to the directory to be removed.

    Note:
        This function is intended for use on rank 0 in a parallel or distributed computing context. It attempts to remove the specified directory and ignores `FileNotFoundError` if the directory does not exist.

    Example:
        remove_path("/path/to/directory")
    """

    if rank == 0:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
    comm.Barrier()


def rename_path(path, new_path):
    """
    Rename a file or directory to a new path.

    This function renames a file or directory pointed to by the 'path' argument to the 'new_path' argument. It also logs
    the renaming process.

    Args:
        path (str or Path): The original file or directory path.
        new_path (str or Path): The new file or directory path to rename to.
    """
    remove_path(new_path)
    if rank == 0:
        if path.exists():
            logging.info(f"renaming {path} to {new_path}")
            path.rename(new_path)
        else:
            logging.info(f"{path} does not exist. Nothing to rename")
    comm.Barrier()


def recursive_replace(d: dict, rd: dict) -> None:
    """
    Recursively replaces placeholders in a dictionary with values.

    This function iterates through a dictionary `d` and replaces placeholders
    in string values with corresponding values from the dictionary `rd`. Placeholders
    are represented as `${placeholder_name}` and are replaced by the corresponding
    values from `rd`. If a placeholder cannot be replaced, the function raises a
    `ValueError`.

    Parameters:
        d (dict): The dictionary to be processed for placeholder replacement.
        rd (dict): A dictionary containing placeholder-value pairs for replacement.

    Raises:
        ValueError: If unresolved placeholders are found in the values of `d`.

    Example:
        d = {"key1": "This is ${token1}", "key2": "Some text ${token2}"}
        rd = {"token1": "value1", "token2": "value2"}
        recursive_replace(d, rd)

    Note:
        This function modifies the input dictionary `d` in place.

    """
    cont = True
    while cont:
        cont = False
        for k, v in d.items():
            if isinstance(v, str):
                tokens = set(re.findall(r"\${(.*?)}", v))
                intersection = tokens & rd.keys()
                for token in intersection:
                    v = v.replace(f"${{{token}}}", rd[token])
                    cont = True
                d[k] = v
                if len(intersection) == len(tokens):
                    rd[k] = v

    for k, v in d.items():
        if isinstance(v, str):
            tokens = set(re.findall(r"\${(.*?)}", v))
            if len(tokens):
                raise ValueError(
                    f"Found unresolved tokens in the value of key '{k}': {', '.join(tokens)}"
                )

    for k, v in d.items():
        if isinstance(v, dict):
            recursive_replace(v, rd)


def bbox(pts):
    """
    Calculate the bounding box of a set of 3D points.

    Parameters:
    pts (np.ndarray): An array of 3D points with shape (n, 3).

    Returns:
    np.ndarray: An array containing the minimum and maximum coordinates of the bounding box.
               The first element is the minimum coordinates, and the second element is the maximum coordinates.

    Example:
    pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bbox(pts) returns an array with the minimum and maximum coordinates of the bounding box.

    """
    return np.array([np.min(pts, axis=0), np.max(pts, axis=0)])


def generate_cube_corners(a, b, n):
    """
    Generate an array of n cube corner points between two given points a and b.

    Parameters:
    a (array-like): The lower corner of the cube.
    b (array-like): The upper corner of the cube.
    n (int): The number of corner points to generate.

    Returns:
    np.ndarray: An array of n corner points.

    Example:
    generate_cube_corners([1, 1, 1], [2, 3, 3], 8) returns an array of 8 corner points within the specified cube.

    """
    l = [a, b]
    ans = np.array(
        [
            np.array([l[i % 2][0], l[(i // 2) % 2][1], l[(i // 4) % 2][2]])
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
