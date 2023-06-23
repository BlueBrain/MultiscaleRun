import logging
import numpy as np
import scipy as sp
from scipy import sparse
import os
import pickle
import time
import psutil

import config

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MrException(Exception):
    pass


def timesteps(end: float, step: float):
    """
    Timestep generator

    Ex: timesteps(1, 0.2) -> [0.2, 0.4, ..., 1.0]
    """
    return ((i + 1) * step for i in range(int(end / step)))


def addDict(d1, d2, datatype):  # d1 and d2 are from different ranks
    """
    Merge nested dicts

    Needed (ATM) for MPI purposes. Probably it can be removed in the future
    """

    return {k: v if k not in d2 else v | d2[k] for k, v in d1.items()}


def joinDict(d1, d2, datatype):
    """
    Merge dicts

    Needed (ATM) for MPI purposes. Probably it can be removed in the future
    """
    return d1 | d2


def print_once(*args, **kwargs):
    """Print only once among ranks"""
    if MPI4PY.COMM_WORLD.Get_rank() == 0:
        print(*args, *kwargs)


join_dict = MPI4PY.Op.Create(joinDict, commute=True)
add_dict = MPI4PY.Op.Create(addDict, commute=True)


def inspect(v, affix=""):
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


def flatten(l):
    return [item for sublist in l for item in sublist]


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError(
            f"works only for CSR format -- use .tocsr() first. Current type: {type(mat)}"
        )
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def rank_print(*args, **kwargs):
    print(f"rank {comm.Get_rank()}:", *args, **kwargs, flush=True)


def cache_decorator(path, is_save, is_load, field_names, only_rank0=False):
    """Caching system for parts of a class

    This decorator must be applied on a function that adds at least 1 field to a class. This field is
    chached in a smart way to memory

    The function cannot return stuff at the moment
    """

    if isinstance(field_names, str):
        field_names = [field_names]

    if only_rank0:    
        file_names = field_names
    else:
        file_names = [f"{i}_rank{rank}" for i in field_names]


    np.testing.assert_equal(len(field_names), len(file_names))
    np.testing.assert_array_less([0], [len(field_names)])

    def decorator_add_field_method(method):
        def wrapper(self, *args, **kwargs):
            fn_pnz = [os.path.join(path, i+".npz") for i in file_names if os.path.exists(os.path.join(path, i+".npz"))]
            fn_pickle = [os.path.join(path, i+".pickle") for i in file_names if os.path.exists(os.path.join(path, i+".pickle"))]

            fn = [*fn_pnz, *fn_pickle]
            for i in file_names:
                os.path.exists(os.path.join(path, i+".npz"))

            if len(fn) > len(file_names):
                raise FileNotFoundError("some files appear as pikle and npz, it is ambiguous")

            all_files_are_present = (len(fn) == len(file_names))

            if is_load and all_files_are_present:
                for field, full_path in zip(field_names, fn):

                    if not (rank == 0 or "rank" in full_path):
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
                try:
                    os.makedirs(path)
                except FileExistsError:
                    pass
                for field, file in zip(field_names, file_names):
                    full_path = os.path.join(path, file)
                    if rank == 0 or "rank" in file:
                        logging.info(f"save {field} to {full_path}")
                        obj = getattr(self, field)
                        if isinstance(obj, sparse.csr_matrix):
                            sparse.save_npz(full_path+".npz", obj)
                        else:
                            with open(full_path+".pickle", "wb") as f:
                                pickle.dump(obj, f)

        return wrapper

    return decorator_add_field_method


def logs_decorator(foo):
    def logs(*args, **kwargs):
        function_name = foo.__name__
        logging.info(f"{function_name}")
        start = time.perf_counter()
        res = foo(*args, **kwargs)
        mem = psutil.Process().memory_info().rss / 1024**2
        stop = time.perf_counter()
        logging.info(f"/{function_name}: mem: {mem}, time: {stop - start}")
        return res

    return logs


def ppf(n):
    return f"{n:.3}"

