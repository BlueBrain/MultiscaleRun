import logging
import numpy as np

from mpi4py import MPI as MPI4PY


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


def check_param(param, idxm):
    """Check that all the parameters are valid floats"""

    s0 = "at least one param is "
    s2 = f"{', '.join([str(i) for i in param])}) (idxm: {idxm}, rank: {MPI4PY.COMM_WORLD.Get_rank()})"

    try:
        param = [float(i) for i in param]
    except:

        return s0 + "not a float" + s2

    if np.isnan(param).any():
        return s0 + "nan" + s2
    if np.isinf(param).any():
        return s0 + "inf" + s2

    return ""


join_dict = MPI4PY.Op.Create(joinDict, commute=True)
add_dict = MPI4PY.Op.Create(addDict, commute=True)
