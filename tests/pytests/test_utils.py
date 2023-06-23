import sys, os, glob, shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import multiscale_run.utils as utils
from scipy import sparse

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np

# this needs to be before "import neurodamus" and before MPI4PY
from neuron import h

h.nrnmpi_init()

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def remove_test_cache_files(path):
    if rank == 0:
        try:    
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
    comm.Barrier()


cache = "cache_tests"


class A:
    @utils.cache_decorator(
        path=cache,
        is_save=True,
        is_load=True,
        field_names="a",
        only_rank0=True,
    )
    def add_obj_a(self, a):
        self.a = None
        if rank == 0:
            self.a = a

    @utils.cache_decorator(
        path=cache,
        is_save=True,
        is_load=True,
        field_names="b",
        only_rank0=True,
    )
    def add_obj_b(self, a):
        self.b = None
        if rank == 0:
            self.b = sparse.csr_matrix((3, 4))
            self.b[1, 2] = a

    @utils.cache_decorator(
        path=cache,
        is_save=True,
        is_load=True,
        field_names="c",
        only_rank0=False,
    )
    def add_obj_c(self, a):
        self.c = a


def instantiate_and_check(a, b, c, aexp, bexp, cexp):
    obj = A()
    obj.add_obj_a(a)
    obj.add_obj_b(b)
    obj.add_obj_c(c)
    np.testing.assert_equal(obj.a, aexp if rank == 0 else None)
    if rank == 0:
        np.testing.assert_equal(obj.b[1, 2], bexp)
    else:
        np.testing.assert_equal(obj.b, None)
    np.testing.assert_equal(obj.c, cexp)


def test_cache_decor():
    remove_test_cache_files(cache)
    instantiate_and_check(1, 2, 3 if rank == 0 else 4, 1, 2, 3 if rank == 0 else 4)
    instantiate_and_check(10, 20, 30 if rank == 0 else 40, 1, 2, 3 if rank == 0 else 4)
    remove_test_cache_files(cache)


class B:
    @utils.logs_decorator
    def do_a(self, a):
        return a


def test_logs_decorator():
    b = B()
    res = b.do_a((3, 4))
    np.testing.assert_equal(res, (3, 4))


def test_timestamps():
    assert list(utils.timesteps(10, 1)) == list(range(1, 11, 1))
    assert list(utils.timesteps(10, 0.9)) == [i * 0.9 for i in range(1, 12, 1)]


if __name__ == "__main__":
    test_cache_decor()
    test_logs_decorator()
