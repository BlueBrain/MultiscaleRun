import logging

import shutil
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy import sparse

# this needs to be before "import neurodamus" and before MPI4PY
from neuron import h

h.nrnmpi_init()

import multiscale_run.utils as utils
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


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
    utils.remove_path(cache)
    instantiate_and_check(1, 2, 3 if rank == 0 else 4, 1, 2, 3 if rank == 0 else 4)
    instantiate_and_check(10, 20, 30 if rank == 0 else 40, 1, 2, 3 if rank == 0 else 4)
    utils.remove_path(cache)


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


def test_heavy_duty_MPI_gather():
    def get_rank_matrix(dtype="i", n=3, m=5, p=68573):
        return np.array(
            [[(j + i * n + rank * n * m) % p for j in range(n)] for i in range(m)],
            dtype=dtype,
        )

    def final_matrix(dtype="i", n=3, m=5, p=68573):
        return np.array(
            [
                [[(j + i * n + r * n * m) % p for j in range(n)] for i in range(m)]
                for r in range(size)
            ],
            dtype=dtype,
        )

    local_mat = get_rank_matrix()
    final_mat = utils.heavy_duty_MPI_Gather(local_mat, root=0)
    final_mat2 = comm.gather(local_mat, root=0)
    if rank == 0:
        assert (final_mat == final_matrix()).all()
        assert (final_mat == final_mat2).all()

    local_mat = get_rank_matrix(dtype="float")
    final_mat = utils.heavy_duty_MPI_Gather(local_mat, root=0)
    final_mat2 = comm.gather(local_mat, root=0)
    if rank == 0:
        assert (final_mat == final_matrix(dtype="float")).all()
        assert (final_mat == final_mat2).all()


def test_recursive_replace():
    d = {"key1": "This is ${token1}"}
    rd = {"token1": "value1"}
    utils.recursive_replace(d, rd)
    assert d["key1"] == "This is value1"

    d = {"key1": "This is ${token1}", "key2": "Another ${token2} example"}
    rd = {"token1": "value1", "token2": "value2"}
    utils.recursive_replace(d, rd)
    assert d["key1"] == "This is value1"
    assert d["key2"] == "Another value2 example"

    d = {"key1": "This is ${token1} with ${token3}"}
    rd = {"token1": "value1", "token2": "value2"}
    try:
        utils.recursive_replace(d, rd)
    except ValueError as e:
        assert "token3" in str(e)

    d = {"outer": {"inner": "Nested ${token1}"}}
    rd = {"token1": "value1"}
    utils.recursive_replace(d, rd)
    assert d["outer"]["inner"] == "Nested value1"

    d = {
        "A": "bau",
        "B": "fusto",
        "C": "pera",
        "F": {"D": "${A}/miao", "L": {"E": "${D}/${M}/bau"}},
        "K": "aaa/${Q}/hola",
        "AA": "${BB}/${B}",
        "BB": "${A}/${B}",
    }
    rd = {"M": "zio", "Q": "pio", "R": "wak"}

    utils.recursive_replace(d, rd)
    assert d["F"]["L"]["E"] == "bau/miao/zio/bau"


def test_clear_and_replace_files_decorator():
    p = Path("bau")

    if rank == 0:
        if not p.exists():
            p.mkdir()

    @utils.clear_and_replace_files_decorator(str(p))
    def f():
        logging.info("check clear_and_replace_files_decorator")
        comm.Barrier()
        assert not p.exists()
        comm.Barrier()

    comm.Barrier()
    assert p.exists()
    comm.Barrier()

    f()

    comm.Barrier()
    assert p.exists()
    comm.Barrier()

    utils.remove_path(p)


if __name__ == "__main__":
    test_cache_decor()
    test_logs_decorator()
    test_heavy_duty_MPI_gather()
    test_recursive_replace()
    test_clear_and_replace_files_decorator()
