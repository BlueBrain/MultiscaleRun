import logging
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

# this needs to be before "import neurodamus" and before MPI4PY
from neuron import h

h.nrnmpi_init()

import multiscale_run.utils as utils

logging.basicConfig(level=logging.INFO)

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
        if utils.rank0():
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
        if utils.rank0():
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
    np.testing.assert_equal(obj.a, aexp if utils.rank0() else None)
    if utils.rank0():
        np.testing.assert_equal(obj.b[1, 2], bexp)
    else:
        np.testing.assert_equal(obj.b, None)
    np.testing.assert_equal(obj.c, cexp)


def test_cache_decor():
    utils.remove_path(cache)
    instantiate_and_check(
        1, 2, 3 if utils.rank0() else 4, 1, 2, 3 if utils.rank0() else 4
    )
    instantiate_and_check(
        10, 20, 30 if utils.rank0() else 40, 1, 2, 3 if utils.rank0() else 4
    )
    utils.remove_path(cache)


class B:
    @utils.logs_decorator
    def do_a(self, a):
        return a


def test_logs_decorator():
    b = B()
    res = b.do_a((3, 4))
    np.testing.assert_equal(res, (3, 4))


def test_heavy_duty_MPI_gather():
    def get_rank_matrix(dtype="i", n=3, m=5, p=68573):
        return np.array(
            [
                [(j + i * n + utils.rank() * n * m) % p for j in range(n)]
                for i in range(m)
            ],
            dtype=dtype,
        )

    def final_matrix(dtype="i", n=3, m=5, p=68573):
        return np.array(
            [
                [[(j + i * n + r * n * m) % p for j in range(n)] for i in range(m)]
                for r in range(utils.size())
            ],
            dtype=dtype,
        )

    local_mat = get_rank_matrix()
    final_mat = utils.heavy_duty_MPI_Gather(local_mat, root=0)
    final_mat2 = utils.comm().gather(local_mat, root=0)
    if utils.rank0():
        assert (final_mat == final_matrix()).all()
        assert (final_mat == final_mat2).all()

    local_mat = get_rank_matrix(dtype="float")
    final_mat = utils.heavy_duty_MPI_Gather(local_mat, root=0)
    final_mat2 = utils.comm().gather(local_mat, root=0)
    if utils.rank0():
        assert (final_mat == final_matrix(dtype="float")).all()
        assert (final_mat == final_mat2).all()


def test_clear_and_replace_files_decorator():
    p = Path("bau")

    if utils.rank0():
        if not p.exists():
            p.mkdir()

    @utils.clear_and_replace_files_decorator(str(p))
    def f():
        logging.info("check clear_and_replace_files_decorator")
        utils.comm().Barrier()
        assert not p.exists()
        utils.comm().Barrier()

    utils.comm().Barrier()
    assert p.exists()
    utils.comm().Barrier()

    f()

    utils.comm().Barrier()
    assert p.exists()
    utils.comm().Barrier()

    utils.remove_path(p)


def test_replacing_algos():
    d = {"a": "${q}/${p}/d", "p": "p", "q": "${p}/q"}
    d_copy = {"a": "${q}/${p}/d", "p": "p", "q": "${p}/q"}
    v = utils.get_resolved_value(d, "a")
    assert v == "p/q/p/d", v
    assert d == d_copy, d

    d = {
        "a": "${m}/${q}/${p}/d",
        "miao": {"p": "p", "pera": {"q": "${m}/${p}/q", 1: 3}},
    }
    utils.resolve_replaces(d, {"m": "bau"})
    assert d == {
        "a": "bau/bau/p/q/p/d",
        "miao": {"p": "p", "pera": {"q": "bau/p/q", 1: 3}},
    }


def test_get_var_name():
    a = 3
    b = {3: 3}

    def _test(v, val):
        assert utils.get_var_name() == val

    _test(a, "a")
    _test(b[3], "b[3]")
    _test(b[a], "b[a]")

    def _test(a, b, val):
        assert utils.get_var_name(pos=1) == val

    _test(3, b[a], "b[a]")

    def _test(v, val):
        def _test2(v, val):
            assert utils.get_var_name() == "v"
            assert utils.get_var_name(lvl=2) == val

        _test2(v, val)

    _test(b[a], "b[a]")


def test_check_value():
    utils.check_value(3, 2)
    utils.check_value(3, 2.0)
    utils.check_value(3.0, 2.0)
    utils.check_value(3.0, 4e-1)
    utils.check_value(3.1e-1, 4e-2)
    utils.check_value(2)

    with pytest.raises(utils.MsrException):
        utils.check_value(None)
    with pytest.raises(utils.MsrException):
        utils.check_value(float("inf"))
    with pytest.raises(utils.MsrException):
        utils.check_value("AAA")
    with pytest.raises(utils.MsrException):
        utils.check_value(3, lb=4)
    with pytest.raises(utils.MsrException):
        utils.check_value(0, leb=0.0)
    with pytest.raises(utils.MsrException):
        utils.check_value(5, hb=4)
    with pytest.raises(utils.MsrException):
        utils.check_value(0.0, heb=0)
    with pytest.raises(utils.MsrException) as exc_info:
        a = 3
        utils.check_value(a, heb=0)

    print(exc_info.value)
    assert str(exc_info.value) == "a (3) >= 0"

    class CustomException(Exception):
        pass

    with pytest.raises(CustomException) as exc_info:
        a = 3
        utils.check_value(a, heb=0, err=CustomException)


def test_timesteps():
    l = utils.timesteps(10.0, 1.0)
    assert np.allclose(l, list(range(1, 11, 1)))
    l = utils.timesteps(10.01, 1.0)
    assert np.allclose(l, list(range(1, 11, 1)))
    l = utils.timesteps(10.99, 1.0)
    assert np.allclose(l, list(range(1, 11, 1)))
    l = utils.timesteps(9.99, 1.0)
    assert np.allclose(l, list(range(1, 10, 1)))


if __name__ == "__main__":
    test_get_var_name()
    test_check_value()
    test_replacing_algos()
    test_cache_decor()
    test_logs_decorator()
    test_heavy_duty_MPI_gather()
    test_clear_and_replace_files_decorator()
