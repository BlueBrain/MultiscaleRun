import logging
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import multiscale_run.utils as utils

CACHE_DIR = "cache_tests"


class A:
    @utils.cache_decorator(
        path=CACHE_DIR,
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
        path=CACHE_DIR,
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
        path=CACHE_DIR,
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
    utils.remove_path(CACHE_DIR)
    instantiate_and_check(
        1, 2, 3 if utils.rank0() else 4, 1, 2, 3 if utils.rank0() else 4
    )
    instantiate_and_check(
        10, 20, 30 if utils.rank0() else 40, 1, 2, 3 if utils.rank0() else 4
    )
    utils.remove_path(CACHE_DIR)


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


def test_py_expr_eval():
    config = dict(
        multiscale_run=dict(
            steps=dict(
                conc_factor=2,
            ),
            version="1.0",
        )
    )

    pyeval = utils.PyExprEval(config)
    # integral and floating-points
    assert pyeval("42") == 42
    assert pyeval("'42'") == "42"
    assert 0.001 == pytest.approx(pyeval("1.0e-3"))
    # test custom function 'config'
    assert pyeval("config['multiscale_run']['version']") == "1.0"
    assert pyeval("config.multiscale_run.steps.conc_factor") == 2
    # test variables
    assert pyeval("foo * 2", foo=21) == 42
    assert pyeval("foo * 2", foo=[]) == [], "cannot handle Python list"
    assert pyeval("foo * 3", foo=[1]) == [1, 1, 1], "cannot handle Python list"
    assert np.array_equal(
        pyeval("foo * 2", foo=np.array([1, 21])), np.array([2, 42])
    ), "cannot handle NumPy arrays"
    assert np.array_equal(
        pyeval("np.arange(6)"), np.arange(6)
    ), "cannot call Numpy functions"
    assert (
        pyeval("math.ceil(41.2)") == 42
    ), "cannot call routines from the 'math' standard module"
    assert pyeval("abs(-42)") == 42, "cannot call builtins from the standard library"


def test_replace_values():
    # Test case 1: Replace values in a simple dictionary
    d1 = {"key1": "value1", "key2": "value2"}
    replace_dict = {"value1": "new_value1", "value2": "new_value2"}
    expected1 = {"key1": "new_value1", "key2": "new_value2"}
    assert utils.replace_values(d1, replace_dict) == expected1

    # Test case 2: Replace values in a nested dictionary
    d2 = {"key1": {"subkey1": "value1"}, "key2": ["value2", "value3"]}
    replace_dict2 = {
        "value1": "new_value1",
        "value2": "new_value2",
        "value3": "new_value3",
    }
    expected2 = {
        "key1": {"subkey1": "new_value1"},
        "key2": ["new_value2", "new_value3"],
    }
    assert utils.replace_values(d2, replace_dict2) == expected2

    # Test case 3: Replace values in a list with nested dictionaries
    d3 = [{"key1": "value1"}, {"key2": "value2"}]
    replace_dict3 = {"value1": "new_value1", "value2": "new_value2"}
    expected3 = [{"key1": "new_value1"}, {"key2": "new_value2"}]
    assert utils.replace_values(d3, replace_dict3) == expected3

    # Test case 4: No replacements when replace_dict is empty
    d4 = {"key1": "value1", "key2": "value2"}
    replace_dict4 = {}
    assert utils.replace_values(d4, replace_dict4) == d4

    # Test case 5: No replacements when d is not a str, list, or dict
    d5 = 123
    replace_dict5 = {"value1": "new_value1"}
    assert utils.replace_values(d5, replace_dict5) == d5


if __name__ == "__main__":
    test_get_var_name()
    test_check_value()
    test_replacing_algos()
    test_cache_decor()
    test_logs_decorator()
    test_heavy_duty_MPI_gather()
    test_py_expr_eval()
    test_replace_values()
