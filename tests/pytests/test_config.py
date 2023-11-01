import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "../.."))

from multiscale_run import config

logging.basicConfig(level=logging.INFO)


def test_load():
    """Test if we are loading correctly"""
    sp = "tests/pytests/test_folder/test_folder1/test_folder2"
    a = config.MsrConfig(base_path_or_dict=sp)

    assert a.a == 2, a.a
    assert a.c == 1, a.c
    assert a.d != {"q": 0}, a.d
    assert a.e == 1, a.e
    assert str(a.d.miao_path) == "aaa/bbb/aaa/hola"

    print(a)

    b = config.MsrConfig(base_path_or_dict=a.d)

    print(a.d.q)


def test_env_overrides():
    sp = "tests/pytests/test_folder/test_folder1/test_folder2"
    a = config.MsrConfig(base_path_or_dict=sp)
    assert a.a == 2, a.a

    os.environ["a"] = "5"
    a = config.MsrConfig(base_path_or_dict=sp)
    assert a.a == 5, a.a


if __name__ == "__main__":
    test_load()
    test_env_overrides()
