import logging
import os
from pathlib import Path

from multiscale_run import config

logging.basicConfig(level=logging.INFO)


def base_path():
    return str(
        Path(__file__).resolve().parent / "test_folder/test_folder1/test_folder2"
    )


def test_load():
    """Test if we are loading correctly"""
    sp = base_path()
    a = config.MsrConfig(base_path_or_dict=sp)

    assert a.a == 2, a.a
    assert a.c == 1, a.c
    assert a.d != {"q": 0}, a.d
    assert a.e == 1, a.e
    assert str(a.d.miao_path) == "aaa/bbb/aaa/hola"

    b = config.MsrConfig(base_path_or_dict=a.d)


if __name__ == "__main__":
    test_load()
    test_env_overrides()
