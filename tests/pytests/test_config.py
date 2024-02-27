import logging
from pathlib import Path

from multiscale_run import config

logging.basicConfig(level=logging.INFO)


def base_path():
    return (
        Path(__file__).resolve().parent
        / "test_folder"
        / "test_folder1"
        / "test_folder2"
        / "msr_config.json"
    )


def test_load():
    """Test if we are loading correctly"""
    sp = base_path()

    def _test_config(conf):
        assert conf.a == 2, conf.a
        assert conf.c == 1, conf.c
        assert conf.d != {"q": 0}, conf.d
        assert conf.e == 1, conf.e
        assert str(conf.d.miao_path) == "aaa/bbb/aaa/hola"
        assert conf.includes == ["RESULTS/a", "RESULTS/b"]

    # config can be a pathlib.Path to a JSON file
    conf1 = config.MsrConfig(sp)
    _test_config(conf1)
    # config can be a pathlib.Path to a directory
    _test_config(config.MsrConfig(sp.parent))
    # config can also be a str to a file or directory
    _test_config(config.MsrConfig(str(sp)))
    # finally, config can be a Python dict
    config.MsrConfig._from_dict(conf1.d)


if __name__ == "__main__":
    test_load()
