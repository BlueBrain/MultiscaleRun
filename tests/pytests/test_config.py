import logging
from pathlib import Path

from multiscale_run import MsrConfig

logging.basicConfig(level=logging.INFO)


def base_path():
    return Path(__file__).resolve().parent / "test_folder" / "simulation_json"


def test_load():
    """Test if we are loading correctly"""
    sp = base_path()

    def _test_config(conf):
        conf = conf.multiscale_run
        assert conf.a == 1, conf.a
        assert conf.c == 1, conf.c
        assert conf.d != {"q": 0}, conf.d
        assert str(conf.d.miao_path) == "RESULTS/bbb/RESULTS/hola"
        assert conf.includes == ["RESULTS/a", "RESULTS/b"]

    # config can be a pathlib.Path to a JSON file
    conf1 = MsrConfig(sp)
    _test_config(conf1)
    # config can be a pathlib.Path to a directory
    _test_config(MsrConfig(sp.parent))
    # config can also be a str to a file or directory
    _test_config(MsrConfig(str(sp)))
    # finally, config can be a Python dict
    MsrConfig._from_dict(conf1.multiscale_run.d)


if __name__ == "__main__":
    test_load()
