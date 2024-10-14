import logging
from pathlib import Path

import pytest

from multiscale_run import MsrConfig
from multiscale_run.config import MsrConfigSchemaError

logging.basicConfig(level=logging.INFO)


CWD = Path(__file__).resolve().parent
TEST_CONFIG = CWD / "test_folder" / "simulation_config.json"


def test_named_circuit():
    conf = MsrConfig.default()
    assert conf.node_set == "testNGVSSCX_AstroMini_CI"


def test_getattr():
    conf = MsrConfig(TEST_CONFIG)
    # ensure proper data transformation
    assert isinstance(conf.config_path, Path)
    assert isinstance(conf.multiscale_run, MsrConfig)
    assert isinstance(conf.multiscale_run.reports, MsrConfig)
    assert isinstance(conf.multiscale_run.foo_path, Path)
    assert isinstance(conf.multiscale_run.d.miao_path, Path)
    assert isinstance(conf.multiscale_run.includes, list)
    assert isinstance(conf.multiscale_run.includes[0], str)
    assert isinstance(conf.multiscale_run.includes[1], str)
    # ensure objects are not copied whenever we access a property
    assert id(conf.multiscale_run) == id(conf.multiscale_run)
    assert id(conf.multiscale_run.includes) == id(conf.multiscale_run.includes)
    assert id(conf.multiscale_run.reports) == id(conf.multiscale_run.reports)


def test_load():
    """Test if we are loading correctly"""

    def _test_config(conf):
        conf = conf.multiscale_run
        assert conf.a == 1, conf.a
        assert conf.c == 1, conf.c
        assert conf.d != {"q": 0}, conf.d
        assert str(conf.d.miao_path) == "RESULTS/bbb/RESULTS/hola"
        assert conf.includes == ["RESULTS/a", "RESULTS/b"]

    # config can be a pathlib.Path to a JSON file
    conf1 = MsrConfig(TEST_CONFIG)
    _test_config(conf1)
    # config can be a pathlib.Path to a directory
    _test_config(MsrConfig(TEST_CONFIG.parent))
    # config can also be a str to a file or directory
    _test_config(MsrConfig(str(TEST_CONFIG)))
    # finally, config can be a Python dict
    MsrConfig._from_dict(conf1.multiscale_run.d)


def test_check():
    default_circuit = MsrConfig.default()
    # default config is valid
    default_circuit.check()

    ndts = default_circuit["multiscale_run"]["metabolism"]["ndts"]
    with pytest.raises(MsrConfigSchemaError) as excinfo:
        default_circuit["multiscale_run"]["metabolism"]["ndts"] = "what?"
        default_circuit.check()
    assert "'what?' is not of type 'integer'" in str(excinfo.value)
    default_circuit["multiscale_run"]["metabolism"]["ndts"] = ndts

    assert isinstance(default_circuit["multiscale_run"]["metabolism"]["u0_path"], Path)
    assert isinstance(type(default_circuit.multiscale_run.metabolism.u0_path), Path)

    del default_circuit["multiscale_run"]["metabolism"]["ndts"]
    with pytest.raises(MsrConfigSchemaError) as excinfo:
        default_circuit.check()
    assert "JSONEncoder" not in str(excinfo.value)
    assert "Error: 'ndts' is a required property" in str(excinfo.value)


if __name__ == "__main__":
    test_getattr()
    test_load()
    test_check()
    test_named_circuit()
