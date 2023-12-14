import json

import pytest

from multiscale_run.cli import argument_parser, main
from multiscale_run.data import BB5_JULIA_ENV


def test_init_without_metabolism(tmp_path):
    sim_path = tmp_path / "sim"
    main(args=["init", "--julia=no", str(sim_path)])

    assert sim_path.is_dir()

    msr_config_f = sim_path / "msr_config.json"
    assert msr_config_f.is_file(), "top JSON config is missing"
    with open(msr_config_f) as istr:
        msr_config = json.load(istr)
    assert "with_metabolism" in msr_config, "JSON config is missing a mandatory key"
    assert msr_config["with_metabolism"] is False, "Metabolism should be disabled"


@pytest.mark.skipif(not BB5_JULIA_ENV.exists(), reason="BB5 resources required")
def test_init_twice(tmp_path):
    sim_path = tmp_path / "sim"

    # Prepare a new simulation with "shared" Julia setup
    main(args=["init", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()

    # Recreate the simulation at the same location
    main(args=["init", "--force", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()

    # mess a little the Julia setup
    (sim_path / ".julia").unlink()
    (sim_path / ".julia").mkdir()

    # Recreate the simulation at the same location
    main(args=["init", "--force", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()


def test_valid_commands(tmp_path):
    path = str(tmp_path)
    ap = argument_parser()
    try:
        ap.parse_args(["init"])
        ap.parse_args(["check"])
        ap.parse_args(["compute"])
        ap.parse_args(["julia"])

        ap.parse_args(["-v", "init"])
        ap.parse_args(["-vv", "init"])

        ap.parse_args(["init", path])
        ap.parse_args(["init", "--force", path])
        ap.parse_args(["init", "--julia", "no", path])
        ap.parse_args(["init", "--julia=no", path])
        ap.parse_args(["init", "--circuit=rat_sscxS1HL_V6", path])

        ap.parse_args(["check", path])
        ap.parse_args(["compute", path])
        ap.parse_args(["julia", 'print("Foo")', 'Pkg.install("Foo")'])
    except:
        pytest.fail("Parsing of valid commands failed")
