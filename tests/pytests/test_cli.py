from pathlib import Path
import json
import os

from multiscale_run.cli import main


def test_init_without_metabolism(tmp_path):
    sim_path = tmp_path / "sim"
    main(args=["init", "--julia=no", str(sim_path)])

    assert sim_path.is_dir()

    mr_config_f = sim_path / "mr_config.json"
    assert mr_config_f.is_file(), "top JSON config is missing"
    with open(mr_config_f) as istr:
        mr_config = json.load(istr)
    assert "with_metabolism" in mr_config, "JSON config is missing a mandatory key"
    assert mr_config["with_metabolism"] is False, "Metabolism should be disabled"
