from pathlib import Path
import json
import os

from multiscale_run.cli import main


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
