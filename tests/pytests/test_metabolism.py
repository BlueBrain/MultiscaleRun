import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from julia import Main
from multiscale_run import config, metabolism_manager
from multiscale_run.data import DEFAULT_CIRCUIT


def test_metabolism():
    conf = config.MsrConfig(base_path_or_dict=DEFAULT_CIRCUIT)
    metab_m = metabolism_manager.MsrMetabolismManager(
        config=conf, main=Main, neuron_pop_name="All"
    )


if __name__ == "__main__":
    test_metabolism()
