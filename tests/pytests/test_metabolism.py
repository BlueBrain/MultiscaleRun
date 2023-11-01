import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from julia import Main
from multiscale_run import metabolism_manager, printer, utils, config


def test_metabolism():
    conf = config.MsrConfig()
    prnt = printer.MsrPrinter(config=conf)
    metab_m = metabolism_manager.MsrMetabolismManager(
        config=conf, main=Main, prnt=prnt, neuron_pop_name="All"
    )


if __name__ == "__main__":
    test_metabolism()
