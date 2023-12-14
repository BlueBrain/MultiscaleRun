import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

from multiscale_run import neurodamus_manager, config
from multiscale_run.data import DEFAULT_CIRCUIT


def test_init():
    conf = config.MsrConfig(base_path_or_dict=DEFAULT_CIRCUIT)
    ndam_m = neurodamus_manager.MsrNeurodamusManager(config=conf)


if __name__ == "__main__":
    test_init()
