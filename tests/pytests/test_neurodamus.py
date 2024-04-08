# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

from multiscale_run import config, neurodamus_manager


def test_init():
    conf = config.MsrConfig.rat_sscxS1HL_V6()
    neurodamus_manager.MsrNeurodamusManager(config=conf)


if __name__ == "__main__":
    test_init()
