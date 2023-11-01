# Test sec mapping algorithm to compute intersections among neuron segments and steps tets
# Test neuron removal tools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from timeit import default_timer as timer

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

from mpi4py import MPI as MPI4PY
from multiscale_run import (
    connection_manager,
    neurodamus_manager,
    steps_manager,
    utils,
    config,
    preprocessor,
)

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

conf0 = config.MsrConfig()


@utils.clear_and_replace_files_decorator([conf0.mesh_path.parent, conf0.cache_path])
@utils.logs_decorator
def test_sync():
    """
    Test synchronization between neurodamus, connection manager, and steps manager.

    This function tests the synchronization between the neurodamus data, connection manager, and steps manager.
    It performs the following steps:
    1. Initializes the configuration using a 'MsrConfig' object.
    2. Creates a connection manager using 'MsrConnectionManager'.
    3. Initializes a neurodamus manager using 'MsrNeurodamusManager'.
    4. Connects neurodamus to neurodamus using 'connect_ndam2ndam'.
    5. Initializes a steps manager using 'MsrStepsManager' and calls 'init_sim'.
    6. Computes the time step using neurodamus and configuration parameters.
    7. Solves the neurodamus system for the computed time step.
    8. Synchronizes neurodamus data with steps manager using 'ndam2steps_sync'.

    This function is used to ensure that the synchronization between these components works correctly.

    """
    conf = config.MsrConfig()

    prep = preprocessor.MsrPreprocessor(conf)
    prep.autogen_node_sets()
    conn_m = connection_manager.MsrConnectionManager(config=conf)

    ndam_m = neurodamus_manager.MsrNeurodamusManager(config=conf)
    conn_m.connect_ndam2ndam(ndam_m)
    prep.autogen_mesh(ndam_m=ndam_m)
    steps_m = steps_manager.MsrStepsManager(config=conf)
    steps_m.init_sim()

    conn_m.connect_ndam2steps(ndam_m=ndam_m, steps_m=steps_m)

    steps_dt = ndam_m.dt() * conf.steps_ndts
    ndam_m.ndamus.solve(steps_dt)
    conn_m.ndam2steps_sync(
        ndam_m=ndam_m, steps_m=steps_m, specs=conf.steps.Volsys.species, DT=steps_dt
    )


if __name__ == "__main__":
    test_sync()
