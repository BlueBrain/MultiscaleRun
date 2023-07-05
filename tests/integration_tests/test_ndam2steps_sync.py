# Test sec mapping algorithm to compute intersections among neuron segments and steps tets
# Test neuron removal tools

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

# this needs to be before "import neurodamus" and before MPI4PY
from neuron import h

h.nrnmpi_init()
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Memory tracking
import logging

from multiscale_run import utils, steps_manager, neurodamus_manager, connection_manager
config = utils.load_config()

import numpy as np

from timeit import default_timer as timer
from scipy.sparse import diags


@utils.logs_decorator
def test_sync():
    conn_m = connection_manager.MsrConnectionManager()

    ndam_m = neurodamus_manager.MsrNeurodamusManager(config.sonata_path)
    steps_m = steps_manager.MsrStepsManager(config.steps_mesh_path)

    conn_m.connect_ndam2steps(ndam_m=ndam_m, steps_m=steps_m)

    steps_dt = ndam_m.dt() * config.n_DT_steps_per_update["steps"]
    ndam_m.ndamus.solve(steps_dt)
    conn_m.ndam2steps_sync(
        ndam_m=ndam_m, steps_m=steps_m, specs=config.Volsys.specs, DT=steps_dt
    )


if __name__ == "__main__":
    test_sync()
