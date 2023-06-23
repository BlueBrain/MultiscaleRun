import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

# this needs to be before "import neurodamus"
from neuron import h

h.nrnmpi_init()

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

from multiscale_run import utils, neurodamus_manager

import logging

from diffeqpy import de

import config


def test_init():
    ndam_m = neurodamus_manager.MsrNeurodamusManager(config.sonata_path)


if __name__ == "__main__":
    test_init()
