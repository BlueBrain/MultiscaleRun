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

import config
import pytest

import numpy as np

import neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from neurodamus.utils.timeit import TimerManager, timeit

from timeit import default_timer as timer
from scipy.sparse import diags


def check_ratio_mat(m):
    np.testing.assert_allclose(m.dot(np.ones(m.shape[1])), np.ones(m.shape[0]))


def check_mats_shape(ndam_m, conn_m, steps_m, nshape=None, segshape=None):
    d = {
        int(nc.CCell.gid): len(
            [seg for sec in ndam_m._gen_secs(nc) for seg in ndam_m._gen_segs(sec)]
        )
        for nc in ndam_m.ncs
    }
    np.testing.assert_equal(conn_m.nXtetMat.shape[0], len(ndam_m.ncs))
    np.testing.assert_equal(conn_m.nXtetMat.shape[1], steps_m.ntets)
    np.testing.assert_equal(
        conn_m.nsegXtetMat.shape[0],
        sum([v for k, v in d.items() if k not in ndam_m.removed_gids]),
    )
    np.testing.assert_equal(conn_m.nsegXtetMat.shape[1], steps_m.ntets)

    if nshape is not None:
        np.testing.assert_equal(conn_m.nXtetMat.shape, nshape)

    if segshape is not None:
        np.testing.assert_equal(conn_m.nsegXtetMat.shape, segshape)


def test_connection():
    conn_m = connection_manager.MsrConnectionManager()

    ndam_m = neurodamus_manager.MsrNeurodamusManager(config.sonata_path)
    steps_m = steps_manager.MsrStepsManager(config.steps_mesh_path)

    conn_m.connect(ndam_m=ndam_m, steps_m=steps_m)

    check_ratio_mat(conn_m.nXtetMat)
    check_ratio_mat(conn_m.nsegXtetMat)

    nshape = conn_m.nXtetMat.shape
    segshape = conn_m.nsegXtetMat.shape
    d = {
        int(nc.CCell.gid): len(
            [seg for sec in ndam_m._gen_secs(nc) for seg in ndam_m._gen_segs(sec)]
        )
        for nc in ndam_m.ncs
    }

    ndam_m.removed_gids.update([933, 1004])
    ndam_m.update(conn_m)

    l = [v for k, v in d.items() if k in [933, 1004]]
    nshape = (nshape[0] - len(l), nshape[1])
    segshape = (segshape[0] - sum(l), segshape[1])
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)
    nshape = conn_m.nXtetMat.shape
    segshape = conn_m.nsegXtetMat.shape

    ndam_m.update(conn_m)
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)

    ndam_m.removed_gids.update([17500])
    ndam_m.update(conn_m)
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)

    ndam_m.removed_gids.update([175])
    ndam_m.update(conn_m)

    check_mats_shape(
        ndam_m,
        conn_m,
        steps_m,
        nshape=nshape if rank != 0 else None,
        segshape=segshape if rank != 0 else None,
    )

    utils.rank_print("end")


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

    utils.rank_print("end")


if __name__ == "__main__":
    test_connection()
    test_sync()
