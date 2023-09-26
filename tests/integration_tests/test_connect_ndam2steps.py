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


def check_ratio_mat(m):
    np.testing.assert_allclose(m.dot(np.ones(m.shape[1])), np.ones(m.shape[0]))


def check_mats_shape(ndam_m, conn_m, steps_m, nshape=None, segshape=None):
    d = {
        int(nc.CCell.gid): len(
            [seg for sec in ndam_m._gen_secs(nc) for seg in ndam_m._gen_segs(sec)]
        )
        for nc in ndam_m.ncs
    }
    nn = len(ndam_m.ncs)
    nseg = sum([v for k, v in d.items() if k not in ndam_m.removed_gids])
    
    np.testing.assert_equal(conn_m.nXtetMat.shape[0], nn)
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

    np.testing.assert_equal(conn_m.nXnsegMatBool.shape, (nn, nseg))

    np.testing.assert_equal(conn_m.nXnsegMatBool.shape, (nn, nseg))


@utils.logs_decorator
def test_connection():
    conn_m = connection_manager.MsrConnectionManager()

    ndam_m = neurodamus_manager.MsrNeurodamusManager(config=config)
    conn_m.connect_ndam2ndam(ndam_m=ndam_m)
    steps_m = steps_manager.MsrStepsManager(config.steps_mesh_path)

    conn_m.connect_ndam2steps(ndam_m=ndam_m, steps_m=steps_m)

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

    utils.rank_print("gid: ", [int(nc.CCell.gid) for nc in ndam_m.ncs])
    ndam_m.remove_gids(failed_cells={933: "remove from rank 0", 1004: "remove from rank 1"}, conn_m=conn_m)
    ndam_m.update(conn_m)

    utils.rank_print([int(nc.CCell.gid) for nc in ndam_m.ncs])

    l = [v for k, v in d.items() if k in [933, 1004]]
    nshape = (nshape[0] - len(l), nshape[1])
    segshape = (segshape[0] - sum(l), segshape[1])
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)
    nshape = conn_m.nXtetMat.shape
    segshape = conn_m.nsegXtetMat.shape

    utils.rank_print([int(nc.CCell.gid) for nc in ndam_m.ncs])

    ndam_m.update(conn_m)
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)

    utils.rank_print([int(nc.CCell.gid) for nc in ndam_m.ncs])

    ndam_m.remove_gids(failed_cells={17500: "reason 0"}, conn_m=conn_m)
    ndam_m.update(conn_m)
    check_mats_shape(ndam_m, conn_m, steps_m, nshape=nshape, segshape=segshape)

    utils.rank_print([int(nc.CCell.gid) for nc in ndam_m.ncs])

    ndam_m.remove_gids(failed_cells={175: "reason 0"}, conn_m=conn_m)
    ndam_m.update(conn_m)

    check_mats_shape(
        ndam_m,
        conn_m,
        steps_m,
        nshape=nshape if rank != 0 else None,
        segshape=segshape if rank != 0 else None,
    )


if __name__ == "__main__":
    test_connection()
