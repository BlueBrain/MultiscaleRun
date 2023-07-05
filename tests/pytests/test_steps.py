import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

# this needs to be before "import neurodamus"
from neuron import h

h.nrnmpi_init()

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

from multiscale_run import steps_manager, utils
config = utils.load_config()

import numpy as np


def gen_segments_in_bbox(msh):
    max0, min0 = np.array(msh.bbox.max.tolist()), np.array(msh.bbox.min.tolist())
    diff = max0 - min0
    low = 0.15 * diff + min0
    top = 0.9 * diff + min0
    middle = 0.5 * diff + min0
    lateral = np.array([0.75, 0.8, 0.5]) * diff + min0

    ans = np.array([low, middle, middle, top, lateral, middle])

    for i in ans:
        np.testing.assert_array_less(i, max0)
        np.testing.assert_array_less(min0, i)

    return ans


def test_only_steps():
    config.cache_load = False
    config.cache_save = False

    steps_m = steps_manager.MsrStepsManager(config.steps_mesh_path)

    pts = gen_segments_in_bbox(steps_m.msh)
    mat, st = steps_m.get_tetXbfSegMat(pts)

    if rank == 0:
        np.testing.assert_allclose(
            mat.transpose().dot(np.ones(mat.shape[0])), np.ones(mat.shape[1])
        )
        np.testing.assert_array_less(st, [steps_m.ntets] * len(st))
    else:
        assert mat == None, mat
        assert st == None, st

    for it in range(10):
        t = 0.025 * config.n_DT_steps_per_update["steps"] * it
        steps_m.sim.run(t / 1000)

        assert np.all(steps_m.get_tet_concs(species="Na") > 0)
        assert np.all(steps_m.get_tet_counts(species="Na") > 0)


if __name__ == "__main__":
    test_only_steps()
