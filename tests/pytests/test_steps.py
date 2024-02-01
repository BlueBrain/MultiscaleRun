import numpy as np
from pathlib import Path

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

from mpi4py import MPI as MPI4PY
from multiscale_run import config, preprocessor, steps_manager, utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def get_mesh_path():
    return Path.cwd() / "tmp/test_mesh.msh"


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


def test_steps_connections_mats():
    conf = config.MsrConfig.rat_sscxS1HL_V6()
    conf.mesh_path = str(get_mesh_path())
    utils.remove_path(Path(conf.mesh_path).parent)
    config.cache_load = False
    config.cache_save = False

    prep = preprocessor.MsrPreprocessor(conf)

    prep.autogen_mesh(pts=np.array([[100, 100, 200], [300, 500, 400]]))
    steps_m = steps_manager.MsrStepsManager(conf)
    steps_m.init_sim()

    pts = gen_segments_in_bbox(steps_m.msh)
    mat, st = steps_m.get_tetXbfSegMat(pts)

    if rank == 0:
        np.testing.assert_allclose(
            mat.transpose().dot(np.ones(mat.shape[0])), np.ones(mat.shape[1])
        )
        np.testing.assert_array_less(np.array(st)[:, 1], [steps_m.ntets] * len(st))
    else:
        assert mat is None, mat
        assert st is None, st

    for it in range(10):
        t = 0.025 * conf.steps_ndts * it
        steps_m.sim.run(t / 1000)

        assert np.all(steps_m.get_tet_concs(species_name="Na") > 0)
        assert np.all(steps_m.get_tet_counts(species_name="Na") > 0)

    utils.remove_path(Path(conf.mesh_path).parent)


def test_steps_with_minimesh():
    """To be used manually with multiple ranks to see if omega_h complains"""

    conf = config.MsrConfig.rat_sscxS1HL_V6()
    conf.mesh_path = str(get_mesh_path())
    conf.preprocessor.mesh.refinement_steps = 0
    utils.remove_path(Path(conf.mesh_path).parent)
    config.cache_load = False
    config.cache_save = False

    prep = preprocessor.MsrPreprocessor(conf)
    prep.autogen_mesh(pts=np.array([[100, 100, 200], [300, 500, 400]]))
    steps_m = steps_manager.MsrStepsManager(conf)
    utils.remove_path(Path(conf.mesh_path).parent)


if __name__ == "__main__":
    test_steps_with_minimesh()
    # test_steps_connections_mats()
