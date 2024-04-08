from pathlib import Path

import numpy as np

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

from multiscale_run import config, preprocessor, steps_manager, utils


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
    conf.multiscale_run.mesh_path = str(get_mesh_path())
    utils.remove_path(Path(conf.multiscale_run.mesh_path).parent)
    config.cache_load = False
    config.cache_save = False

    prep = preprocessor.MsrPreprocessor(conf)

    prep.autogen_mesh(pts=np.array([[100, 100, 200], [300, 500, 400]]))
    steps_m = steps_manager.MsrStepsManager(conf)
    steps_m.init_sim()

    pts = gen_segments_in_bbox(steps_m.msh)
    mat, st = steps_m.get_tetXbfSegMat(pts)

    if utils.rank0():
        np.testing.assert_allclose(
            mat.transpose().dot(np.ones(mat.shape[0])), np.ones(mat.shape[1])
        )
        np.testing.assert_array_less(np.array(st)[:, 1], [steps_m.ntets] * len(st))
    else:
        assert mat is None, mat
        assert st is None, st

    for it in range(10):
        t = 0.025 * conf.multiscale_run.steps.ndts * it
        steps_m.sim.run(t / 1000)

        assert np.all(
            steps_m.get_tet_concs(species_name="KK") > 0
        ), "KK concentration is <=0 in at least one tet"
        assert np.all(
            steps_m.get_tet_counts(species_name="KK") > 0
        ), "KK count is <=0 in at least one tet"

    utils.remove_path(Path(conf.multiscale_run.mesh_path).parent)


def test_steps_with_minimesh():
    """To be used manually with multiple ranks to see if omega_h complains"""

    conf = config.MsrConfig.rat_sscxS1HL_V6()
    conf.multiscale_run.mesh_path = str(get_mesh_path())
    conf.multiscale_run.preprocessor.mesh.refinement_steps = 0
    utils.remove_path(Path(conf.multiscale_run.mesh_path).parent)
    config.cache_load = False
    config.cache_save = False

    prep = preprocessor.MsrPreprocessor(conf)
    prep.autogen_mesh(pts=np.array([[100, 100, 200], [300, 500, 400]]))
    steps_manager.MsrStepsManager(conf)
    utils.remove_path(Path(conf.multiscale_run.mesh_path).parent)


if __name__ == "__main__":
    # test_steps_with_minimesh()
    test_steps_connections_mats()
