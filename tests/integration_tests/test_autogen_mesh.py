import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.sparse import diags

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()


from mpi4py import MPI as MPI4PY

# steps_manager should go before preprocessor until https://github.com/CNS-OIST/HBP_STEPS/issues/1166 is solved
from multiscale_run import (
    utils,
    steps_manager,
    config,
    bloodflow_manager,
    neurodamus_manager,
    preprocessor,
)

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def generate_random_points_in_cube(a, b, n):
    """
    Generate n random points within a cube defined by two given points a and b.

    Parameters:
    a (array-like): The lower corner of the cube.
    b (array-like): The upper corner of the cube.
    n (int): The number of random points to generate.

    Returns:
    np.ndarray: An array of n random points within the cube.

    Example:
    generate_random_points_in_cube([1, 1, 1], [2, 3, 3], 10) returns an array of 10 random points within the specified cube.

    """
    random_points = np.random.uniform(a, b, size=(n, 3))

    return random_points


@utils.logs_decorator
def test_autogen_mesh(f, n):
    """
    Test the autogeneration of a mesh using the provided function f.

    Parameters:
    f (function): A function to generate points within a cube.
    n (int): The number of points to generate using the function f.

    This function generates points using the provided function, creates a mesh, and performs various tests on the generated mesh.

    """
    conf = config.MsrConfig(base_path_or_dict="tests/integration_tests")
    utils.remove_path(conf.mesh_path.parent)

    prep = preprocessor.MsrPreprocessor(config=conf)

    pts = None
    if rank == 0:
        pts = f([1, 1, 1], [2, 3, 3], n)

    pts = comm.bcast(pts, root=0)

    prep.autogen_mesh(pts=pts)

    steps_m = steps_manager.MsrStepsManager(config=conf)

    steps_m.check_pts_inside_mesh_bbox(pts_list=[pts * conf.mesh_scale])
    utils.remove_path(conf.mesh_path.parent)


def test_gen_msh():
    """
    Test mesh generation with specific configurations and managers.

    This function tests mesh generation using specific configurations and managers. It includes steps to create, manage, and clean up the mesh.

    """
    conf = config.MsrConfig()
    tmp_mesh_path = conf.mesh_path.parent.name + "_tmp"
    utils.rename_path(
        conf.mesh_path.parent, conf.mesh_path.parent.with_name(tmp_mesh_path)
    )

    pp = preprocessor.MsrPreprocessor(config=conf)
    ndam_m = neurodamus_manager.MsrNeurodamusManager(conf)
    bf_m = bloodflow_manager.MsrBloodflowManager(
        vasculature_path=ndam_m.get_vasculature_path(),
        params=conf.bloodflow,
    )
    pp.autogen_mesh(ndam_m=ndam_m, bf_m=bf_m)

    # utils.remove_path(conf.mesh_path.parent)
    # utils.rename_path(
    #     conf.mesh_path.parent.with_name(tmp_mesh_path), conf.mesh_path.parent
    # )


if __name__ == "__main__":
    test_autogen_mesh(utils.generate_cube_corners, 8 + 0)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 1)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 2)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 3)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 4)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 8)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 9)
    test_autogen_mesh(generate_random_points_in_cube, 200)
    test_gen_msh()
