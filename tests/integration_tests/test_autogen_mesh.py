from pathlib import Path

import numpy as np

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()


# steps_manager should go before preprocessor until https://github.com/CNS-OIST/HBP_STEPS/issues/1166 is solved
from multiscale_run import (
    bloodflow_manager,
    config,
    neurodamus_manager,
    preprocessor,
    steps_manager,
    utils,
)


def base_path():
    return Path(__file__).resolve().parent


def generate_random_points_in_cube(a, b, n):
    """
    Generate n random points within a cube defined by two given points a and b.

    Args:
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

    Args:
      f (callable): A function to generate points within a cube.
      n (int): The number of points to generate using the function f.

    This function generates points using the provided function, creates a mesh, and performs various tests on the generated mesh.

    """
    conf = config.MsrConfig(base_path())
    mesh_path = conf.multiscale_run.mesh_path.parent
    utils.remove_path(mesh_path)

    prep = preprocessor.MsrPreprocessor(conf)

    pts = None
    if utils.rank0():
        pts = f([1, 1, 1], [2, 3, 3], n)

    pts = utils.comm().bcast(pts, root=0)

    prep.autogen_mesh(pts=pts)

    steps_m = steps_manager.MsrStepsManager(conf)

    steps_m.check_pts_inside_mesh_bbox(pts_list=[pts * conf.multiscale_run.mesh_scale])
    utils.remove_path(mesh_path)


if __name__ == "__main__":
    test_autogen_mesh(utils.generate_cube_corners, 8 + 0)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 1)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 2)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 3)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 4)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 8)
    test_autogen_mesh(utils.generate_cube_corners, 8 + 9)
    test_autogen_mesh(generate_random_points_in_cube, 200)
