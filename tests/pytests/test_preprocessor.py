import numpy as np
from neuron import h

h.nrnmpi_init()

from multiscale_run import preprocessor


def test_explode_pts():
    pts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0]])
    pts = preprocessor.MsrPreprocessor._explode_pts(pts, 1.1)
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.1],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, -1.1],
            [0.0, -1.1, 0.0],
        ]
    )

    assert np.allclose(pts, v)
    pts = np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0], [0, 2, 2]])
    pts = preprocessor.MsrPreprocessor._explode_pts(pts, 1.1)
    v = np.array(
        [[0.0, -0.1, -0.1], [0.0, -0.1, 2.1], [0.0, 2.1, -0.1], [0.0, 2.1, 2.1]]
    )

    assert np.allclose(pts, v)
    pts = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0], [0, 4, 4]])
    pts = preprocessor.MsrPreprocessor._explode_pts(pts, 1.2)
    v = np.array(
        [[0.0, -0.4, -0.4], [0.0, -0.4, 4.4], [0.0, 4.4, -0.4], [0.0, 4.4, 4.4]]
    )

    assert np.allclose(pts, v)
    pts = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0]])
    pts = preprocessor.MsrPreprocessor._explode_pts(pts, 1.0)
    v = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0]])
    assert np.allclose(pts, v)


if __name__ == "__main__":
    test_explode_pts()
