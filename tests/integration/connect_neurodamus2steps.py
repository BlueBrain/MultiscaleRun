# Test sec mapping algorithm to compute intersections among neuron segments and steps tets
# Test neuron removal tools
import numpy as np
from neuron import h

from multiscale_run import (
    MsrConfig,
    MsrConnectionManager,
    MsrNeurodamusManager,
    MsrPreprocessor,
    MsrStepsManager,
    utils,
)


def check_ratio_mat(m):
    """
    Check if a matrix satisfies a specific condition.

    Parameters:
    m (np.ndarray): A NumPy array to be checked.

    This function checks if the sum of each row in the matrix 'm' is approximately equal to 1.
    If the condition is not met, it raises an AssertionError using `np.testing.assert_allclose`.

    Example:
    check_ratio_mat(conn_m.nXtetMat) checks if the provided matrix satisfies the specified condition.

    """

    np.testing.assert_allclose(m.dot(np.ones(m.shape[1])), np.ones(m.shape[0]))


def check_mats_shape(ndam_m, conn_m, steps_m, nshape=None, segshape=None):
    """
    Check the shapes of connection matrices and compare them to provided shapes.

    Parameters:
    ndam_m: An object representing neurodamus data.
    conn_m: An object representing the connection manager.
    steps_m: An object representing steps manager.
    nshape (tuple): The expected shape for the connection matrix nXtetMat.
    segshape (tuple): The expected shape for the connection matrix nsegXtetMat.

    This function checks the shapes of various connection matrices and compares them to the provided expected shapes.
    If the provided shapes are not None, it also checks if the shapes match the expected shapes using `np.testing.assert_equal`.

    Example:
    check_mats_shape(ndam_m, conn_m, steps_m, (10, 20), (15, 30)) checks the shapes of connection matrices
    and compares them to the expected shapes.

    """
    d = {
        int(nc.CCell.gid): len(
            [seg for sec in ndam_m.gen_secs(nc) for seg in ndam_m.gen_segs(sec)]
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


@utils.pushtempd
@utils.logs_decorator
def test_connection():
    """
    Test the connection between neurodamus, connection manager, and steps manager.

    This function tests the connection between neurodamus data, the connection manager, and the steps manager.
    It also checks various conditions using the 'check_ratio_mat' and 'check_mats_shape' functions.

    """
    conf = MsrConfig.default(circuit="tiny_CI", check=False, force=True)

    prep = MsrPreprocessor(conf)
    managers = {}
    conn_m = MsrConnectionManager(config=conf, managers=managers)

    prep.autogen_node_sets()

    managers["neurodamus"] = MsrNeurodamusManager(conf)
    conn_m.connect_neurodamus2neurodamus()

    prep.autogen_mesh(ndam_m=managers["neurodamus"])
    managers["steps"] = MsrStepsManager(conf)

    conn_m.connect_neurodamus2steps()

    check_ratio_mat(conn_m.nXtetMat)
    check_ratio_mat(conn_m.nsegXtetMat)

    nshape = conn_m.nXtetMat.shape
    segshape = conn_m.nsegXtetMat.shape
    d = {
        int(nc.CCell.gid): len(
            [
                seg
                for sec in managers["neurodamus"].gen_secs(nc)
                for seg in managers["neurodamus"].gen_segs(sec)
            ]
        )
        for nc in managers["neurodamus"].ncs
    }

    utils.rank_print("gid: ", [int(nc.CCell.gid) for nc in managers["neurodamus"].ncs])
    failed_cells_dict = {933: "remove from rank 0", 1004: "remove from rank 1"}
    failed_cells = [failed_cells_dict.get(i, None) for i in managers["neurodamus"].gids]
    conn_m.remove_gids(failed_cells=failed_cells)

    utils.rank_print([int(nc.CCell.gid) for nc in managers["neurodamus"].ncs])

    l = [v for k, v in d.items() if k in [933, 1004]]
    nshape = (nshape[0] - len(l), nshape[1])
    segshape = (segshape[0] - sum(l), segshape[1])
    check_mats_shape(
        managers["neurodamus"],
        conn_m,
        managers["steps"],
        nshape=nshape,
        segshape=segshape,
    )
    nshape = conn_m.nXtetMat.shape
    segshape = conn_m.nsegXtetMat.shape

    utils.rank_print([int(nc.CCell.gid) for nc in managers["neurodamus"].ncs])

    check_mats_shape(
        managers["neurodamus"],
        conn_m,
        managers["steps"],
        nshape=nshape,
        segshape=segshape,
    )

    utils.rank_print([int(nc.CCell.gid) for nc in managers["neurodamus"].ncs])

    failed_cells_dict = {17500: "reason 0"}
    failed_cells = [failed_cells_dict.get(i, None) for i in managers["neurodamus"].gids]
    conn_m.remove_gids(failed_cells=failed_cells)
    check_mats_shape(
        managers["neurodamus"],
        conn_m,
        managers["steps"],
        nshape=nshape,
        segshape=segshape,
    )

    utils.rank_print([int(nc.CCell.gid) for nc in managers["neurodamus"].ncs])

    failed_cells_dict = {175: "reason 0"}
    failed_cells = [failed_cells_dict.get(i, None) for i in managers["neurodamus"].gids]

    check_mats_shape(
        managers["neurodamus"],
        conn_m,
        managers["steps"],
        nshape=nshape if not utils.rank0() else None,
        segshape=segshape if not utils.rank0() else None,
    )


if __name__ == "__main__":
    h.nrnmpi_init()
    test_connection()
