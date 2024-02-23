from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI as MPI4PY

from multiscale_run import reporter, utils, config

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def config_path():
    return str(
        Path(__file__).resolve().parent / "test_folder/test_folder1/test_folder2"
    )


def test_simple_report():
    conf = config.MsrConfig(config_path_or_dict=config_path())
    conf.merge_without_priority({"DT": 0.025})

    pop_name = conf.preprocessor.node_sets.neuron_population_name
    sim_end = conf.msr_sim_end
    dt = conf.DT * conf.metabolism_ndts
    idt = 0

    utils.remove_path(conf.results_path)

    # gids are 1-based in input
    gids = {0: [1, 2], 1: [6], 2: [], 3: [5, 7, 11]}
    ps = np.cumsum([len(i) for i in gids.values()])
    ps = [0, *ps[:-1]]
    offset = ps[rank]
    idt = 0
    gids = gids[rank]
    base_gids = gids.copy()
    t_unit = "mss"

    rr = reporter.MsrReporter(config=conf, gids=gids, t_unit=t_unit)

    d = {"bau": ["A", "B"], "miao": [["C", "1"], ["B", "2"]]}

    if len(gids) > 1:
        gids.pop(1)

    for k, v in d.items():
        q = {tuple(i): [rank + 1] * len(gids) for i in v}

        rr.set_group(k, q, ["mol"] * len(q), gids)

    rr.flush_buffer(idt)
    comm.Barrier()

    for group, cols in rr.buffers.items():
        for name, v in cols.items():
            path = rr.file_path(group, name)
            with h5py.File(path, "r") as file:
                data = file[f"{rr.data_loc}/data"]
                q = [rank + 1] * len(v)
                if len(q) > 1:
                    q[1] = 0
                assert np.allclose(data[idt, offset : offset + len(v)], q)
                assert data.attrs["units"] == rr.d_units[group][name]
                data = file[f"/report/{pop_name}/mapping/node_ids"]
                assert np.allclose(
                    data[offset : offset + len(v)], [i - 1 for i in base_gids]
                )
                data = file[f"/report/{pop_name}/mapping/time"]
                assert np.allclose(data, [0, sim_end, dt])
                assert np.allclose(data, [0, sim_end, dt])
                assert data.attrs["units"] == t_unit

    utils.remove_path(conf.results_path)


if __name__ == "__main__":
    test_simple_report()
    test_env_imports()
