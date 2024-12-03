from pathlib import Path

import h5py
import numpy as np

from multiscale_run import MsrConfig, MsrReporter, utils


def config_path():
    return Path(__file__).resolve().parent / "test_folder" / "simulation_config.json"


class FakeNeurodamusManager:
    def __init__(self) -> None:
        self.circuit_offset = 1000
        gids = {0: [1, 2], 1: [6], 2: [], 3: [5, 7, 11]}
        ps = np.cumsum([len(i) for i in gids.values()])
        ps = [0, *ps[:-1]]
        self.offset = ps[utils.rank()]
        self._gids = gids[utils.rank()]
        self.base_gids = gids.copy()

    def gids(self, raw=False):
        if raw:
            return self._gids
        else:
            return [i + self.circuit_offset for i in self._gids]


class FakeMetabolismManager:
    def __init__(self, raw_gids) -> None:
        vals = {
            0: [i + 1 for i in range(len(raw_gids))],
            1: [i + 3 for i in range(len(raw_gids))],
            2: [i + 5 for i in range(len(raw_gids))],
            3: [i + 7 for i in range(len(raw_gids))],
        }
        self.vals = np.array([[-1 for _ in raw_gids], vals[utils.rank()]]).transpose()

    def get_vals(self, idx):
        return self.vals[:, idx]


class FakeBloodflowManager:
    def __init__(self):
        self.vals = np.array(list(range(10)))

    @property
    def n_segs(self):
        return len(self.vals)

    def get_flows(self, idxs=None):
        if idxs is not None:
            return self.vals[idxs]
        return self.vals


def test_simple_reports():
    conf = MsrConfig(config_path())
    folder_path = conf.config_path.parent / conf.output.output_dir

    utils.remove_path(folder_path)

    pop_name = conf.multiscale_run.preprocessor.node_sets.neuron_population_name
    idt = 1
    managers = {}
    managers["neurodamus"] = FakeNeurodamusManager()
    managers["bloodflow"] = FakeBloodflowManager()
    gids = managers["neurodamus"].gids()
    offset = managers["neurodamus"].offset
    n_bf_segs = managers["bloodflow"].n_segs
    managers["metabolism"] = FakeMetabolismManager(
        raw_gids=managers["neurodamus"].gids(raw=True)
    )

    t_unit = "mss"

    rr = MsrReporter(config=conf, gids=gids, n_bf_segs=n_bf_segs, t_unit=t_unit)

    rr.record(idt=idt, manager_name="metabolism", managers=managers, when="after_sync")
    rr.record(idt=idt, manager_name="bloodflow", managers=managers, when="after_sync")
    utils.comm().Barrier()

    for rep in conf.multiscale_run.reports.metabolism.values():
        path = rr._file_path(rep.file_name)
        with h5py.File(path, "r") as file:
            data = file[f"{rr.data_loc}/data"]
            assert np.allclose(
                data[idt, offset : offset + len(gids)],
                managers["metabolism"].get_vals(1),
            )
            assert np.allclose(
                data[idt - 1, offset : offset + len(gids)], [0] * len(gids)
            )
            assert data.attrs["units"] == rep.unit
            data = file[f"/report/{pop_name}/mapping/node_ids"]
            assert np.allclose(data[offset : offset + len(gids)], [i - 1 for i in gids])
            data = file[f"/report/{pop_name}/mapping/time"]
            assert np.allclose(data, [0, conf.run.tstop, conf.metabolism_dt])
            assert data.attrs["units"] == t_unit

    if utils.rank0:
        for rep in conf.multiscale_run.reports.bloodflow.values():
            path = rr._file_path(rep.file_name)
            with h5py.File(path, "r") as file:
                data = file[f"{rr.data_loc}/data"]
                assert np.allclose(
                    data[idt, :],
                    managers["bloodflow"].get_flows(**rep.src_get_kwargs),
                )
                assert np.allclose(
                    data[idt - 1, :],
                    [0] * len(rep.src_get_kwargs.idxs),
                )
                assert data.attrs["units"] == rep.unit
                data = file[f"/report/{pop_name}/mapping/node_ids"]
                assert np.allclose(data, rep.src_get_kwargs.idxs)
                data = file[f"/report/{pop_name}/mapping/time"]
                assert np.allclose(data, [0, conf.run.tstop, conf.metabolism_dt])
                assert data.attrs["units"] == t_unit

    utils.remove_path(folder_path)


if __name__ == "__main__":
    test_simple_reports()
