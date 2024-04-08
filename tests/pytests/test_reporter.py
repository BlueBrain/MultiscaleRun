from pathlib import Path

import h5py
import numpy as np

from multiscale_run import config, reporter, utils


def config_path():
    return Path(__file__).resolve().parent / "test_folder" / "simulation_config.json"


class FakeNeurodamusManager:
    def __init__(self) -> None:
        gids = {0: [1, 2], 1: [6], 2: [], 3: [5, 7, 11]}
        ps = np.cumsum([len(i) for i in gids.values()])
        ps = [0, *ps[:-1]]
        self.offset = ps[utils.rank()]
        self._gids = gids[utils.rank()]
        self.base_gids = gids.copy()

    @property
    def gids(self):
        return self._gids


class FakeMetabolismManager:
    def __init__(self, gids) -> None:
        vals = {
            0: [i + 1 for i in range(len(gids))],
            1: [i + 3 for i in range(len(gids))],
            2: [i + 5 for i in range(len(gids))],
            3: [i + 7 for i in range(len(gids))],
        }
        self.vals = np.array([[-1 for _ in gids], vals[utils.rank()]]).transpose()

    def get_vals(self, idx):
        return self.vals[:, idx]


def test_simple_report():
    conf = config.MsrConfig(config_path())
    folder_path = conf.config_path.parent / conf.output.output_dir

    utils.remove_path(folder_path)

    pop_name = conf.multiscale_run.preprocessor.node_sets.neuron_population_name
    idt = 1
    managers = {}
    managers["neurodamus"] = FakeNeurodamusManager()
    gids = managers["neurodamus"].gids
    offset = managers["neurodamus"].offset
    managers["metabolism"] = FakeMetabolismManager(gids=gids)

    t_unit = "mss"

    rr = reporter.MsrReporter(config=conf, gids=gids, t_unit=t_unit)

    rr.record(idt=idt, manager_name="metabolism", managers=managers, when="after_sync")
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

    utils.remove_path(folder_path)


if __name__ == "__main__":
    test_simple_report()
