import h5py
import numpy as np

from multiscale_run import utils


class MsrReporterException(Exception):
    pass


class MsrReporter:
    """A class for reporting multiscale simulation data."""

    def __init__(self, config, gids: list[int], t_unit="ms"):
        """Initializes the MsrReporter instance.

        Args:
            config (MsrConfig): Configuration argument.
            gids : List of gids.
            t_unit (optional): Time unit. Defaults to "ms".
        """
        self.config = config
        self.t_unit = t_unit

        self.init_offsets(gids)
        self.init_files()

    @utils.logs_decorator
    def init_offsets(self, gids: list[int]):
        """Initializes offsets for the gids.

        Args:
            gids: List of gids.
        """
        self.all_gids = utils.comm().gather(gids, root=0)
        ps = []
        if utils.rank0():
            ps = [0, *np.cumsum([len(i) for i in self.all_gids[:-1]])]
            self.all_gids = [j for i in self.all_gids for j in i]
        self.offset = utils.comm().scatter(ps, root=0)
        self.gid2pos = {gid: idx + self.offset for idx, gid in enumerate(gids)}

    @property
    def data_loc(self):
        """Returns the data location string.

        Returns
            str: A string representing the data location within the HDF5 file.
        """
        return f"/report/{self.config.multiscale_run.preprocessor.node_sets.neuron_population_name}"

    def _file_path(self, file_name):
        return (
            self.config.config_path.parent / self.config.output.output_dir / file_name
        )

    @utils.logs_decorator
    def init_files(self):
        """Initializes files for reporting."""
        if utils.rank0():
            sim_end = self.config.run.tstop

            for manager, reps in self.config.multiscale_run.reports.items():
                if not self.config.is_manager_active(manager):
                    continue

                dt = getattr(self.config, f"{manager}_dt")
                # timesteps generates a vector like [1, 2, 3, ..., n]. We also have a recording at t=0.
                # Thus, we need n+1 rows
                nrows = len(utils.timesteps(sim_end, dt)) + 1

                for rep in reps.values():
                    data = np.zeros((nrows, len(self.all_gids)), dtype=np.float32)

                    path = self._file_path(rep.file_name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(str(path), "w") as file:
                        base_group = file.create_group(self.data_loc)
                        data = np.zeros((nrows, len(self.all_gids)), dtype=np.float32)
                        data_dataset = base_group.create_dataset("data", data=data)
                        data_dataset.attrs["units"] = rep.unit
                        mapping_group = base_group.create_group("mapping")
                        data = np.array([i - 1 for i in self.all_gids], dtype=np.uint64)
                        mapping_group.create_dataset("node_ids", data=data)
                        data = np.array([0, sim_end, dt], dtype=np.float64)
                        time_dataset = mapping_group.create_dataset("time", data=data)
                        time_dataset.attrs["units"] = self.t_unit

        utils.comm().Barrier()

    def record(self, idt: int, manager_name: str, managers: dict, when: str):
        """Records simulation data.

        Args:
            idt: time step index of the current manager.
            manager_name: current manager.
            managers: dict of managers
            when: When the data should be recorded (before or after sync).
        """
        if (
            not self.config.is_manager_active(manager_name)
            or manager_name not in self.config.multiscale_run.reports
        ):
            return

        manager = managers[manager_name]
        idxs = np.array([self.gid2pos[gid] for gid in managers["neurodamus"].gids])
        for rep in getattr(self.config.multiscale_run.reports, manager_name).values():
            if rep.when == when:
                path = self._file_path(rep.file_name)
                with h5py.File(path, "a", driver="mpio", comm=utils.comm()) as file:
                    dataset = file[f"{self.data_loc}/data"]
                    vals = np.array(
                        getattr(manager, rep.src_get_func)(**rep.src_get_kwargs),
                        dtype=np.float64,
                    )
                    if not len(vals):
                        continue
                    dataset[idt, idxs] = vals
