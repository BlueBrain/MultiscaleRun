import h5py
import numpy as np

from mpi4py import MPI as MPI4PY
from multiscale_run import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrReporterException(Exception):
    pass


class MsrReporter:
    """A class to handle the reporting of multiscale simulations.
    
    Attributes:
        config (config.MsrConfig): Configuration object containing simulation parameters.
        t_unit (str): The time units for the simulation.
        d_units (dict): Dictionary to store data units for each group.
        buffers (dict): Dictionary to store data buffers for each group.
        gids (list): List of global identifiers for the nodes.
    """

    def __init__(self, config, gids, t_unit="ms"):
        """
        Initializes the MsrReporter object.

        Parameters:
            config (config.MsrConfig): Configuration object containing simulation parameters.
            gids (list): List of global identifiers for the nodes.
            t_unit (str, optional): Time units for the simulation. Defaults to 'ms'.
        """
        self.config = config
        self.t_unit = t_unit
        self.d_units = {}
        self.buffers = {}

        self.init_offsets(gids)

    @utils.logs_decorator
    def init_offsets(self, gids):
        """
        Initializes offsets for data recording based on global identifiers.

        Parameters:
            gids (list): List of global identifiers for the nodes.
        """
        self.gids = gids
        self.all_gids = comm.gather(self.gids, root=0)
        ps = []
        if rank == 0:
            ps = [0, *np.cumsum([len(i) for i in self.all_gids])[:-1]]
            self.all_gids = [j for i in self.all_gids for j in i]
        self.offset = comm.scatter(ps, root=0)
        self.gids = {i: idx for idx, i in enumerate(self.gids)}

    @utils.logs_decorator
    def register_group_cols(self, group, cols, units):
        """
        Registers a group with its columns and units for data reporting.

        Parameters:
            group (str): The name of the group.
            cols (dict): A dictionary of columns and their corresponding data.
            units (list): A list of units corresponding to each column in cols.
        """
        if len(cols) == 0:
            raise MsrReporterException(
                f"Adding an empty group of columns is not permitted"
            )
        if group in self.buffers:
            return

        self.buffers[group] = {i: np.zeros(len(self.gids)) for i in cols.keys()}
        self.d_units[group] = dict(zip(self.buffers[group].keys(), units))

    @utils.logs_decorator
    def reset_buffers(self):
        """
        Resets the data buffers to zero for all groups.
        This is typically used after data has been flushed to disk.
        """
        for buffer in self.buffers.values():
            for col in buffer:
                buffer[col] = np.zeros(len(self.gids))

    @utils.logs_decorator
    def set_group(self, group, cols, units, gids):
        """
        Sets data for a specific group based on the provided global identifiers (gids).

        Parameters:
            group (str): The name of the group.
            cols (dict): A dictionary of columns and their corresponding data.
            units (list): A list of units corresponding to each column in cols.
            gids (list): List of global identifiers for the nodes in the group.
        """
        self.register_group_cols(group, cols, units)

        rows = [self.gids[i] for i in gids]
        for col, v in cols.items():
            self.buffers[group][col][rows] = v

    @utils.logs_decorator
    def flush_buffer(self, idt):
        """
        Flushes the buffer data to disk. This method writes the data stored in buffers to an HDF5 file.

        Parameters:
            idt (int): The index of the timestep at which data is being flushed.
        """
        for group, d in self.buffers.items():
            for col, v in d.items():
                path = self.file_path(group, col)
                self.init_file_if_necessary(path, group, col)

                with h5py.File(path, "a", driver="mpio", comm=comm) as file:
                    file[f"{self.data_loc}/data"][
                        idt, self.offset : self.offset + len(self.gids)
                    ] = v

        self.reset_buffers()

    @property
    def data_loc(self):
        """
        Returns the data location string. This is used to create the path in the HDF5 file where data is stored.

        Returns:
            str: A string representing the data location within the HDF5 file.
        """
        return f"/report/{self.config.preprocessor.node_sets.neuron_population_name}"

    def file_path(self, group, name):
        """
        Generates the file path for storing data of a specific group and data name.

        Parameters:
            group (str): The name of the group.
            name (str): The name of the data being stored.

        Returns:
            Path: A Path object representing the file path.
        """
        if isinstance(name, (tuple, list)):
            name = "_".join(name)
        return self.config.results_path / f"msr_{group}_{name}.h5"

    @utils.logs_decorator
    def init_file_if_necessary(self, path, group, name):
        """
        Initializes the HDF5 file for data storage. This sets up the file structure and metadata.

        Parameters:
            path (Path): The path where the HDF5 file will be created.
            group (str): The name of the group.
            name (str): The name of the data being stored.
        """
        if rank == 0 and not path.exists():
            dt = self.config.dt("metabolism")
            sim_end = self.config.msr_sim_end
            t_unit = self.t_unit

            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(str(path), "w") as file:
                base_group = file.create_group(self.data_loc)
                nrows = int(sim_end // dt)
                data = np.zeros((nrows, len(self.all_gids)), dtype=np.float32)
                data_dataset = base_group.create_dataset("data", data=data)
                data_dataset.attrs["units"] = self.d_units[group][name]
                mapping_group = base_group.create_group("mapping")
                data = np.array([i - 1 for i in self.all_gids], dtype=np.uint64)
                node_ids_dataset = mapping_group.create_dataset("node_ids", data=data)
                data = np.array([0, sim_end, dt], dtype=np.float64)
                time_dataset = mapping_group.create_dataset("time", data=data)
                time_dataset.attrs["units"] = t_unit

        comm.Barrier()
