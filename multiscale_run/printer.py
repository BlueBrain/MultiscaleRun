from pathlib import Path

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrPrinter:
    """
    A class for managing and printing data to multiple files based on a configuration.
    """

    def __init__(self, config):
        """
        Initialize an instance of MsrPrinter.

        Parameters:
        - config (dict): A configuration dictionary that specifies the behavior of the printer.
        """
        self.config = config
        self.files = {}

    def __del__(self):
        """
        Destructor method that is automatically called when the object is no longer in use.
        Closes all open files associated with this printer.
        """
        for file in self.files.values():
            file.close()

    def append_to_file(self, file, values, root=None, keys=[]):
        """
        Append values to a specified file, optionally creating multiple files based on rank.

        Parameters:
        - file (str or Path): The name of the file or a Path object to which the data will be appended.
        - values (list, str, or any): The values to be appended to the file. Can be a list of values, a string, or any data.
        - root (int, optional): The rank (an integer) that determines if the data should be appended. If None, set to 0.
        - keys (str, optional): Header keys to be written to the file, typically used for the first time the file is created.

        Notes:
        - If the 'root' parameter is specified and does not match the current rank, no data will be appended.
        - If the 'values' parameter is a list, it will be converted to a comma-separated string before appending.
        - If the specified file does not exist, it will be created, and any necessary parent directories will be created as well.

        Example usage:
        ```
        msr_printer = MsrPrinter(config)
        msr_printer.append_to_file("data.txt", [1, 2, 3], root=0, keys="A, B, C\n")
        ```

        This example appends the values [1, 2, 3] to a file named "data.txt" for rank 0 and includes the header "A, B, C" in the file.
        """
        if root is None:
            root = 0
        else:
            file = Path(file)
            file = Path(str(file.with_suffix("")) + f"_rank{rank}" + file.suffix)

        if rank != root:
            return

        if type(values) is list:
            values = ", ".join([str(i) for i in values])
        else:
            values = str(values)
        values += "\n"

        if file not in self.files:
            file.parent.mkdir(parents=True, exist_ok=True)
            self.files[file] = open(file, "a")
            if len(keys):
                self.files[file].write(keys)

        self.files[file].write(values)
