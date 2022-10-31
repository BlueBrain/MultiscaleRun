from mpi4py import MPI as MPI4PY
import os
from pathlib import Path


class MsrPrinter:
    def __init__(self, result_path):
        self.result_path = result_path
        Path(result_path).mkdir(parents=True, exist_ok=True)
        self.files = {}

    def __del__(self):
        for i in self.files.values():
            i.close()

    def file_path(self, file):
        return os.path.join(self.result_path, file)

    def append_to_file(self, file, values, rank=None):
        if rank is None:
            rank = 0
        else:
            file += f"_rank{rank}"

        if MPI4PY.COMM_WORLD.Get_rank() != rank:
            return

        if type(values) is list:
            values = ", ".join([str(i) for i in values])
        else:
            values = str(values)
        values += "\n"

        if file not in self.files:
            self.files[file] = open(self.file_path(file), "a")

        self.files[file].write(values)
