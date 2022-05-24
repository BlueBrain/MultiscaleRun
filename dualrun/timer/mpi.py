from mpi4py import MPI
import numpy as np
import sys

from contextlib import contextmanager
import time

class timer:
    """ A simple mpi timer class"""
    _timings = dict()

    @contextmanager
    def region(var):
        start = time.time()
        yield
        elapsed = time.time() - start
        if var in timer._timings:
            timer._timings[var][0] += elapsed
            timer._timings[var][1] += 1
        else:
            timer._timings[var] = [elapsed, 1]

    def print():
        comm = MPI.COMM_WORLD
        nRanks = comm.Get_size()
        myRank = comm.Get_rank()

        # let everyone flush to get a clean output
        sys.stdout.flush()
        comm.Barrier()

        llbl = 0
        for reg in timer._timings:
            llbl = max(len(reg), llbl)

        if myRank == 0:
            print(80*"-")
            print("region", (llbl-6)*" ", "max", 12*" ", "min", 12*" ", "ave", 12*" ", "times" )
            print(80*"-")

        for reg in timer._timings:
            region = np.zeros(1)
            region[0] = timer._timings[reg][0]
            count = timer._timings[reg][1]
            region_max = np.zeros(1)
            comm.Reduce(region, region_max, op=MPI.MAX, root=0)
            region_min = np.zeros(1)
            comm.Reduce(region, region_min, op=MPI.MIN, root=0)
            region_ave = np.zeros(1)
            comm.Reduce(region, region_ave, op=MPI.SUM, root=0)
            region_ave /= nRanks

            if myRank == 0:
                s = f'{reg}{(llbl-len(reg))*" "}{region_max[0]:17.11f}{region_min[0]:17.11f}{region_ave[0]:17.11f}{count:7d}'
                print(s)

        if myRank == 0:
            print(80*"-")


if(__name__ =='__main__'):
    comm = MPI.COMM_WORLD
    nRanks = comm.Get_size()
    myRank = comm.Get_rank()

    assert nRanks == 2

    with timer.region("region"):
        req = [MPI.REQUEST_NULL  for i in range(0,2)]

        if myRank == 0:
            data = np.array([0,1,2], dtype=np.int64)
        else:
            data = np.zeros(3, dtype=np.int64 )

        if myRank == 0:
            comm.Isend([data, 3, MPI.LONG], dest=1, tag=myRank)
        else:
            req[1] = comm.Irecv([data, 3, MPI.LONG], source=0, tag=MPI.ANY_TAG)

    with timer.region("wait_print"):
        MPI.Request.Waitall(req)

        if myRank == 0:
            print("sent     ", data)
        else:
            print("received ", data)


    timer.print()