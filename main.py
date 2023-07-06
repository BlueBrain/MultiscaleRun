from __future__ import print_function

import os
import logging

import copy

import numpy as np
import pandas as pd
from julia import Main

# this needs to be before "import neurodamus" and before MPI4PY
from neuron import h

h.nrnmpi_init()

# the object MPI exists already. It is the steps one
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

import neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from neurodamus.utils.timeit import TimerManager, timeit


# Memory tracking
import psutil

from multiscale_run import (
    metabolism_manager,
    utils,
    steps_manager,
    neurodamus_manager,
    bloodflow_manager,
    connection_manager,
    printer,
)
config = utils.load_config()


def main():
    config.print_config()
    prnt = printer.MsrPrinter()
    conn_m = connection_manager.MsrConnectionManager()

    with timeit(name="initialization"):
        ndam_m = neurodamus_manager.MsrNeurodamusManager(config.sonata_path)

        # Times are in ms (for NEURON, because STEPS works with SI)
        DT = ndam_m.dt()  # 0.025  #ms i.e. = 25 usec which is timstep of ndam
        SIM_END = ndam_m.duration()  # 500.0 #10.0 #1000.0 #ms

        rss = []  # Memory tracking

        logging.info(f"DT: {DT}")
        logging.info(f"SIM_END: {SIM_END}")
        for k, v in config.n_DT_steps_per_update.items():
            logging.info(f"{k} dt: {v*DT}")

        logging.info(f"Initializing simulations...")

        if config.with_steps:
            steps_m = steps_manager.MsrStepsManager(config.steps_mesh_path)
            conn_m.connect_ndam2steps(ndam_m=ndam_m, steps_m=steps_m)
            
        if config.with_metabolism:
            metab_m = metabolism_manager.MsrMetabolismManager(
                u0_file=config.u0_file, main=Main, prnt=prnt
            )

        if config.with_bloodflow:
            bf_m = bloodflow_manager.MsrBloodflowManager(
                vasculature_path=ndam_m.get_vasculature_path(),
                params=config.bloodflow_params,
            )
            conn_m.connect_bf2steps(bf_m=bf_m, steps_m=steps_m)


    log_stage("===============================================")
    log_stage("Running the selected solvers ...")

    # i_* is the number of time steps of that particular simulator
    i_ndam, i_metab = 0, 0
    for t in ProgressBar(int(SIM_END / (DT*config.n_DT_steps_per_update["mr"])))(utils.timesteps(SIM_END, DT*config.n_DT_steps_per_update["mr"])):
        i_ndam += config.n_DT_steps_per_update["mr"]
        with timeit(name="main_loop"):
            with timeit(name="neurodamus_solver"):
                ndam_m.ndamus.solve(t)

            if config.with_steps and (i_ndam % config.n_DT_steps_per_update["steps"] == 0):
                with timeit(name="steps_loop"):
                    log_stage("steps loop")
                    with timeit(name="steps_solver"):
                        steps_m.sim.run(t / 1000)  # ms to sec

                    with timeit(name="neurodamus_2_steps"):
                        conn_m.ndam2steps_sync(
                            ndam_m=ndam_m,
                            steps_m=steps_m,
                            specs=config.Volsys.specs,
                            DT=config.n_DT_steps_per_update["steps"]*DT,
                        )

            if config.with_bloodflow and (i_ndam % config.n_DT_steps_per_update["bf"] == 0):
                with timeit(name="bf_loop"):
                    log_stage("bf loop")
                    conn_m.ndam2bloodflow_sync(ndam_m=ndam_m, bf_m=bf_m)
                    bf_m.update_static_flow()

            if config.with_metabolism and (i_ndam % config.n_DT_steps_per_update["metab"] == 0):
                with timeit(name="metabolism_loop"):
                    log_stage("metab loop")
                    with timeit(name="neurodamus_2_metabolism"):
                        conn_m.ndam2metab_sync(ndam_m=ndam_m, metab_m=metab_m)

                    if config.with_steps:
                        with timeit(name="steps_2_metabolism"):
                            conn_m.steps2metab_sync(steps_m=steps_m, metab_m=metab_m)

                    if config.with_bloodflow:
                        with timeit(name="bloodflow_2_metabolism"):
                            conn_m.bloodflow2metab_sync(bf_m=bf_m, metab_m=metab_m)

                    with timeit(name="solve_metabolism"):
                        failed_cells = metab_m.advance(ndam_m.ncs, i_metab, config.n_DT_steps_per_update["metab"]*DT)

                    ndam_m.remove_gids(failed_cells, conn_m)

                    with timeit(name="metabolism_2_neurodamus"):
                        conn_m.metab2ndam_sync(
                            metab_m=metab_m, ndam_m=ndam_m, i_metab=i_metab
                        )

                    comm.Barrier()

                    i_metab += 1

            rss.append(
                psutil.Process().memory_info().rss / (1024**2)
            )  # memory consumption in MB

    ndam_m.ndamus.spike2file(os.getcwd() + "/" + prnt.file_path("out.dat"))
    TimerManager.timeit_show_stats()
    comm.Barrier()


if __name__ == "__main__":
    main()
