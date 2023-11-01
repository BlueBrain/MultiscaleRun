import logging
import os

import numpy as np
import pandas as pd
from julia import Main

# this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs
from neuron import h

h.nrnmpi_init()

# the object MPI exists already. It is the steps one
from mpi4py import MPI as MPI4PY

# this goes before neurodamus for correct logging
from multiscale_run import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

import neurodamus

# Memory tracking
import psutil
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from neurodamus.utils.timeit import TimerManager, timeit

# steps_manager should go before preprocessor until https://github.com/CNS-OIST/HBP_STEPS/issues/1166 is solved
from multiscale_run import (
    bloodflow_manager,
    connection_manager,
    metabolism_manager,
    neurodamus_manager,
    steps_manager,
    preprocessor,
    printer,
    config,
)


def main():
    logging.basicConfig(level=logging.INFO)
    conf = config.MsrConfig()

    prep = preprocessor.MsrPreprocessor(conf)

    prnt = printer.MsrPrinter(conf)
    conn_m = connection_manager.MsrConnectionManager(conf)

    with timeit(name="initialization"):
        prep.autogen_node_sets()
        ndam_m = neurodamus_manager.MsrNeurodamusManager(conf)
        conn_m.connect_ndam2ndam(ndam_m=ndam_m)

        conf.merge_without_priority({"DT": ndam_m.dt()})

        logging.info(str(conf))
        logging.info(conf.dt_info())

        rss = []  # Memory tracking

        logging.info(f"Initializing simulations...")

        if conf.with_bloodflow:
            bf_m = bloodflow_manager.MsrBloodflowManager(
                vasculature_path=ndam_m.get_vasculature_path(),
                params=conf.bloodflow,
            )

        if conf.with_steps:
            prep.autogen_mesh(ndam_m=ndam_m, bf_m=bf_m if conf.with_bloodflow else None)
            steps_m = steps_manager.MsrStepsManager(config=conf)
            steps_m.init_sim()
            conn_m.connect_ndam2steps(ndam_m=ndam_m, steps_m=steps_m)
            if conf.with_steps:
                conn_m.connect_bf2steps(bf_m=bf_m, steps_m=steps_m)

        if conf.with_metabolism:
            metab_m = metabolism_manager.MsrMetabolismManager(
                config=conf,
                main=Main,
                prnt=prnt,
                neuron_pop_name=ndam_m.neuron_manager.population_name,
            )

    log_stage("===============================================")
    log_stage("Running the selected solvers ...")

    # i_* is the number of time steps of that particular simulator
    i_ndam, i_metab = 0, 0
    for t in ProgressBar(int(conf.msr_sim_end / (conf.DT * conf.mr_ndts)))(
        utils.timesteps(conf.msr_sim_end, conf.DT * conf.mr_ndts)
    ):
        i_ndam += conf.mr_ndts
        with timeit(name="main_loop"):
            with timeit(name="neurodamus_solver"):
                ndam_m.ndamus.solve(t)

            if conf.with_steps and i_ndam % conf.steps_ndts == 0:
                with timeit(name="steps_loop"):
                    log_stage("steps loop")
                    with timeit(name="steps_solver"):
                        steps_m.sim.run(t / 1000)  # ms to sec

                    with timeit(name="neurodamus_2_steps"):
                        conn_m.ndam2steps_sync(
                            ndam_m=ndam_m,
                            steps_m=steps_m,
                            specs=conf.steps.Volsys.species,
                            DT=conf.steps_ndts * conf.DT,
                        )

            if conf.with_bloodflow and i_ndam % conf.bloodflow_ndts == 0:
                with timeit(name="bf_loop"):
                    log_stage("bf loop")
                    conn_m.ndam2bloodflow_sync(ndam_m=ndam_m, bf_m=bf_m)
                    bf_m.update_static_flow()

            if conf.with_metabolism and i_ndam % conf.metabolism_ndts == 0:
                with timeit(name="metabolism_loop"):
                    log_stage("metab loop")
                    with timeit(name="neurodamus_2_metabolism"):
                        conn_m.ndam2metab_sync(ndam_m=ndam_m, metab_m=metab_m)

                    if conf.with_steps:
                        with timeit(name="steps_2_metabolism"):
                            conn_m.steps2metab_sync(steps_m=steps_m, metab_m=metab_m)

                    if conf.with_bloodflow and conf.with_steps:
                        with timeit(name="bloodflow_2_metabolism"):
                            conn_m.bloodflow2metab_sync(bf_m=bf_m, metab_m=metab_m)

                    with timeit(name="solve_metabolism"):
                        failed_cells = metab_m.advance(
                            ndam_m.ncs,
                            i_metab,
                            conf.metabolism_ndts * conf.DT,
                        )

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

    ndam_m.ndamus.spike2file(str(conf.results_path / "out.dat"))
    TimerManager.timeit_show_stats()
    comm.Barrier()


if __name__ == "__main__":
    main()
