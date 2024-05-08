"""This module provides an API to instantiate, initialize,
and run simulations. It manipulates the "manager" classes
and orchestrate the different models and pass data between
them to perform the simulation
"""

import functools
import logging

from . import utils


def _run_once(f):
    """Decorator to ensure a function is called only once."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class MsrSimulation:
    def __init__(self, base_path=None):
        self._base_path = base_path

    def main(self):
        self.compute()

    @_run_once
    def warmup(self):
        """Instantiate the simulators in the proper and sensitive order"""
        logging.info("warmup simulators...")
        # this needs to be before "import neurodamus" and before MPI4PY otherwise mpi hangs

        from neuron import h

        h.nrnmpi_init()

        import neurodamus  # noqa: F401

        # steps_manager should go before preprocessor until https://github.com/CNS-OIST/HBP_STEPS/issues/1166 is solved
        from multiscale_run import (  # noqa: F401
            bloodflow_manager,
            connection_manager,
            metabolism_manager,
            neurodamus_manager,
            preprocessor,
            steps_manager,
        )

    @_run_once
    def configure(self):
        self.warmup()
        logging.info("configure simulators")

        from multiscale_run import config, connection_manager, preprocessor

        self.conf = config.MsrConfig(self._base_path)
        self.conf.check()
        self.prep = preprocessor.MsrPreprocessor(self.conf)

        self.managers = {}
        self.conn_m = connection_manager.MsrConnectionManager(
            config=self.conf, managers=self.managers
        )

    @_run_once
    def initialize(self):
        self.configure()
        logging.info("Initialize simulation")
        if self.conf.is_metabolism_active():
            from diffeqpy import de
            from julia import Main as JMain
        from neurodamus.utils.timeit import timeit

        from multiscale_run import (
            bloodflow_manager,
            metabolism_manager,
            neurodamus_manager,
            reporter,
            steps_manager,
        )

        with timeit(name="initialization"):
            self.prep.autogen_node_sets()
            self.neurodamus_manager = neurodamus_manager.MsrNeurodamusManager(self.conf)
            self.managers["neurodamus"] = self.neurodamus_manager
            failed_cells = [None] * len(self.neurodamus_manager.ncs)

            # this is here because neurodamus is in charge of setting the log level
            logging.info(str(self.conf.multiscale_run))
            logging.info(self.conf.dt_info())

            logging.info("Initializing simulations...")
            self.rep = reporter.MsrReporter(
                config=self.conf, gids=self.neurodamus_manager.gids
            )

            self.conn_m.connect_neurodamus2neurodamus()

            self.managers["bloodflow"] = None
            if self.conf.is_bloodflow_active():
                self.managers["bloodflow"] = bloodflow_manager.MsrBloodflowManager(
                    vasculature_path=self.neurodamus_manager.get_vasculature_path(),
                    parameters=self.conf.multiscale_run.bloodflow,
                )
            # the communication between bf and ndam is mediated by the steps mesh. We need the
            # connect calls in this case

            self.managers["steps"] = None
            if self.conf.is_steps_active() or (
                self.conf.is_bloodflow_active() and self.conf.is_metabolism_active()
            ):
                self.prep.autogen_mesh(
                    ndam_m=self.neurodamus_manager,
                    bf_m=self.managers["bloodflow"],
                )
                self.prep.check_mesh()
                self.managers["steps"] = steps_manager.MsrStepsManager(config=self.conf)
                self.managers["steps"].init_sim()
                self.conn_m.connect_neurodamus2steps()
                if self.conf.is_bloodflow_active():
                    self.conn_m.connect_bloodflow2steps()

            self.managers["metabolism"] = None
            if self.conf.is_metabolism_active():
                self.managers["metabolism"] = metabolism_manager.MsrMetabolismManager(
                    config=self.conf,
                    main=JMain,
                    neuron_pop_name=self.managers[
                        "neurodamus"
                    ].neuron_manager.population_name,
                    gids=self.neurodamus_manager.gids,
                )

            # sync bloodflow to give initial values to metabolism
            self.conn_m.process_syncs(sync_event="before_bloodflow_advance")
            if self.conf.is_bloodflow_active():
                self.managers["bloodflow"].update_static_flow()

            # apply all the connections for initialization
            self.rep.record(
                idt=0,
                manager_name="metabolism",
                managers=self.managers,
                when="before_sync",
            )

            self.conn_m.process_syncs(sync_event="after_metabolism_advance")

            # remove cells that already fail
            if self.conf.is_metabolism_active():
                self.managers["metabolism"].check_inputs(failed_cells=failed_cells)
            self.conn_m.remove_gids(failed_cells=failed_cells)

            self.rep.record(
                idt=0,
                manager_name="metabolism",
                managers=self.managers,
                when="after_sync",
            )

    @_run_once
    def compute(self):
        """Perform the actual simulation"""
        self.initialize()
        logging.info("Starting simulation")

        # Memory tracking
        import psutil
        from neurodamus.core import ProgressBarRank0 as ProgressBar
        from neurodamus.utils.logging import log_stage
        from neurodamus.utils.timeit import TimerManager, timeit

        from multiscale_run import utils as msr_utils

        log_stage("===============================================")
        log_stage("Running the selected solvers ...")

        self.rss = []  # Memory tracking

        # i_* is the number of time steps of that particular simulator
        i_ndam, i_metab = 0, 0
        time_step_n = int(self.conf.run.tstop / (self.conf.multiscale_run_dt))
        time_steps = msr_utils.timesteps(
            self.conf.run.tstop, self.conf.multiscale_run_dt
        )
        for t in ProgressBar(time_step_n)(time_steps):
            i_ndam += self.conf.multiscale_run.ndts
            with timeit(name="main_loop"):
                failed_cells = [None] * len(self.neurodamus_manager.ncs)
                with timeit(name="neurodamus_advance"):
                    self.neurodamus_manager.ndamus.solve(t)

                if (
                    self.conf.is_steps_active()
                    and i_ndam % self.conf.multiscale_run.steps.ndts == 0
                ):
                    with timeit(name="steps_advance"):
                        self.managers["steps"].sim.run(t / 1000)  # ms to sec

                    self.conn_m.process_syncs(sync_event="after_steps_advance")

                if (
                    self.conf.is_bloodflow_active()
                    and i_ndam % self.conf.multiscale_run.bloodflow.ndts == 0
                ):
                    # bloodflow has an implicit scheme because it works with quasi-static assumption
                    self.conn_m.process_syncs(sync_event="before_bloodflow_advance")
                    with timeit(name="bloodflow_advance"):
                        self.managers["bloodflow"].update_static_flow()

                if (
                    self.conf.is_metabolism_active()
                    and i_ndam % self.conf.multiscale_run.metabolism.ndts == 0
                ):
                    with timeit(name="metabolism_advance"):
                        self.managers["metabolism"].check_inputs(
                            failed_cells=failed_cells
                        )
                        self.managers["metabolism"].advance(
                            i_metab=i_metab, failed_cells=failed_cells
                        )
                    i_metab += 1
                    utils.comm().Barrier()

                    self.conn_m.remove_gids(failed_cells=failed_cells)

                    self.rep.record(
                        idt=i_metab,
                        manager_name="metabolism",
                        managers=self.managers,
                        when="before_sync",
                    )

                    self.conn_m.process_syncs(sync_event="after_metabolism_advance")

                    self.rep.record(
                        idt=i_metab,
                        manager_name="metabolism",
                        managers=self.managers,
                        when="after_sync",
                    )

                self.rss.append(
                    psutil.Process().memory_info().rss / (1024**2)
                )  # memory consumption in MB

        self.neurodamus_manager.ndamus.sonata_spikes()
        TimerManager.timeit_show_stats()
        utils.comm().Barrier()
        self.neurodamus_manager.ndamus._touch_file(
            self.neurodamus_manager.ndamus._success_file
        )


def main():
    logging.basicConfig(level=logging.INFO)
    sim = MsrSimulation()
    sim.main()


if __name__ == "__main__":
    main()
