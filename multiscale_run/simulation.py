"""
This module provides an API to instantiate, initialize,
and run simulations. It manipulates the "manager" classes
and orchestrate the different models and pass data between
them to perform the simulation
"""

import logging


def _run_once(f):
    """Decorator to ensure a function is called only once."""

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

        # the object MPI exists already. It is the steps one
        from mpi4py import MPI as MPI4PY  # noqa: F401

        # this goes before neurodamus for correct logging
        from multiscale_run import utils  # noqa: F401

        import neurodamus  # noqa: F401

        # steps_manager should go before preprocessor until https://github.com/CNS-OIST/HBP_STEPS/issues/1166 is solved
        from multiscale_run import (  # noqa: F401
            bloodflow_manager,
            connection_manager,
            metabolism_manager,
            neurodamus_manager,
            steps_manager,
            preprocessor,
        )

    @_run_once
    def configure(self):
        self.warmup()
        logging.info("configure simulators")
        from multiscale_run import config
        from multiscale_run import preprocessor
        from multiscale_run import connection_manager

        self.conf = config.MsrConfig(self._base_path)
        self.prep = preprocessor.MsrPreprocessor(self.conf)
        self.conn_m = connection_manager.MsrConnectionManager(self.conf)

    @_run_once
    def initialize(self):
        self.configure()
        logging.info("Initialize simulation")
        if self.conf.with_metabolism:
            from julia import Main as JMain
            from diffeqpy import de
        from neurodamus.utils.timeit import timeit
        from multiscale_run import bloodflow_manager
        from multiscale_run import metabolism_manager
        from multiscale_run import neurodamus_manager
        from multiscale_run import reporter
        from multiscale_run import steps_manager

        with timeit(name="initialization"):
            self.prep.autogen_node_sets()
            self.ndam_m = neurodamus_manager.MsrNeurodamusManager(self.conf)
            self.rep = reporter.MsrReporter(config=self.conf, gids=self.ndam_m.gids())
            self.conn_m.connect_ndam2ndam(ndam_m=self.ndam_m)

            self.conf.merge_without_priority({"DT": self.ndam_m.dt()})

            logging.info(str(self.conf))
            logging.info(self.conf.dt_info())

            logging.info("Initializing simulations...")

            if self.conf.with_bloodflow:
                self.bf_m = bloodflow_manager.MsrBloodflowManager(
                    vasculature_path=self.ndam_m.get_vasculature_path(),
                    params=self.conf.bloodflow,
                )

            if self.conf.with_steps:
                self.prep.autogen_mesh(
                    ndam_m=self.ndam_m,
                    bf_m=self.bf_m if self.conf.with_bloodflow else None,
                )
                self.steps_m = steps_manager.MsrStepsManager(config=self.conf)
                self.steps_m.init_sim()
                self.conn_m.connect_ndam2steps(ndam_m=self.ndam_m, steps_m=self.steps_m)
                if self.conf.with_bloodflow:
                    self.conn_m.connect_bf2steps(bf_m=self.bf_m, steps_m=self.steps_m)

            if self.conf.with_metabolism:
                self.metab_m = metabolism_manager.MsrMetabolismManager(
                    config=self.conf,
                    main=JMain,
                    neuron_pop_name=self.ndam_m.neuron_manager.population_name,
                )

    @_run_once
    def compute(self):
        """Perform the actual simulation"""
        self.initialize()
        logging.info("Starting simulation")
        from mpi4py import MPI as MPI4PY

        # Memory tracking
        import psutil
        from neurodamus.core import ProgressBarRank0 as ProgressBar
        from neurodamus.utils.logging import log_stage
        from neurodamus.utils.timeit import TimerManager, timeit
        from multiscale_run import utils as msr_utils

        log_stage("===============================================")
        log_stage("Running the selected solvers ...")

        comm = MPI4PY.COMM_WORLD

        self.rss = []  # Memory tracking

        # i_* is the number of time steps of that particular simulator
        i_ndam, i_metab = 0, 0
        for t in ProgressBar(
            int(self.conf.msr_sim_end / (self.conf.DT * self.conf.msr_ndts))
        )(
            msr_utils.timesteps(
                self.conf.msr_sim_end, self.conf.DT * self.conf.msr_ndts
            )
        ):
            i_ndam += self.conf.msr_ndts
            with timeit(name="main_loop"):
                with timeit(name="neurodamus_solver"):
                    self.ndam_m.ndamus.solve(t)

                if self.conf.with_steps and i_ndam % self.conf.steps_ndts == 0:
                    with timeit(name="steps_loop"):
                        log_stage("steps loop")
                        with timeit(name="steps_solver"):
                            self.steps_m.sim.run(t / 1000)  # ms to sec

                        with timeit(name="neurodamus_2_steps"):
                            self.conn_m.ndam2steps_sync(
                                ndam_m=self.ndam_m,
                                steps_m=self.steps_m,
                                specs=self.conf.steps.Volsys.species,
                                DT=self.conf.steps_ndts * self.conf.DT,
                            )

                if self.conf.with_bloodflow and i_ndam % self.conf.bloodflow_ndts == 0:
                    with timeit(name="bf_loop"):
                        log_stage("bf loop")
                        self.conn_m.ndam2bloodflow_sync(
                            ndam_m=self.ndam_m, bf_m=self.bf_m
                        )
                        self.bf_m.update_static_flow()

                if (
                    self.conf.with_metabolism
                    and i_ndam % self.conf.metabolism_ndts == 0
                ):
                    with timeit(name="metabolism_loop"):
                        log_stage("metab loop")
                        # gids may change during the sim. We need to get the updated version
                        lgids = self.ndam_m.gids()
                        with timeit(name="neurodamus_2_metabolism"):
                            self.conn_m.ndam2metab_sync(
                                ndam_m=self.ndam_m, metab_m=self.metab_m
                            )

                            units = [
                                getattr(
                                    getattr(self.conf.species, k[0]).neurodamus, k[1]
                                ).unit
                                if isinstance(k, tuple)
                                else ""
                                for k in self.metab_m.ndam_vars.keys()
                            ]
                            self.rep.set_group(
                                "ndam", self.metab_m.ndam_vars, units, lgids
                            )

                        if self.conf.with_steps:
                            with timeit(name="steps_2_metabolism"):
                                self.conn_m.steps2metab_sync(
                                    steps_m=self.steps_m, metab_m=self.metab_m
                                )
                                self.rep.set_group(
                                    "steps",
                                    self.metab_m.steps_vars,
                                    ["M"] * len(self.metab_m.steps_vars),
                                    lgids,
                                )

                        if self.conf.with_bloodflow and self.conf.with_steps:
                            with timeit(name="bloodflow_2_metabolism"):
                                self.conn_m.bloodflow2metab_sync(
                                    bf_m=self.bf_m, metab_m=self.metab_m
                                )

                                self.rep.set_group(
                                    "bf",
                                    self.metab_m.bloodflow_vars,
                                    [
                                        "um^3" if "vol" in i else "um^3/s"
                                        for i in self.metab_m.bloodflow_vars
                                    ],
                                    lgids,
                                )
                        self.rep.flush_buffer(i_metab)

                        with timeit(name="solve_metabolism"):
                            failed_cells = self.metab_m.advance(
                                self.ndam_m.ncs,
                                i_metab,
                                self.conf.metabolism_ndts * self.conf.DT,
                            )

                        self.ndam_m.remove_gids(failed_cells, self.conn_m)

                        with timeit(name="metabolism_2_neurodamus"):
                            self.conn_m.metab2ndam_sync(
                                metab_m=self.metab_m,
                                ndam_m=self.ndam_m,
                                i_metab=i_metab,
                            )

                        comm.Barrier()

                        i_metab += 1

                self.rss.append(
                    psutil.Process().memory_info().rss / (1024**2)
                )  # memory consumption in MB

        self.ndam_m.ndamus.sonata_spikes()
        TimerManager.timeit_show_stats()
        comm.Barrier()


def main():
    logging.basicConfig(level=logging.INFO)
    sim = MsrSimulation()
    sim.main()


if __name__ == "__main__":
    main()
