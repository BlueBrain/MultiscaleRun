import enum
import logging

import numpy as np
import pandas as pd
from bluepysnap import Circuit

from . import config, utils


class MsrMetabManagerException(Exception):
    """Generic Metabolism Manager Exception"""


class MsrExcludeNeuronException(Exception):
    """This error should be recoverable. We just want to kick the neuron out
    of the simulation because it is misbehaving
    """


class MsrAbortSimulationException(Exception):
    """This error should not be recoverable. Something went very wrong and continuing the simulation is meaningless"""


class MsrMetabolismParam(enum.Enum):
    ina_density = 0
    ik_density = 1
    mito_scale = 2
    bloodflow_Fin = 3
    bloodflow_Fout = 4
    bloodflow_vol = 5


class MsrMetabolismManager:
    """Wrapper to manage the metabolism julia model"""

    errors = {
        "abort_simulation": MsrAbortSimulationException,
        "exclude_neuron": MsrExcludeNeuronException,
    }

    def __init__(self, config, main, neuron_pop_name: str, gids: list[int]):
        """Initialize the MsrMetabolismManager.

        Args:
            config: The configuration for the metabolism manager.
            main: An instance of the main class.
            neuron_pop_name: The name of the neuron population.
            gids: list of cells to reset.
        """
        self.config = config
        self.load_metabolism_data(main)
        self.gen_metabolism_model(main)
        self.vm = None  # read/write values for metab
        self.parameters = None  # read values for metab
        self.tspan_m = (-1, -1)
        self.neuro_df = Circuit(
            str(self.config.config_path.parent / self.config.network)
        ).nodes[neuron_pop_name]
        self.reset(gids)

    def get_error(self, key: str):
        try:
            return self.errors[key]
        except KeyError as e:
            raise config.MsrConfigException(
                f"The error `{key}` does not exist in the config file: '{self.config.config_path}' or its logic is not implemented in metabolism. Possible values: {', '.join(self.errors)}"
            ) from e

    @property
    def ngids(self):
        """Gid number"""
        return self.parameters.shape[0]

    def set_parameters_idxs(self, vals: list[float], idxs: list[int]):
        """Set one or more parameters slices equal to the vals list"""
        for idx in idxs:
            self.parameters[:, idx] = vals

    def set_vm_idxs(self, vals, idxs):
        """Set one or more vm slices equal to the vals list"""
        for idx in idxs:
            self.vm[:, idx] = vals

    def get_parameters_idx(self, idx):
        """Get parameters slice"""
        return self.parameters[:, idx]

    def get_vm_idx(self, idx):
        """Get vm slice"""
        return self.vm[:, idx]

    def alive_gids(self):
        """Convenience function to report which gids are still alive.

        All the gids present are still alive.
        """
        return [1] * self.parameters.shape[0]

    @utils.logs_decorator
    def load_metabolism_data(self, main):
        """Load metabolism data and parameters from Julia scripts.

        This method loads metabolism data and parameters from Julia scripts if the metabolism type is "main".

        Args:
            main: An instance of the main class.
        """
        # includes
        cmd = (
            "\n".join(
                [
                    f'include("{item}")'
                    for item in self.config.multiscale_run.metabolism.model.includes
                ]
            )
            + "\n"
            + "\n".join(
                [
                    f"{k} = {v}"
                    for k, v in self.config.multiscale_run.metabolism.model.constants.items()
                ]
            )
        )
        main.eval(cmd)

    @utils.logs_decorator
    def gen_metabolism_model(self, main):
        """Generate the metabolism model from Julia code.

        This method generates the metabolism model using Julia code.

        Args:
            main: An instance of the main class.

        Returns:
            None
        """
        with open(str(self.config.multiscale_run.metabolism.julia_code_path), "r") as f:
            julia_code = f.read()
        self.model = main.eval(julia_code)

    @utils.logs_decorator
    def _advance_gid(self, igid: int, i_metab: int, failed_cells: list[str]):
        """Advance metabolism simulation for gid: gids[igid].

        Args:
            igid: Index of the gid.
            i_metab: metabolism, time step counter.
            failed_cells: List of errors for the failed cells. Cells that are alive have `None` as value here.

        Raises:
            MsrMetabManagerException: If sol is None.
        """
        from diffeqpy import de

        metab_dt = self.config.metabolism_dt
        tspan_m = (
            1e-3 * float(i_metab) * metab_dt,
            1e-3 * (float(i_metab) + 1.0) * metab_dt,
        )

        prob = de.ODEProblem(
            self.model, self.vm[igid, :], tspan_m, self.parameters[igid, :]
        )
        try:
            logging.info("   solve ODE problem")
            sol = de.solve(
                prob,
                de.Rosenbrock23(autodiff=False),
                **self.config.multiscale_run.metabolism.solver_kwargs,
            )
            logging.info("   /solve ODE problem")
            if str(sol.retcode) != "<PyCall.jlwrap Success>":
                utils.rank_print(f" !!! sol.retcode: {str(sol.retcode)}")

        except Exception as e:
            failed_cells[igid] = f"solver failed: {str(sol.retcode)}"
            error_solver = e

        if sol is None:
            raise MsrMetabManagerException(f"sol is None: {error_solver}")

        self.vm[igid, :] = sol.u[-1]

    @utils.logs_decorator
    def advance(self, i_metab: int, failed_cells: list) -> None:
        """Advance metabolism simulation

        Already failed cells are skipped.

        Args:
            i_metab: metabolism, time step counter.
            failed_cells: List of errors for the failed cells. Cells that are alive have `None` as value here.
        """
        for igid, err in enumerate(failed_cells):
            if err is not None:
                continue

            self._advance_gid(igid=igid, i_metab=i_metab, failed_cells=failed_cells)

    def _get_GLY_a_and_mito_vol_frac(self, c_gid: int):
        """Get glycogen (GLY_a) and mitochondrial volume fraction.

        This method calculates glycogen (GLY_a) and mitochondrial volume fraction for a given neuron based on its layer.

        Args:
            c_gid: The Global ID of the neuron.

        Returns:
            a tuple (glycogen, mito_volume_fraction) where glycogen is
            the calculated glycogen value, and mito_volume_fraction is
            the calculated mitochondrial volume fraction.
        """
        # idx: layers are 1-based while python vectors are 0-based
        # c_gid: ndam is 1-based while libsonata and bluepysnap are 0-based
        idx = self.neuro_df.get(c_gid - 1).layer - 1

        glycogen_au = np.array(
            self.config.multiscale_run.metabolism.constants.glycogen_au
        )
        mito_volume_fraction = np.array(
            self.config.multiscale_run.metabolism.constants.mito_volume_fraction
        )
        glycogen_scaled = glycogen_au * (14.0 / max(glycogen_au))
        mito_volume_fraction_scaled = mito_volume_fraction * (
            1.0 / max(mito_volume_fraction)
        )
        return (
            glycogen_scaled[idx],
            mito_volume_fraction_scaled[idx],
        )

    @utils.logs_decorator
    def reset(self, gids: list[int]):
        """Reset the parameters and initial conditions for metabolic simulation.

        Args:
            gids: List of cells to reset.

        Returns:
            None
        """
        metab_conf = self.config.multiscale_run.metabolism
        mito_scale_idx = MsrMetabolismParam.mito_scale.value

        ngids = len(gids)
        self.vm = np.tile(
            pd.read_csv(metab_conf.u0_path, sep=",", header=None)[0].tolist(),
            (ngids, 1),
        )
        self.parameters = np.tile(metab_conf.parameters, (ngids, 1))

        # TODO this may be made more general. Atm it has low priority.
        self.parameters[:, mito_scale_idx] = [
            self._get_GLY_a_and_mito_vol_frac(c_gid)[1] for c_gid in gids
        ]

    def _check_input_for_currently_valid_gids(
        self,
        vec: np.ndarray,
        check_value_kwargs: dict,
        err,
        msg_func,
        failed_cells: list[str],
    ):
        """Check input values for every valid gid.

        Args:
            vec: Input array.
            check_value_kwargs: Keyword arguments for value checking.
            err: Error argument.
            msg_func: Message argument. It is a callable that requires the gid.
            failed_cells: List of failed cells.

        Notes:
            `check_value` can throw exceptions. `MsrExcludeNeuronException` is recoverable by marking
            the neuron as broken and kicking it out of the simulation. The others are not.
        """
        for igid in range(self.ngids):
            if failed_cells[igid] is not None:
                continue

            try:
                utils.check_value(
                    v=vec[igid], **check_value_kwargs, err=err, msg=msg_func(igid)
                )
            except MsrExcludeNeuronException as e:
                failed_cells[igid] = str(e)

    @utils.logs_decorator
    def check_inputs(self, failed_cells: list[str]) -> None:
        """Check that some values stay in the prescribed bands

        Loop over the inputs that require checking. Print in the logging for the same VIP values.
        If no checking is specified in the config file just check that the values are still proper floats.

        Args:
            failed_cells: List of errors for the failed cells. Cells that are alive have `None` as value here.
        """
        base_ck_conf = {"kwargs": {}, "response": "abort_simulation", "name": ""}

        d = self.config.multiscale_run.metabolism.checks.parameters
        checks = [
            d.get(str(idx), base_ck_conf) for idx in range(self.parameters.shape[1])
        ]

        for idx, ck_conf in enumerate(checks):

            def msg_func(igid):
                return f"parameters[{igid}, {idx}], {ck_conf['name']}"

            self._check_input_for_currently_valid_gids(
                vec=self.parameters[:, idx],
                check_value_kwargs=ck_conf["kwargs"],
                err=self.get_error(ck_conf["response"]),
                msg_func=msg_func,
                failed_cells=failed_cells,
            )
            if str(idx) in self.config.multiscale_run.metabolism.checks.parameters:
                name = ck_conf["name"]
                utils.log_stats(
                    vec=self.parameters[:, idx],
                    **ck_conf["kwargs"],
                    msg=f"parameters[:,  {idx}], {name}{' '*(16-len(name))}",
                )

        d = self.config.multiscale_run.metabolism.checks.vm
        checks = [d.get(str(idx), base_ck_conf) for idx in range(self.vm.shape[1])]

        for idx, ck_conf in enumerate(checks):

            def msg_func(igid):
                return f"vm[{igid}, {idx}], {ck_conf['name']}"

            self._check_input_for_currently_valid_gids(
                vec=self.vm[:, idx],
                check_value_kwargs=ck_conf["kwargs"],
                err=self.get_error(ck_conf["response"]),
                msg_func=msg_func,
                failed_cells=failed_cells,
            )
            if str(idx) in self.config.multiscale_run.metabolism.checks.vm:
                name = ck_conf["name"]
                utils.log_stats(
                    vec=self.vm[:, idx],
                    **ck_conf["kwargs"],
                    msg=f"    vm[:, {idx}], {name}{' '*(16-len(name))}",
                )
