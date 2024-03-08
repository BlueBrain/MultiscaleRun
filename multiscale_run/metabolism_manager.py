import logging

from bluepysnap import Circuit
import numpy as np
import pandas as pd

from . import utils


class MsrMetabManagerException(Exception):
    """Generic Metabolism Manager Exception"""


class MsrKickNeuronException(Exception):
    """This error should be recoverable. We just want to kick the neuron out
    of the simulation because it is misbehaving"""


class MsrMetabParameters:
    """Class that holds the parameters for metabolism in the l order"""

    l = ["ina_density", "ik_density", "mito_scale", "bf_Fin", "bf_Fout", "bf_vol"]

    def __init__(self, metab_type: str):
        """Initialize MsrMetabParameters with the given metabolism type.

        Args:
            metab_type: The type of metabolism, e.g., "main" or another value.
        """
        self.metab_type = metab_type
        for i in self.l:
            setattr(self, i, None)

    def __iter__(self):
        """Iterate over the parameters in the defined order.

        Yields:
            The values of the parameters in the defined order.
        """
        for i in self.l:
            yield getattr(self, i)


class MsrMetabolismManager:
    """Wrapper to manage the metabolism julia model"""

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
        self.vm = {}  # read/write values for metab
        self.params = {}  # read values for metab
        self.tspan_m = (-1, -1)
        self.failed_cells = {}
        self.neuro_df = Circuit(config.circuit_path).nodes[neuron_pop_name]
        self.reset(gids)

    @utils.logs_decorator
    def load_metabolism_data(self, main):
        """Load metabolism data and parameters from Julia scripts.

        This method loads metabolism data and parameters from Julia scripts if the metabolism type is "main".

        Args:
            main: An instance of the main class.
        """
        if self.config.metabolism.type != "main":
            return

        # includes
        cmd = (
            "\n".join(
                [f'include("{item}")' for item in self.config.metabolism.model.includes]
            )
            + "\n"
            + "\n".join(
                [
                    f"{k} = {v}"
                    for k, v in self.config.metabolism.model.constants.items()
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
        with open(str(self.config.metabolism.julia_code_path), "r") as f:
            julia_code = f.read()
        self.model = main.eval(julia_code)

    @utils.logs_decorator
    def _advance_gid(self, c_gid: int):
        """Advance metabolism simulation for a specific GID (Global ID).

        This method advances the metabolism simulation for a specific GID.

        Args:
            c_gid: The Global ID of the neuron.

        Returns:
            None
        """

        from diffeqpy import de

        prob = de.ODEProblem(
            self.model, self.vm[c_gid], self.tspan_m, list(self.params[c_gid])
        )
        try:
            logging.info("   solve ODE problem")
            sol = de.solve(
                prob,
                de.Rosenbrock23(autodiff=False),
                reltol=1e-8,
                abstol=1e-8,
                saveat=1,
                maxiters=1e6,
            )
            logging.info("   /solve ODE problem")
            if str(sol.retcode) != "<PyCall.jlwrap Success>":
                utils.rank_print(f" !!! sol.retcode: {str(sol.retcode)}")

        except Exception as e:
            self.failed_cells[c_gid] = f"solver failed: {str(sol.retcode)}"
            error_solver = e

        if sol is None:
            utils.comm().Abort(f"sol is None: {error_solver}")

        self.vm[c_gid] = sol.u[-1]

    @utils.logs_decorator
    def advance(self, gids: list[int], i_metab: int, metab_dt: float):
        """Advance metabolism simulation for a list of neurons.

        This method advances the metabolism simulation for a list of neurons using the provided parameters.

        Args:
            gids: A list of neurons to simulate.
            i_metab: The current metabolism iteration.
            metab_dt: The time step for metabolism simulation.

        Returns:
            failed_cells: A dictionary containing information about neurons with failed simulations.
        """

        self.tspan_m = (
            1e-3 * float(i_metab) * metab_dt,
            1e-3 * (float(i_metab) + 1.0) * metab_dt,
        )

        self.failed_cells = {}
        for c_gid in gids:
            try:
                self.check_inputs(c_gid=c_gid)
            except MsrKickNeuronException as e:
                self.failed_cells[c_gid] = str(e)
                logging.warning(self.failed_cells[c_gid])
                continue
            except Exception as e:
                utils.comm().Abort(str(e))

            self._advance_gid(c_gid=c_gid)

        return self.failed_cells

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

        glycogen_au = np.array(self.config.metabolism.constants.glycogen_au)
        mito_volume_fraction = np.array(
            self.config.metabolism.constants.mito_volume_fraction
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
        u0 = pd.read_csv(self.config.metabolism.u0_path, sep=",", header=None)[
            0
        ].tolist()
        for c_gid in gids:
            self.params[c_gid] = MsrMetabParameters(
                metab_type=self.config.metabolism.type
            )
            self.params[c_gid].bf_Fin = (
                self.config.metabolism.constants.base_bloodflow.Fin
            )
            self.params[c_gid].bf_Fout = (
                self.config.metabolism.constants.base_bloodflow.Fout
            )
            self.params[c_gid].bf_vol = (
                self.config.metabolism.constants.base_bloodflow.vol
            )
            _, self.params[c_gid].mito_scale = self._get_GLY_a_and_mito_vol_frac(c_gid)

            self.vm[c_gid] = u0

    @utils.logs_decorator
    def set_bloodflow_vars(
        self, gids: list[int], bloodflow_vars: dict[str, list[float]]
    ):
        """Set blood flow variables.

        This method overrules the previous values in 'params'.

        Args:
            gids: gids of the cells.
            bloodflow_vars: Dictionary containing blood flow variables.
                Ex: {"Fin": 0.001, "vol": 0.023}. Fout is the same as Fin so it
                is not included.
        """
        for inc, c_gid in enumerate(gids):
            # set params
            self.params[c_gid].bf_Fin = bloodflow_vars["Fin"][inc]
            # for bloodflow, Fin is always == Fout. No need to transfer 2 numbers
            self.params[c_gid].bf_Fout = bloodflow_vars["Fin"][inc]
            self.params[c_gid].bf_vol = bloodflow_vars["vol"][inc]

    @utils.logs_decorator
    def set_ndam_vars(
        self, gids: list[int], ndam_vars: dict[tuple[str, str], list[float]]
    ):
        """Set neurodamus variables

        This method overrules the previous values in 'params' and 'vm'.
        The formulae will change in the near future (Katta).

        Args:
            gids: cell gids.
            ndam_vars: Dictionary containing neurodamus variables.
        """

        vm_idxs = self.config.metabolism.vm_idxs

        for inc, c_gid in enumerate(gids):
            # set params
            self.params[c_gid].ina_density = ndam_vars[("Na", "curr")][inc]
            self.params[c_gid].ik_density = ndam_vars[("K", "curr")][inc]

            # set vm
            atpi_mean = ndam_vars[("atp", "conci")][inc]
            self.vm[c_gid][vm_idxs.atpn] = 0.5 * 1.384727988648391 + 0.5 * atpi_mean
            self.vm[c_gid][vm_idxs.adpn] = 0.5 * 1.384727988648391 / 2 * (
                -0.92
                + np.sqrt(
                    0.92 * 0.92
                    + 4
                    * 0.92
                    * (
                        self.config.metabolism.constants.ATDPtot_n / 1.384727988648391
                        - 1
                    )
                )
            ) + 0.5 * atpi_mean / 2 * (
                -0.92
                + np.sqrt(
                    0.92 * 0.92
                    + 4
                    * 0.92
                    * (self.config.metabolism.constants.ATDPtot_n / atpi_mean - 1)
                )
            )
            self.vm[c_gid][vm_idxs.nai] = ndam_vars[("Na", "conci")][inc]
            self.vm[c_gid][vm_idxs.ko] = 3.0 - 1.33 * (
                ndam_vars[("K", "conci")][inc] - 140.0
            )

    @utils.logs_decorator
    def set_steps_vars(self, gids: list[int], steps_vars: dict[str, list[float]]):
        """Set steps variables

        This method overrules the previous values in 'vm'.
        The formulae will change in the near future (Katta).

        Args:
            gids: gids of the cells.
            steps_vars: Dictionary containing step variables (typically K conc outside).
        """

        vm_idxs = self.config.metabolism.vm_idxs

        for inc, c_gid in enumerate(gids):
            k_steps_name = self.config.species.K.steps.name
            self.vm[c_gid][vm_idxs.ko] = steps_vars[k_steps_name][inc]

    @utils.logs_decorator
    def check_inputs(self, c_gid: int) -> None:
        """Check the inputs and parameters related to metabolism for a given cell group ID.

        This function logs information about the metabolism variables and performs
        value checks on them and the parameters.
        It also raises exceptions if any values are outside the specified bounds.

        Args:
            c_gid: The cell group ID for which inputs and parameters are checked.

        Returns:
            None

        Raises:
            MsrException: if the error is not recoverable.
                Example: a value is NaN.
            MsrKickNeuronException: if we recover by kicking the neuron and continuing.
                Example: there is no blood flow
        """

        # tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0))

        # um[(0, c_gid)][127] = GLY_a
        # vm[161] = vm[161] - outs_r_glu.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)
        # vm[165] = vm[165] - outs_r_gaba.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)

        # KKconc is computed in 3 different simulators: ndam, metab, steps. We calculate all of them to do computations later

        vm_idxs = self.config.metabolism.vm_idxs

        l = [
            *[
                f"vm[{c_gid}][{i}]: {utils.ppf(self.vm[c_gid][i])}"
                for i in vm_idxs.values()
            ],
            f"bf_Fin: {self.params[c_gid].bf_Fin}",
            f"bf_vol: {self.params[c_gid].bf_vol}",
        ]
        l = "\n".join(l)
        logging.info(f"metab VIP vars:\n{l}")

        # Katta: "Polina suggested the following asserts as rule of thumb. In this way we detect
        # macro-problems like K+ accumulation faster. For now the additional computation is minimal.
        # Improvements are possible if needed."

        utils.check_value(v=self.vm[c_gid][vm_idxs.atpn], lb=0.25, hb=2.5)
        utils.check_value(v=self.vm[c_gid][vm_idxs.nai], lb=5, hb=30)

        # TODO remove this if when https://bbpteam.epfl.ch/project/issues/browse/BBPP40-371 is fixed
        if self.config.with_steps:
            utils.check_value(v=self.vm[c_gid][vm_idxs.ko], lb=1, hb=10)
        # check params
        # TODO remove the following lb conditions
        utils.check_value(v=self.params[c_gid].ina_density)
        utils.check_value(v=self.params[c_gid].ik_density)
        utils.check_value(
            v=self.params[c_gid].bf_vol, leb=0.0, err=MsrKickNeuronException
        )
        utils.check_value(
            v=self.params[c_gid].bf_Fin, leb=0.0, err=MsrKickNeuronException
        )
        utils.check_value(
            v=self.params[c_gid].bf_Fout, leb=0.0, err=MsrKickNeuronException
        )
