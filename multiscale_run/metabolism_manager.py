##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy

import logging

import numpy as np

import neurodamus
import pandas as pd
from bluepysnap import Circuit
from diffeqpy import de
from mpi4py import MPI as MPI4PY

from . import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrMetabParameterException(Exception):
    pass


class MsrMetabManagerException(Exception):
    pass


class MsrMetabParameters:
    """Class that holds the parameters for metabolism in the l order"""

    l = ["ina_density", "ik_density", "mito_scale", "bf_Fin", "bf_Fout", "bf_vol"]

    def __init__(self, metab_type):
        """Initialize MsrMetabParameters with the given metabolism type.

        Args:
            metab_type (str): The type of metabolism, e.g., "main" or another value.

        Returns:
            None
        """
        self.metab_type = metab_type
        for i in self.l:
            setattr(self, i, None)

    def __iter__(self):
        """Iterate over the parameters in the defined order.

        Yields:
            The values of the parameters in the defined order.

        Returns:
            None
        """
        for i in self.l:
            yield getattr(self, i)

    def is_valid(self):
        """Check the validity of parameter values and their data types.

        This method validates the parameter values by checking if they are not None, are of float data type, and not NaN or Inf. It also checks specific validity conditions for "bf_vol" when the metabolism type is "main".

        Raises:
            MsrMetabParameterException: If any of the parameter values are not valid.

        Returns:
            None
        """
        for k in self.l:
            v = getattr(self, k)
            msg = f"{k}: {v} is "
            if v is None:
                raise MsrMetabParameterException(msg + "None")
            try:
                float(v)
            except:
                raise MsrMetabParameterException(msg + "not float")

            if np.isnan(v):
                raise MsrMetabParameterException(msg + "NaN")
            if np.isinf(v):
                raise MsrMetabParameterException(msg + "Inf")

        if self.metab_type == "main" and self.bf_vol <= 0:
            raise MsrMetabParameterException(f"bf_vol ({self.bf_vol}) <= 0")


class MsrMetabolismManager:
    """Wrapper to manage the metabolism julia model"""

    def __init__(self, config, main, prnt, neuron_pop_name):
        """Initialize the MsrMetabolismManager.

        Args:
            config: The configuration for the metabolism manager.
            main: An instance of the main class.
            prnt: An instance for printing data.
            neuron_pop_name: The name of the neuron population.

        Returns:
            None
        """
        self.config = config
        self.u0 = pd.read_csv(self.config.metabolism.u0_path, sep=",", header=None)[
            0
        ].tolist()
        self.load_metabolism_data(main)
        self.gen_metabolism_model(main)
        self.vm = {}
        self.tspan_m = (-1, -1)
        self.ndam_vars = {}
        self.steps_vars = {}
        self.bloodflow_vars = {}
        self.failed_cells = {}
        self.prnt = prnt
        self.neuro_df = Circuit(config.circuit_path).nodes[neuron_pop_name]

    @utils.logs_decorator
    def load_metabolism_data(self, main):
        """Load metabolism data and parameters from Julia scripts.

        This method loads metabolism data and parameters from Julia scripts if the metabolism type is "main".

        Args:
            main: An instance of the main class.

        Returns:
            None
        """

        self.ATDPtot_n = 1.4449961078157665

        if self.config.metabolism.type != "main":
            return None

        main.eval(
            """
        modeldirname = "/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/metabolism_unit_models/"

        include(string(modeldirname,"FINAL_CLEAN/data_model_full/u0_db_refined_selected_oct2021.jl"))

        pardirname = string(modeldirname,"optimiz_unit/enzymes/enzymes_preBigg/COMBO/parameters_GLYCOGEN_cleaned4bigg/")

        include(string(pardirname,"general_parameters.jl"))
        include(string(pardirname,"ephys_parameters.jl"))
        include(string(pardirname,"bf_input.jl"))
        include(string(pardirname,"generalisations.jl")) # Jolivet NADH shuttles, resp
        include(string(pardirname,"GLC_transport.jl"))
        include(string(pardirname,"GLYCOLYSIS.jl"))
        include(string(pardirname,"glycogen.jl"))

        include(string(pardirname,"creatine.jl"))

        include(string(pardirname,"ATDMP.jl"))

        include(string(pardirname,"pyrTrCytoMito.jl"))
        include(string(pardirname,"lactate.jl"))
        include(string(pardirname,"TCA.jl"))

        include(string(pardirname,"ETC.jl"))

        include(string(pardirname,"PPP_n.jl"))
        include(string(pardirname,"PPP_a.jl"))
        include(string(pardirname,"gshgssg.jl"))

        include(string(pardirname,"MAS.jl"))
        include(string(pardirname,"gltgln.jl"))
        include(string(pardirname,"pyrCarb.jl"))
        include(string(pardirname,"ketones.jl"))

        # for NEmodulation
        xNEmod = 0.025 # 0.1 #0.00011
        KdNEmod = 3.0e-4 # 3.6e-5  # 3.0e-4 #

        Iinj = 0.0
        synInput = 0.0

        """
        )

        return None

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
    def _advance_gid(self, c_gid, i_metab, param):
        """Advance metabolism simulation for a specific GID (Global ID).

        This method advances the metabolism simulation for a specific GID using the provided parameters.

        Args:
            c_gid: The Global ID of the neuron.
            i_metab: The current metabolism iteration.
            param: An instance of MsrMetabParameters containing metabolism parameters.

        Returns:
            None
        """

        prob = de.ODEProblem(self.model, self.vm[c_gid], self.tspan_m, list(param))
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
            comm.Abort(f"sol is None: {error_solver}")

        self.vm[c_gid] = sol.u[-1]
        self.prnt.append_to_file(
            self.config.metabolism.um_path, [c_gid, i_metab, sol.u[-1]], root=rank
        )

    @utils.logs_decorator
    def advance(self, ncs, i_metab, metab_dt):
        """Advance metabolism simulation for a list of neurons.

        This method advances the metabolism simulation for a list of neurons using the provided parameters.

        Args:
            ncs: A list of neurons to simulate.
            i_metab: The current metabolism iteration.
            metab_dt: The time step for metabolism simulation.

        Returns:
            failed_cells: A dictionary containing information about neurons with failed simulations.
        """

        self.failed_cells = {}
        for inc, nc in enumerate(ncs):
            c_gid = int(nc.CCell.gid)

            param = self.init_inputs(
                c_gid=c_gid, inc=inc, i_metab=i_metab, metab_dt=metab_dt
            )

            try:
                param.is_valid()
            except MsrMetabParameterException as e:
                self.failed_cells[c_gid] = str(e)
                logging.warning(self.failed_cells[c_gid])
                continue

            self._advance_gid(c_gid=c_gid, i_metab=i_metab, param=param)

        return self.failed_cells

    def _get_GLY_a_and_mito_vol_frac(self, c_gid):
        """Get glycogen (GLY_a) and mitochondrial volume fraction.

        This method calculates glycogen (GLY_a) and mitochondrial volume fraction for a given neuron based on its layer.

        Args:
            c_gid: The Global ID of the neuron.

        Returns:
            glycogen: The calculated glycogen value.
            mito_volume_fraction: The calculated mitochondrial volume fraction.
        """

        # idx: layers are 1-based while python vectors are 0-based
        # c_gid: ndam is 1-based while libsonata and bluepysnap are 0-based
        idx = self.neuro_df.get(c_gid - 1).layer - 1

        glycogen_au = np.array(self.config.metabolism.glycogen_au)
        mito_volume_fraction = np.array(self.config.metabolism.mito_volume_fraction)
        glycogen_scaled = glycogen_au * (14.0 / max(glycogen_au))
        mito_volume_fraction_scaled = mito_volume_fraction * (
            1.0 / max(mito_volume_fraction)
        )
        return (
            glycogen_scaled[idx],
            mito_volume_fraction_scaled[idx],
        )

    @utils.logs_decorator
    def init_inputs(self, c_gid, inc, i_metab, metab_dt):
        """Initialize parameters for metabolism simulation.

        This method initializes parameters for metabolism simulation based on the provided neuron's characteristics.

        Args:
            c_gid: The Global ID of the neuron.
            inc: An index for the neuron.
            i_metab: The current metabolism iteration.
            metab_dt: The time step for metabolism simulation.

        Returns:
            param: An instance of MsrMetabParameters containing initialized parameters.
        """

        param = MsrMetabParameters(metab_type=self.config.metabolism.type)

        kis_mean = self.ndam_vars[("K", "conci")][inc]
        atpi_mean = self.ndam_vars[("atp", "conci")][inc]
        nais_mean = self.ndam_vars[("Na", "conci")][inc]
        param.ina_density = self.ndam_vars[("Na", "curr")][inc]
        param.ik_density = self.ndam_vars[("K", "curr")][inc]

        GLY_a, param.mito_scale = self._get_GLY_a_and_mito_vol_frac(c_gid)

        self.tspan_m = (
            1e-3 * float(i_metab) * metab_dt,
            1e-3 * (float(i_metab) + 1.0) * metab_dt,
        )  # tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0))

        # um[(0, c_gid)][127] = GLY_a
        if c_gid not in self.vm:
            if i_metab != 0:
                raise MsrMetabManagerException(
                    f"i_metab = {i_metab} (not 0) and neuron id: {c_gid} not registered in vm.\n vm keys: {self.vm.keys()}"
                )
            self.vm[c_gid] = self.u0

        # vm[161] = vm[161] - outs_r_glu.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)
        # vm[165] = vm[165] - outs_r_gaba.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)

        idx_atpn = self.config.metabolism.vm_idxs.atpn
        idx_adpn = self.config.metabolism.vm_idxs.adpn
        idx_nai = self.config.metabolism.vm_idxs.nai
        idx_ko = self.config.metabolism.vm_idxs.ko

        self.vm[c_gid][idx_atpn] = 0.5 * 1.384727988648391 + 0.5 * atpi_mean

        self.vm[c_gid][idx_adpn] = 0.5 * 1.384727988648391 / 2 * (
            -0.92
            + np.sqrt(0.92 * 0.92 + 4 * 0.92 * (self.ATDPtot_n / 1.384727988648391 - 1))
        ) + 0.5 * atpi_mean / 2 * (
            -0.92 + np.sqrt(0.92 * 0.92 + 4 * 0.92 * (self.ATDPtot_n / atpi_mean - 1))
        )

        self.vm[c_gid][idx_nai] = nais_mean

        # KKconc is computed in 3 different simulators: ndam, metab, steps. We calculate all of them to do computations later

        k_steps_name = self.config.species.K.steps.name
        self.vm[c_gid][idx_ko] = (
            self.steps_vars[k_steps_name][inc]
            if k_steps_name in self.steps_vars
            else 3.0 - 1.33 * (kis_mean - 140.0)
        )

        param.bf_Fout = param.bf_Fin = 0.0001
        if "Fin" in self.bloodflow_vars:
            param.bf_Fout = param.bf_Fin = self.bloodflow_vars["Fin"][inc]
        param.bf_vol = 0.023
        if "vol" in self.bloodflow_vars:
            param.bf_vol = self.bloodflow_vars["vol"][inc]
        l = [
            *[
                f"vm[{c_gid}][{i}]: {utils.ppf(self.vm[c_gid][i])}"
                for i in self.config.metabolism.vm_idxs.values()
            ],
            f"kis_mean[{c_gid}]: {utils.ppf(kis_mean)}",
            f"bf_Fin: {param.bf_Fin}",
            f"bf_vol: {param.bf_vol}",
        ]
        l = "\n".join(l)
        logging.info(f"metab VIP vars:\n{l}")

        # Katta: "Polina suggested the following asserts as rule of thumb. In this way we detect
        # macro-problems like K+ accumulation faster. For now the additional computation is minimal.
        # Improvements are possible if needed."

        assert 0.25 <= self.vm[c_gid][idx_atpn] <= 2.5, self.vm[c_gid][idx_atpn]
        assert 5 <= self.vm[c_gid][idx_nai] <= 30, self.vm[c_gid][
            idx_nai
        ]  # usually around 10
        assert 1 <= self.vm[c_gid][idx_ko] <= 10, self.vm[c_gid][idx_ko]
        assert 100 <= kis_mean <= 160, kis_mean

        # 2.2 should coincide with the BC METypePath field & with u0_path
        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model
        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model

        self.prnt.append_to_file(
            self.config.metabolism.param_path,
            [c_gid, i_metab, *list(param)],
            root=rank,
        )

        return param
