##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy

import logging

from . import utils
config = utils.load_config()

import neurodamus
import numpy as np
import pandas as pd
from diffeqpy import de

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

class MsrMetabParameterException(Exception):
    pass

class MsrMetabParameters:
    """ Class that holds the parameters for metabolism in the l order """
    l = ["ina_density", "ik_density", "mito_scale", "bf_Fin", "bf_Fout", "bf_vol"]

    def __init__(self):
        for i in self.l:
            setattr(self, i, None)

    def __iter__(self):
        for i in self.l:
            yield getattr(self, i)

    def is_valid(self):

        for k, v in self.__dict__.items():
            e = MsrMetabParameterException(f"{k}: {v} is invalid")
            if v is None:
                raise e
            try:
                float(v)
            except:
                raise e
            
            if np.isnan(v) or np.isinf(v):
                raise e
        
        if config.metabolism_type == 'main' and self.bf_vol <= 0:
            raise MsrMetabParameterException(f"bf_vol ({self.bf_vol}) <= 0")
        

class MsrMetabolismManager:
    """Wrapper to manage the metabolism julia model"""

    def __init__(self, u0_file, main, prnt):       
        self.u0 = pd.read_csv(u0_file, sep=",", header=None)[0].tolist()
        self.load_metabolism_data(main)
        self.gen_metabolism_model(main)
        self.um = {}
        self.vm = []
        self.tspan_m = (-1, -1)
        self.ndam_vars = {}
        self.steps_vars = {}
        self.bloodflow_vars = {}
        self.failed_cells = {}
        self.prnt = prnt

    @utils.logs_decorator
    def load_metabolism_data(self, main):
        self.ATDPtot_n = 1.4449961078157665

        if config.metabolism_type != 'main':
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
        """import jl metabolism diff eq system code to py"""
        with open(config.julia_code_file, "r") as f:
            julia_code = f.read()
        self.model = main.eval(julia_code)

    @utils.logs_decorator
    def _advance_gid(self, c_gid, i_metab, param):
        """advance metabolism simulation for gid"""

        prob = de.ODEProblem(self.model, self.vm, self.tspan_m, list(param))
        try:
            sol = de.solve(
                prob,
                de.Rosenbrock23(autodiff=False),
                reltol=1e-8,
                abstol=1e-8,
                saveat=1,
                maxiters=1e6,
            )
            if str(sol.retcode) != "<PyCall.jlwrap Success>":
                utils.rank_print(f" !!! sol.retcode: {str(sol.retcode)}")

        except Exception as e:
            self.failed_cells[c_gid] = f"solver failed: {str(sol.retcode)}"
            error_solver = e

        if sol is None:
            raise error_solver

        self.um[(i_metab + 1, c_gid)] = sol.u[-1]
        self.prnt.append_to_file(config.um_out_file, [c_gid, i_metab, sol.u[-1]], rank)
        self.um[(i_metab, c_gid)] = None

    @utils.logs_decorator
    def advance(self, ncs, i_metab, metab_dt):
        """advance metabolism simulation"""

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
                continue


            self._advance_gid(c_gid=c_gid, i_metab=i_metab, param=param)

        return self.failed_cells

    @utils.logs_decorator
    def init_inputs(self, c_gid, inc, i_metab, metab_dt):
        """set params for metabolism"""

        param = MsrMetabParameters()

        kis_mean = self.ndam_vars["kis_mean"][inc]
        atpi_mean = self.ndam_vars["atpi_mean"][inc]
        nais_mean = self.ndam_vars["nais_mean"][inc]
        param.ina_density = self.ndam_vars["ina_density"][inc]
        param.ik_density = self.ndam_vars["ik_density"][inc]

        GLY_a, param.mito_scale = config.get_GLY_a_and_mito_vol_frac(c_gid)
        self.tspan_m = (
            1e-3 * float(i_metab) * metab_dt,
            1e-3 * (float(i_metab) + 1.0) * metab_dt,
        )  # tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0))
        self.um[(0, c_gid)] = self.u0

        # um[(0, c_gid)][127] = GLY_a
        self.vm = self.um[(i_metab, c_gid)]

        # vm[161] = vm[161] - outs_r_glu.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)
        # vm[165] = vm[165] - outs_r_gaba.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)

        idx_atpn = config.metab_vm_indexes["atpn"]
        idx_adpn = config.metab_vm_indexes["adpn"]
        idx_nai = config.metab_vm_indexes["nai"]
        idx_ko = config.metab_vm_indexes["ko"]

        
        
        self.vm[idx_atpn] = (
            0.5 * 1.384727988648391 + 0.5 * atpi_mean
        )  

        self.vm[idx_adpn] = 0.5 * 1.384727988648391 / 2 * (
            -0.92
            + np.sqrt(0.92 * 0.92 + 4 * 0.92 * (self.ATDPtot_n / 1.384727988648391 - 1))
        ) + 0.5 * atpi_mean / 2 * (
            -0.92 + np.sqrt(0.92 * 0.92 + 4 * 0.92 * (self.ATDPtot_n / atpi_mean - 1))
        )

        self.vm[idx_nai] = nais_mean

        # KKconc is computed in 3 different simulators: ndam, metab, steps. We calculate all of them to do computations later
        self.vm[idx_ko] = (
            self.steps_vars[config.KK.name][inc]
            if config.KK.name in self.steps_vars
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
                f"vm[{i}]: {utils.ppf(self.vm[i])}"
                for i in config.metab_vm_indexes.values()
            ],
            f"um[(0, {c_gid})][{str(idx_ko)}]: {utils.ppf(self.um[(0, c_gid)][idx_ko])}",
            f"kis_mean[c_gid]: {utils.ppf(kis_mean)}",
            f"bf_Fin: {param.bf_Fin}",
            f"bf_vol: {param.bf_vol}",
            
        ]
        l = "\n".join(l)
        logging.info(f"metab VIP vars:\n{l}")

        # Katta: "Polina suggested the following asserts as rule of thumb. In this way we detect
        # macro-problems like K+ accumulation faster. For now the additional computation is minimal.
        # Improvements are possible if needed."
            
        assert 0.25 <= self.vm[idx_atpn] <= 2.5, self.vm[idx_atpn]
        assert 5 <= self.vm[idx_nai] <= 30, self.vm[idx_nai]  # usually around 10
        assert 1 <= self.vm[idx_ko] <= 10, self.vm[idx_ko]
        assert 1 <= self.um[(0, c_gid)][idx_ko] <= 10, self.um[(0, c_gid)][idx_ko]
        assert 100 <= kis_mean <= 160, kis_mean

        # 2.2 should coincide with the BC METypePath field & with u0_file
        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model
        # commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model

        
        self.prnt.append_to_file(
            config.param_out_file,
            [c_gid, i_metab, *list(param)],
            rank,
        )

        return param

