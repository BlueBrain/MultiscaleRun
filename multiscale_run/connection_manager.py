from neurodamus.connection_manager import SynapseRuleManager
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

from . import utils

config = utils.load_config()

import logging
import neurodamus
import numpy as np
from scipy import sparse
import steps

import os


class MsrConnectionManager:
    """This keeps track of all the connection matrices

    For maximum efficiency matrices are cached with: cache_decorator
    """

    @utils.cache_decorator(
        path=config.cache_path,
        is_save=False,  # deactivated, not critical atm
        is_load=False,  # deactivated, not critical atm
        only_rank0=False,
        field_names=["nXtetMat", "nsegXtetMat", "nXnsegMatBool"],
    )
    @utils.logs_decorator
    def connect_ndam2steps(self, ndam_m, steps_m):
        """neuron volume fractions in tets"""
        pts = ndam_m.get_seg_points(steps_m.msh._scale)

        nsegXtetMat = steps_m.get_nsegXtetMat(pts)
        nsecXnsegMat = ndam_m.get_nsecXnsegMat(pts)
        nXsecMat = ndam_m.get_nXsecMat()

        # Connection matrices
        self.nXtetMat = nXsecMat.dot(nsecXnsegMat.dot(nsegXtetMat))
        self.nsegXtetMat = nsegXtetMat
        self.nXnsegMatBool = nXsecMat.dot(nsecXnsegMat) > 0

    @utils.cache_decorator(
        path=config.cache_path,
        is_save=config.cache_save,
        is_load=config.cache_load,
        only_rank0=True,
        field_names=["tetXbfVolsMat", "tetXbfFlowsMat", "tetXtetMat"],
    )
    @utils.logs_decorator
    def connect_bf2steps(self, bf_m, steps_m):
        """vols and flows fractions in tets"""
        pts = None
        if rank == 0:
            pts = bf_m.get_seg_points(steps_m.msh._scale)
        pts = comm.bcast(pts, root=0)

        mat, starting_tets = steps_m.get_tetXbfSegMat(pts)
        self.tetXbfVolsMat, self.tetXbfFlowsMat = None, None
        if rank == 0:
            # resize based on tet volume
            flowsMat = mat.sign()

            for isec, itet in enumerate(starting_tets):
                if isec not in bf_m.entry_nodes:
                    flowsMat[itet, isec] = 0

            flowsMat.eliminate_zeros()

            # Connection matrices
            self.tetXtetMat = steps_m.get_tetXtetMat()
            self.tetXbfVolsMat = mat
            self.tetXbfFlowsMat = flowsMat

    def delete_rows(self, m, to_be_removed):
        """We need to remove rows in case of failed neurons"""
        attr = getattr(self, m)
        setattr(self, m, utils.delete_rows_csr(attr, to_be_removed))

    def delete_cols(self, m, to_be_removed):
        """We need to remove rows in case of failed neurons"""
        attr = getattr(self, m)
        setattr(self, m, utils.delete_cols_csr(attr, to_be_removed))

    def ndam2steps_sync(self, ndam_m, steps_m, specs, DT):
        """use ndam to correct steps concentrations"""
        for s in specs:
            # there are 1e8 µm2 in a cm2, final output in mA
            seg_curr = ndam_m.get_var(var=s.current_var, weight="area") * 1e-8

            tet_curr = self.nsegXtetMat.transpose().dot(seg_curr)
            comm.Allreduce(tet_curr, tet_curr, op=MPI4PY.SUM)

            steps_m.correct_concs(species=s, curr=tet_curr, DT=DT)

    def ndam2metab_sync(self, ndam_m, metab_m):
        """use ndam to compute current and concentration concentrations for metab"""
        vars_weights = {
            "ina_density": (config.Na.current_var, "area"),
            "ik_density": (config.KK.current_var, "area"),
            "nais_mean": (config.Na.nai_var, "volume"),
            "kis_mean": (config.KK.ki_var, "volume"),
            "cais_mean": (config.Ca.current_var, "volume"),
            "atpi_mean": (config.ATP.atpi_var, "volume"),
            "adpi_mean": (config.ADP.adpi_var, "volume"),
        }

        ans = {}
        d = {
            "area": [i[0] for i in ndam_m.nc_areas],
            "volume": [i[0] for i in ndam_m.nc_vols],
        }
        for k, (var, weight) in vars_weights.items():
            ans[k] = self.nXnsegMatBool.dot(ndam_m.get_var(var=var, weight=weight))

            l = d.get(weight, None)
            ans[k] = np.divide(ans[k], l)

        metab_m.ndam_vars = ans

    def steps2metab_sync(self, steps_m, metab_m):
        """steps concentrations for metab"""

        r = 1.0 / (1e-3 * config.CONC_FACTOR)

        l = [config.KK.name]
        ans = {}
        for name in l:
            ans[name] = steps_m.get_tet_concs(name) * r
            ans[name] = self.nXtetMat.dot(ans[name])
        metab_m.steps_vars = ans

    def ndam2bloodflow_sync(self, ndam_m, bf_m):
        """ndam vasculature radii for bloodflow"""

        vasc_ids, radii = ndam_m.get_vasc_radii()
        vasc_ids = comm.gather(vasc_ids, root=0)
        radii = comm.gather(radii, root=0)
        if rank == 0:
            vasc_ids = [j for i in vasc_ids for j in i]
            radii = [j for i in radii for j in i]
            bf_m.set_radii(vasc_ids, radii)

    def bloodflow2metab_sync(self, bf_m, metab_m):
        """bloodflow flows and volumes for metab"""
        Fin, vol = None, None
        if rank == 0:
            # 1e-12 to pass from um^3 to ml
            # 500 is 1/0.0002 (1/0.2%) since we discussed that the vasculature is only 0.2% of the total
            # and it is not clear to what the winter paper is referring too exactly for volume and flow
            # given that we are sure that we are not double counting on a tet Fin and Fout, we can use
            # and abs value to have always positive input flow
            Fin = (
                self.nXtetMat.dot(np.abs(self.tetXbfFlowsMat.dot(bf_m.get_flows())))
                * 1e-12
                * 500
            )
            vol = (
                self.nXtetMat.dot(
                    self.tetXtetMat.dot(self.tetXbfVolsMat.dot(bf_m.get_vols()))
                )
                * 1e-12
                * 500
            )

        Fin = comm.bcast(Fin, root=0)
        vol = comm.bcast(vol, root=0)

        metab_m.bloodflow_vars = {"Fin": Fin, "vol": vol}

    def metab2ndam_sync(self, metab_m, ndam_m, i_metab):
        """metab concentrations for ndam"""

        if len(ndam_m.ncs) == 0:
            return

        # u stands for Julia ODE var and m stands for metabolism
        atpi_weighted_mean = np.array(
            [
                metab_m.um[(i_metab + 1, int(nc.CCell.gid))][
                    config.metab_vm_indexes["atpn"]
                ]
                for nc in ndam_m.ncs
            ]
        )  # 1.4
        # 0.5 * 1.2 + 0.5 * um[(idxm + 1, c_gid)][22] #um[(idxm+1,c_gid)][27]

        def f(atpiwm):
            # based on Jolivet 2015 DOI:10.1371/journal.pcbi.1004036 page 6 botton eq 1
            qak = 0.92
            return (atpiwm / 2) * (
                -qak + np.sqrt(qak * qak + 4 * qak * ((metab_m.ATDPtot_n / atpiwm) - 1))
            )

        adpi_weighted_mean = np.array([f(i) for i in atpi_weighted_mean])  # 0.03
        #                         nao_weighted_mean = 0.5 * 140.0 + 0.5 * (
        #                             140.0 - 1.33 * (um[(idxm + 1, c_gid)][6] - 10.0)
        #                         )  # 140.0 - 1.33*(param[3] - 10.0) #14jan2021  # or 140.0 - .. # 144  # param[3] because pyhton indexing is 0,1,2.. julia is 1,2,..

        ko_weighted_mean = np.array(
            [metab_m.vm[config.metab_vm_indexes["ko"]]] * len(ndam_m.ncs)
        )  # 5
        #                         nai_weighted_mean = (
        #                             0.5 * 10.0 + 0.5 * um[(idxm + 1, c_gid)][6]
        #                         )  # 0.5*10.0 + 0.5*um[(idxm+1,c_gid)][6] #um[(idxm+1,c_gid)][6]

        #                         ki_weighted_mean = 0.5 * 140.0 + 0.5 * param[4]  # 14jan2021
        #                         # sync loop to constrain ndamus by metabolism output

        # for seg in neurodamus_manager.gen_segs(nc, config.seg_filter):
        #     seg.nao = nao_weighted_mean  # 140
        #     seg.nai = nai_weighted_mean  # 10
        #     seg.ko = ko_weighted_mean  # 5
        #     seg.ki = ki_weighted_mean  # 140
        #     seg.atpi = atpi_weighted_mean  # 1.4
        #     seg.adpi = adpi_weighted_mean  # 0.03

        # for seg in neurodamus_manager.gen_segs(nc, [Na.nai_var]):
        #     seg.nao = nao_weighted_mean  # 140                       # TMP disable sync from STEPS to NDAM

        #                             for seg in neurodamus_manager.gen_segs(nc, [Na.nai_var]):
        #                                 seg.nai = nai_weighted_mean  # 10

        l = [
            (config.KK.ko_var, ko_weighted_mean),
            (config.ATP.atpi_var, atpi_weighted_mean),
            (config.ADP.adpi_var, adpi_weighted_mean),
        ]
        for i, v in l:
            ndam_m.set_var(i, v, filter=[i])
