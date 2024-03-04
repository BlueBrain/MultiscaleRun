import numpy as np
from scipy import sparse

import neurodamus
import steps
from mpi4py import MPI as MPI4PY
from neurodamus.connection_manager import SynapseRuleManager

from . import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrConnectionManager:
    """Tracks connection matrices for various models, enabling efficient caching.

    This class maintains various connection matrices used in the multiscale simulation, improving efficiency by caching them when necessary.

    Attributes:
        config (MrConfig): The multiscale run configuration.
    """

    def __init__(self, config):
        """Initialize the MsrConnectionManager with a given configuration.

        Args:
            config (MrConfig): The multiscale run configuration.

        Returns:
            None
        """
        self.config = config

    @utils.cache_decorator(
        only_rank0=False,
        field_names=["nXsecMat", "nsecXnsegMat", "nXnsegMatBool"],
    )
    @utils.logs_decorator
    def connect_ndam2ndam(self, ndam_m):
        """Add some useful matrices that map ndam points, segments, sections, and neurons.

        This method calculates and stores several matrices that provide mappings between various components of neuronal and segment data.

        Args:
            ndam_m (NeurodamusModel): The neuronal digital anatomy model.

        Returns:
            None
        """
        pts = ndam_m.get_seg_points(self.config.mesh_scale)

        self.nXsecMat = ndam_m.get_nXsecMat()
        self.nsecXnsegMat = ndam_m.get_nsecXnsegMat(pts)
        self.nXnsegMatBool = self.nXsecMat.dot(self.nsecXnsegMat) > 0

    @utils.cache_decorator(
        only_rank0=False,
        field_names=["nsegXtetMat", "nXtetMat"],
    )
    @utils.logs_decorator
    def connect_ndam2steps(self, ndam_m, steps_m):
        """Neuron volume fractions in tets.

        This method calculates the neuron volume fractions within tetrahedral elements and stores the results in connection matrices.

        Args:
            ndam_m (NeurodamusModel): The neuronal digital anatomy model.
            steps_m (StepsModel): The steps model representing the vasculature.

        Returns:
            None
        """
        pts = ndam_m.get_seg_points(scale=self.config.mesh_scale)

        # Connection matrices
        self.nsegXtetMat = steps_m.get_nsegXtetMat(pts)
        self.nXtetMat = self.nXsecMat.dot(self.nsecXnsegMat.dot(self.nsegXtetMat))

    @utils.cache_decorator(
        only_rank0=True,
        field_names=["tetXbfVolsMat", "tetXbfFlowsMat", "tetXtetMat"],
    )
    @utils.logs_decorator
    def connect_bf2steps(self, bf_m, steps_m):
        """Volumes and flows fractions in tets.

        This method calculates and stores volumes and flows fractions within tetrahedral elements and synchronization matrices.

        Args:
            bf_m (BloodflowModel): The bloodflow model.
            steps_m (StepsModel): The steps model representing the vasculature.

        Returns:
            None
        """
        pts = None
        if rank == 0:
            pts = bf_m.get_seg_points(steps_m.msh._scale)
        pts = comm.bcast(pts, root=0)

        mat, starting_tets = steps_m.get_tetXbfSegMat(pts)
        self.tetXbfVolsMat, self.tetXbfFlowsMat = None, None
        if rank == 0:
            # resize based on tet volume
            flowsMat = mat.sign()

            for isec, itet in starting_tets:
                if isec not in bf_m.entry_nodes:
                    flowsMat[itet, isec] = 0

            flowsMat.eliminate_zeros()

            # Connection matrices
            self.tetXtetMat = steps_m.get_tetXtetMat()
            self.tetXbfVolsMat = mat
            self.tetXbfFlowsMat = flowsMat

    def delete_rows(self, m, to_be_removed):
        """Remove rows from a matrix.

        This method removes specified rows from a matrix to accommodate changes, particularly in cases of failed neurons.

        Args:
            m: The matrix to be modified.
            to_be_removed: A list of row indices to be removed.

        Returns:
            None
        """
        if hasattr(self, m):
            attr = getattr(self, m)
            setattr(self, m, utils.delete_rows_csr(attr, to_be_removed))

    def delete_cols(self, m, to_be_removed):
        """Remove columns from a matrix.

        This method removes specified columns from a matrix, typically in cases of failed neurons or changes in the network structure.

        Args:
            m: The matrix to be modified.
            to_be_removed: A list of column indices to be removed.

        Returns:
            None
        """
        if hasattr(self, m):
            attr = getattr(self, m)
            setattr(self, m, utils.delete_cols_csr(attr, to_be_removed))

    @utils.logs_decorator
    def ndam2steps_sync(self, ndam_m, steps_m, specs: list[str], DT: float):
        """Use neuronal data to correct step concentrations.

        Sync steps concentrations adding the output currents from neurodamus.

        Args:
            ndam_m (MsrNeurodamusManager): neurodamus manager. Source of the data to be synchronized.
            steps_m (MsrStepsManager): The source manager containing concentration data.
            specs: A list of species and their names.
            DT: steps dt.

        Returns:
            None
        """

        for s in specs:
            sp = getattr(self.config.species, s)
            # there are 1e8 µm2 in a cm2, final output in mA
            seg_curr = ndam_m.get_var(var=sp.neurodamus.curr.var, weight="area") * 1e-8

            tet_curr = self.nsegXtetMat.transpose().dot(seg_curr)
            comm.Allreduce(tet_curr, tet_curr, op=MPI4PY.SUM)

            steps_m.update_concs(species=sp, curr=tet_curr, DT=DT)

    @utils.logs_decorator
    def ndam2metab_sync(self, gids: list[int], ndam_m, metab_m):
        """Use neuronal data to compute current and concentration concentrations for metabolism.

        This method utilizes neurodamus to calculate the current and concentration concentrations for a metabolism model.

        Args:

            gids: list of neurons (not kicked, on this rank)
            ndam_m (MsrNeurodamusManager): neurodamus manager. Source of the data to be synchronized.
            metab_m (MsrMetabolismManager): metabolism manager to be syncronized.

        Returns:
            None
        """

        ndam_vars = {}
        d = {
            "area": [i[0] for i in ndam_m.nc_areas],
            "volume": [i[0] for i in ndam_m.nc_vols],
        }
        for s, t in self.config.metabolism.ndam_input_vars:
            weight = "area" if t == "curr" else "volume"
            var = getattr(getattr(self.config.species, s).neurodamus, t).var
            ndam_vars[(s, t)] = self.nXnsegMatBool.dot(
                ndam_m.get_var(var=var, weight=weight)
            )

            l = d.get(weight, None)
            ndam_vars[(s, t)] = np.divide(ndam_vars[(s, t)], l)

        gids = ndam_m.gids()
        ndam_vars["valid_gid"] = np.ones(len(gids))

        # necessary for reporting. It will go away in a future MR
        metab_m.ndam_vars = ndam_vars
        metab_m.set_ndam_vars(gids=gids, ndam_vars=ndam_vars)

    @utils.logs_decorator
    def steps2metab_sync(self, gids: list[int], steps_m, metab_m):
        """
        Synchronize concentration data from Steps to Metabolism models for specific species.

        This function converts concentration data from a 'steps_m' model to a 'metab_m' model for specified species, taking into account scaling factors.

        Args:

            gids: list of neurons (not kicked, on this rank)
            steps_m (MsrStepsManager): The source manager containing concentration data.
            metab_m (MsrMetabolismManager): metabolism manager to be syncronized.

        Returns:
            None
        """

        r = 1.0 / (1e-3 * self.config.steps.conc_factor)

        l = [
            getattr(self.config.species, i).steps.name
            for i in self.config.metabolism.steps_input_concs
        ]
        steps_vars = {}

        for name in l:
            steps_vars[name] = steps_m.get_tet_concs(name) * r
            steps_vars[name] = self.nXtetMat.dot(steps_vars[name])

        # necessary for reporting. It will go away in a future MR
        metab_m.steps_vars = steps_vars
        metab_m.set_steps_vars(gids=gids, steps_vars=steps_vars)

    @utils.logs_decorator
    def bloodflow2metab_sync(self, gids: list[int], bf_m, metab_m):
        """
        Synchronize bloodflow parameters including flows and volumes from a bloodflow model to a metabolism model.

        This function calculates and synchronizes blood flow and volume data from a bloodflow model to a metabolism model. It performs the necessary transformations and scaling based on the models' data to ensure accurate synchronization.

        Args:

            gids: list of neurons (not kicked, on this rank)
            bf_m (MsrBloodflowManager): bloodFlow manager containing bloodflow data.
            metab_m (MsrMetabolismManager): metabolism manager to be synchronized.

        Returns:
            None
        """

        Fin, vol = None, None
        if rank == 0:
            # 1e-12 to pass from um^3 to ml
            # 500 is 1/0.0002 (1/0.2%) since we discussed that the vasculature is only 0.2% of the total
            # and it is not clear to what the winter paper is referring too exactly for volume and flow
            # given that we are sure that we are not double counting on a tet Fin and Fout, we can use
            # and abs value to have always positive input flow
            Fin = np.abs(self.tetXbfFlowsMat.dot(bf_m.get_flows())) * 1e-12 * 500
            vol = (
                self.tetXtetMat.dot(self.tetXbfVolsMat.dot(bf_m.get_vols()))
                * 1e-12
                * 500
            )

        Fin = comm.bcast(Fin, root=0)
        vol = comm.bcast(vol, root=0)

        Fin = self.nXtetMat.dot(Fin)
        vol = self.nXtetMat.dot(vol)

        bloodflow_vars = {"Fin": Fin, "vol": vol}

        # necessary for reporting. It will go away in a future MR
        metab_m.bloodflow_vars = bloodflow_vars
        metab_m.set_bloodflow_vars(gids=gids, bloodflow_vars=bloodflow_vars)

    def ndam2bloodflow_sync(self, ndam_m, bf_m):
        """
        Synchronize vascular radii data from a ndam model to a bloodflow model.

        This function gathers vascular IDs and radii information from the NDAM (Neuronal Digital Anatomy Model) model and synchronizes it with the bloodflow model. The gathered data is used to set radii in the bloodflow model for corresponding vascular segments.

        Args:
            ndam_m (MsrNeurodamusManager): neurodamus manager. Source of the radii to be synchronized.
            bf_m (MsrBloodflowManager): bloodFlow manager containing radii to be corrected.

        Returns:
            None
        """

        vasc_ids, radii = ndam_m.get_vasc_radii()
        vasc_ids = comm.gather(vasc_ids, root=0)
        radii = comm.gather(radii, root=0)
        if rank == 0:
            vasc_ids = [j for i in vasc_ids for j in i]
            radii = [j for i in radii for j in i]
            bf_m.set_radii(vasc_ids=vasc_ids, radii=radii)

    def metab2ndam_sync(self, metab_m, ndam_m):
        """
        Synchronize metabolic concentrations from a metabolism model to a NDAM (Neuronal Digital Anatomy Model) for specific variables.

        This function calculates and synchronizes metabolic concentrations, including ATP, ADP, and potassium (K+), from a metabolism model to a NDAM model for specific variables, taking into account weighted means. It corrects ndam with the metabolic output.

        Args:
            metab_m (MsrMetabolismManager): metabolism manager. Source of the data to be synchronized.
            ndam_m (MsrNeurodamusManager): neurodamus manager. Destination of the data to be synchronized.

        Returns:
            None
        """

        if len(ndam_m.ncs) == 0:
            return

        # u stands for Julia ODE var and m stands for metabolism
        atpi_weighted_mean = np.array(
            [
                metab_m.vm[int(nc.CCell.gid)][self.config.metabolism.vm_idxs.atpn]
                for nc in ndam_m.ncs
            ]
        )  # 1.4
        # 0.5 * 1.2 + 0.5 * um[(idxm + 1, c_gid)][22] #um[(idxm+1,c_gid)][27]

        def f(atpiwm):
            # based on Jolivet 2015 DOI:10.1371/journal.pcbi.1004036 page 6 botton eq 1
            qak = 0.92
            return (atpiwm / 2) * (
                -qak
                + np.sqrt(
                    qak * qak
                    + 4
                    * qak
                    * ((self.config.metabolism.constants.ATDPtot_n / atpiwm) - 1)
                )
            )

        adpi_weighted_mean = np.array([f(i) for i in atpi_weighted_mean])  # 0.03
        #                         nao_weighted_mean = 0.5 * 140.0 + 0.5 * (
        #                             140.0 - 1.33 * (um[(idxm + 1, c_gid)][6] - 10.0)
        #                         )  # 140.0 - 1.33*(param[3] - 10.0) #14jan2021  # or 140.0 - .. # 144  # param[3] because pyhton indexing is 0,1,2.. julia is 1,2,..

        ko_weighted_mean = np.array(
            [
                metab_m.vm[int(nc.CCell.gid)][self.config.metabolism.vm_idxs.ko]
                for nc in ndam_m.ncs
            ]
        )
        # 5
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
        ndam_k_conco = self.config.species.K.neurodamus.conco.var
        ndam_atp_conco = self.config.species.atp.neurodamus.conci.var
        ndam_adp_conco = self.config.species.adp.neurodamus.conci.var

        l = [
            (ndam_k_conco, ko_weighted_mean),
            (ndam_atp_conco, atpi_weighted_mean),
            (ndam_adp_conco, adpi_weighted_mean),
        ]
        for i, v in l:
            ndam_m.set_var(i, v, filter=[i])
