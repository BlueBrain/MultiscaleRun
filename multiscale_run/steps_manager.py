import time
import logging
import numpy as np
from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
import os

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *
from steps.utils import *

from scipy import sparse


from . import utils
config = utils.load_config()


class MsrStepsManager:
    def __init__(self, mesh_path):
        self._init_model()
        self._init_mesh(mesh_path)
        self._init_solver()
        self.init_concentrations()
        self.trackers = {}

    @utils.logs_decorator
    def init_concentrations(self):
        # there are 0.001 M/mM
        self.sim.extra.Na.Conc = 1e-3 * config.Na.conc_0 * config.CONC_FACTOR
        self.sim.extra.KK.Conc = 1e-3 * config.KK.conc_0 * config.CONC_FACTOR

    @utils.logs_decorator
    def _init_model(self):
        self.mdl = Model()
        with self.mdl:
            extra_volsys = VolumeSystem(name=config.Volsys.name)
            Na = Species(name=config.Na.name)
            KK = Species(name=config.KK.name)
            with extra_volsys:
                diff_Na = Diffusion.Create(Na, config.Na.diffcst)
                diff_KK = Diffusion.Create(KK, config.KK.diffcst)

    @utils.logs_decorator
    def _init_mesh(self, mesh_path):
        # STEPS default length scale is m
        # NEURON default length scale is um

        mesh_path = self._auto_select_mesh(mesh_path)

        self.msh = DistMesh(mesh_path, scale=1e-6)

        with self.msh:
            extra = Compartment(name=config.Mesh.compname, vsys=config.Volsys.name)

        self.ntets = len(self.msh.tets)
        self.tet_vols = np.array([i.Vol for i in self.msh.tets])

    @staticmethod
    def _auto_select_mesh(mesh_path):
        """use split mesh if present in split_*. Otherwise let STEPS 4 auto-split"""

        s = os.path.basename(mesh_path).split(".")
        if len(s) > 1 or "split" in mesh_path:
            if rank == 0:
                logging.info(f"mesh path: {mesh_path}")
            return mesh_path

        mesh_name = os.path.basename(mesh_path)
        split_mesh_path = os.path.join(mesh_path, f"split_{size}")
        if os.path.exists(split_mesh_path) and len(os.listdir(split_mesh_path)) >= size:
            ans = os.path.join(split_mesh_path, mesh_name)
        else:
            ans = os.path.join(mesh_path, mesh_name + ".msh")

        if rank == 0:
            logging.info(f"mesh path guessed: {ans}")
        return ans

    @utils.logs_decorator
    def _init_solver(self):
        logging.info("init solver")
        self.rng = RNG("mt19937", 512, int(time.time() % 4294967295))
        self.sim = Simulation(
            "DistTetOpSplit",
            self.mdl,
            self.msh,
            self.rng,
            searchMethod=NextEventSearchMethod.GIBSON_BRUCK,
        )
        self.sim.newRun()

    @utils.logs_decorator
    def get_tetXtetMat(self):
        """Diagonal matrix that gives a measure of how "dispersed" a species is in a tet compared to the average tet

        We assume that the dispersion is linearly related to volume
        Used to translate species from bloodflow to metabolism
        """
        return sparse.diags(np.reciprocal(self.tet_vols) * np.mean(self.tet_vols))

    def get_tetXtetInvMmat(self):
        """Inverse of Tmat. Used for debugging"""
        return sparse.diags(self.tet_vols * (1 / np.mean(self.tet_vols)))

    @utils.logs_decorator
    def get_nsegXtetMat(self, local_pts):
        """Get secXtet csr_matrix with the weights

        local_pts represent neuron segments

        Every element gives the ratio of section (isec) in a certain tet (itet).
        The base idea is that every rank has some sections/segments and some tets.

        local_pts[isec] = np.array(n, 3) of contiguous points

        Intersect returns a list of lists of (tet, ratio) intersections
        """

        with self.msh.asLocal():
            for i in range(comm.Get_size()):
                pts = None

                if i == rank:
                    pts = local_pts

                pts = comm.bcast(pts, root=i)

                # this is an array of partial sums (ps) to get the
                #  iseg offset once we flatten everything
                # in one simple array
                ps = [0, *np.cumsum([len(i) - 1 for i in pts])]

                # data is always a nX3 array: 
                # col[0] is the ratio (dimensionless).
                # col[1] is the row at which it should be added in the sparse matrix
                # col[2] is the col at which it should be added in the sparse matrix
                data = np.array(
                    [
                        [
                            ratio,
                            iseg + ps[isec],
                            tet.toGlobal().idx,
                        ]
                        for isec, sec in enumerate(pts)
                        for iseg, seg in enumerate(self.msh.intersect(sec))
                        for tet, ratio in seg
                    ]
                )

                data = comm.gather(data, root=i)

                if i == rank:
                    if len(pts) > 0:
                        ans = sum(
                            [
                                sparse.csr_matrix(
                                    (q[:, 0], (q[:, 1], q[:, 2])),
                                    shape=(ps[-1], self.ntets),
                                )
                                for q in data
                                if q.size
                            ]
                        )
                    else:
                        ans = sparse.csr_matrix((ps[-1], self.ntets))

        return ans

    @utils.logs_decorator
    def get_tetXbfSegMat(self, pts):
        """Get bfsecXtet csr_matrix with the weights

        local_pts represent bloodflow segments

        Every element gives the ratio of segment (iseg) in a certain tet (itet).
        The base idea is that every rank has some segments/segments and some tets.

        local_pts[isec] = np.array(n, 3) of contiguous points

        Intersect returns a list of lists of (tet, ratio) intersections
        """

        with self.msh.asLocal():
            # list of (tet_id, ratio) lists
            l = self.msh.intersectIndependentSegments(pts)
            np.testing.assert_equal(2 * len(l), pts.shape[0])

            # data is always a nX3 array: 
            # col[0] is the ratio (dimensionless).
            # col[1] is the row at which it should be added in the sparse matrix
            # col[2] is the col at which it should be added in the sparse matrix
            data = np.array(
                [
                    [ratio, tet.toGlobal().idx, idx]
                    for idx, q in enumerate(l)
                    for tet, ratio in q
                ]
            )
            # 0 is the invalid value. Summing of invalid and valid values gives the index given that only one tet reports a non-0 value
            # the +1 is for testing that everything is ok
            starting_tets = np.array(
                [
                    q[0][0].toGlobal().idx
                    if len(q) and q[0][0].containsPoint(pts[idx * 2, :])
                    else 0
                    for idx, q in enumerate(l)
                ]
            )

        data = comm.gather(data, root=0)
        starting_tets = comm.gather(starting_tets, root=0)

        mat, st = None, None
        if rank == 0:
            mat = sum(
                [
                    sparse.csr_matrix(
                        (i[:, 0], (i[:, 1], i[:, 2])),
                        shape=(self.ntets, int(len(pts) / 2)),
                    )
                    for i in data
                    if len(i)
                ]
            )

            st = np.sum(np.array(starting_tets), axis=0)
        return mat, st

    def get_tet_counts(self, species, idxs=None):
        return self.get_tet_quantity(
            species=species, f="getBatchTetSpecCountsNP", idxs=idxs
        )

    def get_tet_concs(self, species, idxs=None):
        return self.get_tet_quantity(
            species=species, f="getBatchTetSpecConcsNP", idxs=idxs
        )

    def get_tet_quantity(self, species, f, idxs=None):
        if idxs == None:
            idxs = np.array(range(self.ntets), dtype=np.int64)

        if not isinstance(species, str):
            species = species.name

        ans = np.zeros(len(idxs), dtype=float)
        getattr(self.sim.stepsSolver, f)(idxs, species, ans)

        return ans

    def correct_concs(self, species, curr, DT, idxs=None):
        """
        steps concs in M/L
        curr in mA
        DT in ms
        tet_vols in m^3
        """
        conc = self.get_tet_concs(species=species.name)
        # correct tetCons according to the currents
        # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
        # In steps use M/L and apply the SIM_REAL ratio
        # A = C/s = n * c*e/s. To get mol/L we
        # the 1e+-3 factor are for: mA -> A, ms -> s, m^3 -> L
        corr_conc = np.divide(
            curr
            * 1e-3
            * config.CONC_FACTOR
            * (DT * 1e-3)
            / (config.AVOGADRO * species.charge),
            self.tet_vols * 1e3,
        )

        conc += corr_conc

        if idxs == None:
            idxs = np.array(range(self.ntets), dtype=np.int64)

        # only owned tets are set
        self.sim.stepsSolver.setBatchTetSpecConcsNP(idxs, species.name, conc)
