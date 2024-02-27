from pathlib import Path
import sys
import time

import logging
from tqdm import tqdm
from mpi4py import MPI as MPI4PY
import numpy as np
from scipy import constants as spc
from scipy import sparse

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *
from steps.utils import *

from . import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrStepsManager:
    """
    Manages STEPS simulations and mesh operations.

    Attributes:
        config: Configuration data for the simulation.
        mdl: The STEPS model.
        msh: The mesh for the simulation.
        ntets: Number of tetrahedra in the mesh.
        tet_vols: Array of tetrahedra volumes.
    """

    def __init__(self, config):
        """
        Initialize an instance of MsrStepsManager.

        Args:
            config: Configuration data for the simulation.
        """
        self.config = config
        self.init_mesh()

    @utils.logs_decorator
    def init_sim(self):
        """
        Initialize the STEPS simulation.

        This method initializes the STEPS model and solver, sets the initial concentrations, and prepares for simulations.
        """
        self._init_model()
        self._init_solver()
        self.init_concentrations()
        self.trackers = {}

    @utils.logs_decorator
    def init_concentrations(self):
        """
        Initialize species concentrations.

        This method initializes the concentrations of species based on the configuration.
        """
        # there are 0.001 M/mM

        for s in self.config.steps.Volsys.species:
            spec = getattr(self.config.species, s)
            steps_spec = getattr(
                getattr(self.sim, self.config.steps.compname), spec.steps.name
            )
            steps_spec.Conc = 1e-3 * spec.steps.conc_0 * self.config.steps.conc_factor

    @utils.logs_decorator
    def _init_model(self):
        """
        Initialize the STEPS model.

        This method initializes the STEPS model by defining species and diffusion constants.
        """
        self.mdl = Model()
        with self.mdl:
            volsys = VolumeSystem(name=self.config.steps.Volsys.name)
            for s in self.config.steps.Volsys.species:
                spec = getattr(self.config.species, s)
                v = Species(name=spec.steps.name)
                with volsys:
                    Diffusion(
                        elem=v, Dcst=spec.steps.diffcst, name=f"diff_{spec.steps.name}"
                    )

    @utils.logs_decorator
    def init_mesh(self):
        """
        Initialize the mesh for the simulation.

        Args:
            mesh_path: The path to the mesh file or directory.

        This method initializes the mesh by loading the mesh data from the specified file or directory.
        """
        # STEPS default length scale is m
        # NEURON default length scale is um

        mesh_path = self._auto_select_mesh(self.config.mesh_path)

        self.msh = DistMesh(str(mesh_path), scale=self.config.mesh_scale)

        with self.msh:
            Compartment(
                name=self.config.steps.compname, vsys=self.config.steps.Volsys.name
            )

        self.ntets = len(self.msh.tets)
        self.tet_vols = np.array([i.Vol for i in self.msh.tets])

    @staticmethod
    def _auto_select_mesh(mesh_path):
        """
        Automatically select the mesh file for parallel simulations.

        If a split mesh is present in the format 'split_*' in the specified directory, it will be used. Otherwise, the function
        allows STEPS 4 to perform auto-splitting.

        Args:
            mesh_path (str or Path): The path to the mesh file or directory.

        Returns:
            Path: The selected mesh file path, either the original file or a split mesh if available.

        Note:
            The function assumes parallel execution with a 'rank' variable indicating the process rank. Logging is performed if
            rank is 0.
        """

        mesh_path = Path(mesh_path)
        if mesh_path.suffix or "split" in mesh_path.name:
            if rank == 0:
                logging.info(f"mesh path: {mesh_path}")
            return mesh_path

        split_mesh_path = mesh_path / f"split_{size}"
        if split_mesh_path.exists() and len(list(split_mesh_path.iterdir())) >= size:
            ans = split_mesh_path / mesh_path.name
        else:
            ans = (mesh_path / mesh_path.name).with_suffix(".msh")

        if rank == 0:
            logging.info(f"mesh path guessed: {ans}")
        return ans

    @utils.logs_decorator
    def _init_solver(self):
        """
        Initialize the STEPS solver.

        This method initializes the STEPS solver by creating a simulation object and setting up the random number generator.
        """
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

    def bbox(self):
        """
        Get the bounding box of the mesh.

        Returns:
            tuple: A tuple containing the minimum and maximum coordinates of the mesh bounding box.
        """
        return np.array(self.msh.bbox.min.tolist()), np.array(
            self.msh.bbox.max.tolist()
        )

    @utils.logs_decorator
    def get_tetXtetMat(self):
        """
        Get a matrix for measuring species dispersion in tetrahedra.

        Returns:
            sparse.csr_matrix: A sparse CSR matrix for measuring species dispersion in tetrahedra.
        """
        return sparse.diags(
            np.reciprocal(self.tet_vols) * np.mean(self.tet_vols), format="csr"
        )

    def get_tetXtetInvMmat(self):
        """
        Get the inverse of the tetrahedra matrix for debugging.

        Returns:
            sparse.csr_matrix: A sparse CSR matrix representing the inverse of the tetrahedra matrix.
        """
        return sparse.diags(self.tet_vols * (1 / np.mean(self.tet_vols)), format="csr")

    def check_pts_inside_mesh_bbox(self, pts_list):
        """
        Check if the given points are inside the mesh bounding box.

        Args:
            pts_list (list of numpy.ndarray): A list of NumPy arrays, where each array has a shape of (n, 3) representing the points to check.

        Raises:
            AssertionError: If the ratio of points inside the mesh bounding box is less than the specified ratio.

        Returns:
            None
        """
        if pts_list is None:
            return

        bbox_min, bbox_max = self.bbox()

        for pts in pts_list:
            npts = pts.shape[0]

            v = (
                (pts[:, 0] >= bbox_min[0])
                & (pts[:, 0] <= bbox_max[0])
                & (pts[:, 1] >= bbox_min[1])
                & (pts[:, 1] <= bbox_max[1])
                & (pts[:, 2] >= bbox_min[2])
                & (pts[:, 2] <= bbox_max[2])
            )

            out_pts = pts[v, :]

            n_inside = sum(v)

            if npts < n_inside:
                utils.rank_print(f"npts < n_inside: {npts} < {n_inside}\n{out_pts}")
                comm.Abort()

    @utils.logs_decorator
    def get_nsegXtetMat(self, local_pts):
        """
        Get a matrix of section ratios in tetrahedra.

        Args:
            local_pts: Represent neuron segments.

        Returns:
            sparse.csr_matrix: A sparse CSR matrix representing section ratios in tetrahedra.
        """
        self.check_pts_inside_mesh_bbox(local_pts)

        with self.msh.asLocal():
            for i in tqdm(range(size), file=sys.stdout) if rank == 0 else range(size):
                if rank == 0:
                    # needed, otherwise tqdm output is not flushed.
                    print("", flush=True)

                pts = None

                if i == rank:
                    pts = local_pts

                # Chritos and Katta: We tried to use Bcast but there was no significant speed-up
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
                            global_tet_idx,
                        ]
                        for isec, sec in enumerate(pts)
                        for iseg, seg in enumerate(
                            self.msh.intersect(sec, raw=True, local=False)
                        )
                        for global_tet_idx, ratio in seg
                    ]
                )

                # Christos and Katta: We tried to use Gather but there was no significant speed-up
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
        """
        Get a matrix of bloodflow segments in tetrahedra.

        Args:
            pts: Represent bloodflow segments.

        Returns:
            sparse.csr_matrix: A sparse CSR matrix representing bloodflow segments in tetrahedra.
        """

        self.check_pts_inside_mesh_bbox([pts])

        with self.msh.asLocal():
            # list of (tet_id, ratio) lists
            l = self.msh.intersectIndependentSegments(pts, raw=True, local=False)
            np.testing.assert_equal(2 * len(l), pts.shape[0])

            # data is always a nX3 array:
            # col[0] is the ratio (dimensionless).
            # col[1] is the row at which it should be added in the sparse matrix
            # col[2] is the col at which it should be added in the sparse matrix

            data = np.array(
                [
                    [ratio, tet_global_idx, idx]
                    for idx, q in enumerate(l)
                    for tet_global_idx, ratio in q
                ]
            )

            # 0 is the invalid value. Summing of invalid and valid values gives the index given that only one tet reports a non-0 value
            # the +1 is for testing that everything is ok

            starting_tets = [
                [idx, tet_global_idx[0][0]]
                for idx, tet_global_idx in enumerate(l)
                if len(tet_global_idx)
                and steps.geom.TetReference(
                    tet_global_idx[0][0], mesh=self.msh, local=False
                )
                .toLocal()
                .containsPoint(pts[idx * 2, :])
            ]

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
            # flatten
            st = [i for r in starting_tets for i in r]

        return mat, st

    def get_tet_counts(self, species_name, idxs=None):
        """
        Get tetrahedra species counts.

        Args:
            species: A species object.
            idxs: An array of indices representing the tetrahedra to be counted (optional).

        Returns:
            numpy.ndarray: An array of species counts.
        """

        return self.get_tet_quantity(
            species_name=species_name, f="getBatchTetSpecCountsNP", idxs=idxs
        )

    def get_tet_concs(self, species_name, idxs=None):
        """
        Get tetrahedra species concentrations.

        Args:
            species_name (str): The name of the species.
            idxs: An array of indices representing the tetrahedra to obtain concentrations from (optional).

        Returns:
            numpy.ndarray: An array of species concentrations.
        """

        return self.get_tet_quantity(
            species_name=species_name, f="getBatchTetSpecConcsNP", idxs=idxs
        )

    def get_tet_quantity(self, species_name, f, idxs=None):
        """
        Get a specific quantity for tetrahedra.

        Args:
            species_name (str): The name of the species.
            f: A function to obtain the quantity.
            idxs: An array of indices representing the tetrahedra to obtain the quantity from (optional).

        Returns:
            numpy.ndarray: An array of the specified quantity.
        """

        if idxs is None:
            idxs = np.array(range(self.ntets), dtype=np.int64)

        ans = np.zeros(len(idxs), dtype=float)
        getattr(self.sim.stepsSolver, f)(idxs, species_name, ans)

        return ans

    def update_concs(self, species, curr, DT, idxs=None):
        """
        Update concentrations in a STEPS model based on membrane currents.

        This function updates concentrations of a specified species in the model using the provided membrane currents. It calculates the required change in concentration based on the input currents and time step and then adds this change to the existing concentrations. The function also ensures that concentrations remain non-negative and updates the concentrations for a specific set of tetrahedra.

        Parameters:
            species_name (str): The name of the species to be updated.
            species: Additional parameter (not described in the function, consider adding a description).
            curr (float): The membrane currents.
            DT (float): The time step.
            idxs (numpy.ndarray, optional): An array of indices representing the tetrahedra to be updated. If not provided, all tetrahedra are updated.

        Units:
            - Concentrations are in M/L.
            - curr is in mA.
            - DT is in ms.
            - tet_vols are in m^3.

        Returns:
            None
        """

        conc = self.get_tet_concs(species_name=species.steps.name)
        # correct tetCons according to the currents
        # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
        # In steps use M/L and apply the SIM_REAL ratio
        # A = C/s = n * c*e/s. To get mol/L we
        # the 1e+-3 factor are for: mA -> A, ms -> s, m^3 -> L
        corr_conc = np.divide(
            curr
            * 1e-3
            * self.config.steps.conc_factor
            * (DT * 1e-3)
            / (
                spc.N_A
                * species.steps.ncharges
                * spc.physical_constants["elementary charge"][0]
            ),
            self.tet_vols * 1e3,
        )

        conc += corr_conc

        negative_indices = np.where(conc < 0)
        if negative_indices[0].size > 0:
            negative_values = conc[negative_indices]
            error_message = "Negative values found at indices and values: "
            error_message += ", ".join(
                [
                    f"({idx}, {val})"
                    for idx, val in zip(negative_indices[0], negative_values)
                ]
            )
            raise ValueError(error_message)

        if idxs is None:
            idxs = np.array(range(self.ntets), dtype=np.int64)

        # only owned tets are set
        self.sim.stepsSolver.setBatchTetSpecConcsNP(idxs, species.steps.name, conc)
