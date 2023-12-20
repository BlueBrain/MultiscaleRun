import logging

import numpy as np
from scipy import sparse

import neurodamus
import steps
from mpi4py import MPI as MPI4PY
from neurodamus.connection_manager import SynapseRuleManager

from . import utils

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrNeurodamusManager:
    """Handles neurodamus and keeps track of what neurons are working."""

    def __init__(self, config):
        """Initialize the MsrNeurodamusManager with the given configuration.

        Args:
            config: The configuration for the neurodamus manager.

        Returns:
            None
        """
        logging.info("instantiate ndam")
        self.ndamus = neurodamus.Neurodamus(
            str(config.sonata_path),
            logging_level=config.logging_level,
            enable_coord_mapping=True,
            cleanup_atexit=False,
        )
        logging.info("ndam sim init")
        self.ndamus.sim_init()
        logging.info("ndam is ready")

        self.set_managers()
        self.ncs = np.array([nc for nc in self.neuron_manager.cells])
        # useful for reporting
        self.num_neurons_per_rank = comm.gather(len(self.ncs), root=0)
        self.init_ncs_len = len(self.ncs)
        self.acs = np.array([nc for nc in self.astrocyte_manager.cells])
        self.nc_vols = self._cumulate_nc_sec_quantity("volume")
        self.nc_areas = self._cumulate_nc_sec_quantity("area")
        self.removed_gids = {}

    def gids(self):
        """Convenience function to get the gids from ncs"""
        return [int(nc.CCell.gid) for nc in self.ncs]

    def remove_gids(self, failed_cells, conn_m):
        """Add GIDs (Global IDs) to the removed_gids list and update connections.

        This method adds the GIDs of failed cells to the removed_gids list and updates the connections in the given connection manager.

        Args:
            failed_cells: A dictionary containing information about failed cells.
            conn_m: The connection manager to update connections.

        Returns:
            None
        """

        self.removed_gids |= failed_cells

        self.update(conn_m=conn_m)

    def dt(self):
        """Get the time step (Dt) used in neurodamus simulations.

        Returns:
            float: The time step value.
        """
        return float(self.ndamus._run_conf["Dt"])

    def duration(self):
        """Get the duration used in neurodamus simulations.

        Returns:
            float: The duration value.
        """

        return float(self.ndamus._run_conf["Duration"])

    def set_managers(self):
        """Find useful node managers for neurons, astrocytes, and glio-vascular management.

        This method sets the neuron_manager, astrocyte_manager, and glio_vascular_manager attributes based on available node managers.

        Returns:
            None
        """

        self.neuron_manager = [
            i
            for i in self.ndamus.circuits.all_node_managers()
            if isinstance(i, neurodamus.cell_distributor.CellDistributor)
            and i.total_cells > 0
        ][0]
        self.astrocyte_manager = [
            i
            for i in self.ndamus.circuits.all_node_managers()
            if isinstance(i, neurodamus.ngv.AstrocyteManager) and i.total_cells > 0
        ][0]

        self.glio_vascular_manager = self.ndamus.circuits.get_edge_manager(
            "vasculature", "astrocytes", neurodamus.ngv.GlioVascularManager
        )

    @staticmethod
    def _gen_secs(nc, filter=[]):
        """Generator of filtered sections for a neuron.

        This method generates filtered sections for a neuron based on the provided filter.

        Args:
            nc: A neuron to generate sections from.
            filter: A list of attributes to filter sections by.

        Yields:
            Filtered sections for the neuron.

        Returns:
            None
        """
        for sec in nc.CellRef.all:
            if not all(hasattr(sec, i) for i in filter):
                continue
            if not sec.n3d():
                continue
            yield sec

    @staticmethod
    def _gen_segs(sec):
        """Generator of segments for a neuron section.

        This method generates segments for a neuron section.

        Args:
            sec: A neuron section.

        Yields:
            Segments in the neuron section.

        Returns:
            None
        """
        for seg in sec:
            yield seg

    def _cumulate_nc_sec_quantity(self, f):
        """Calculate cumulative quantity for neuron sections.

        This method calculates a cumulative quantity for neuron sections, such as volume or area.

        Args:
            f: The quantity to calculate (e.g., "volume" or "area").

        Returns:
            np.ndarray: An array of tuples containing the sum and the individual values for each neuron section.
        """
        v = [
            [
                sum([getattr(seg, f)() for seg in self._gen_segs(sec)])
                for sec in self._gen_secs(nc)
            ]
            for nc in self.ncs
        ]
        return np.array([(sum(i), i) for i in v], dtype=object)

    def get_seg_points(self, scale):
        """Get the segment points for all neurons.

        This method retrieves the extreme points of every neuron segment, returning a consistent structure across ranks.

        Args:
            scale: A scale factor for the points.

        Returns:
            list: A list of lists of local points for each neuron segment.
        """

        if hasattr(self, "seg_points"):
            return [i * scale for i in self.seg_points]

        def get_seg_extremes(sec, loc2glob):
            """Get extremes and roto-translate in global coordinates"""

            def get_local_seg_extremes(nseg, pp):
                """Compute the position of beginning and end of each compartment in a section

                Assumption: all the compartments have the same length

                Inputs:
                    - nseg: number of compartments. only non "joint" compartments (no no-vol compartments)
                    - pp is a nX4 matrix of positions of points. The first col give the relative position, (x in neuron) along the
                    axis. It is in the interval [0, 1]. The other 3 columns give x,y,z triplets of the points in a global system of
                    reference.

                Outputs:
                - a matrix 3Xn of the position of the extremes of every proper compartment (not the extremes)
                """

                pp = np.array(pp)
                x_rel, xp, yp, zp = pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3]
                x = np.linspace(0, 1, nseg + 1)
                xp_seg = np.interp(x, x_rel, xp)
                yp_seg = np.interp(x, x_rel, yp)
                zp_seg = np.interp(x, x_rel, zp)

                return np.transpose([xp_seg, yp_seg, zp_seg])

            ll = [
                [sec.arc3d(i) / sec.L, sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                for i in range(sec.n3d())
            ]
            ans = get_local_seg_extremes(
                sec.nseg,
                ll,
            )
            ans = np.array(
                loc2glob(ans),
                dtype=float,
                order="C",
            )
            return ans

        self.seg_points = [
            get_seg_extremes(sec, nc.local_to_global_coord_mapping)
            for nc in self.ncs
            for sec in self._gen_secs(nc)
        ]
        return [i * scale for i in self.seg_points]

    def get_nsecXnsegMat(self, pts):
        """Get the nsecXnsegMat matrix.

        This method calculates the nsecXnsegMat matrix, which gives the fraction of neuron section in a tet.

        Args:
            pts: A list of neuron segment points.

        Returns:
            sparse.csr_matrix: The nsecXnsegMat matrix.
        """

        segl = [
            [np.linalg.norm(sec[i, :] - sec[i + 1, :]) for i in range(len(sec) - 1)]
            for sec in pts
        ]
        secl = [sum(i) for i in segl]

        # this is an array of partial sums (ps) to get the iseg offset once we flatten everything
        # in one simple array
        ps = [0, *np.cumsum([len(i) - 1 for i in pts])]

        # data is always a nX3 array:
        # col[0] is the ratio (dimensionless).
        # col[1] is the row at which it should be added in the sparse matrix
        # col[2] is the col at which it should be added in the sparse matrix
        data = np.array(
            [
                [
                    segl[isec][iseg] / secl[isec],
                    isec,
                    iseg + ps[isec],
                ]
                for isec, sec in enumerate(pts)
                for iseg in range(len(sec) - 1)
            ]
        )

        if len(data) > 0 and data.shape[1] > 0:
            ans = sparse.csr_matrix(
                (data[:, 0], (data[:, 1], data[:, 2])),
                shape=(len(secl), ps[-1]),
            )
        else:
            ans = sparse.csr_matrix((len(secl), ps[-1]))

        return ans

    @utils.logs_decorator
    def get_nXsecMat(self):
        """Get the nXsecMat matrix.

        This method calculates the nXsecMat matrix, which gives the fraction of neuron in a tet.

        Returns:
            sparse.csr_matrix: The nXsecMat matrix.
        """

        def gen_data():
            itot = 0
            for inc, nc in enumerate(self.ncs):
                for isec, _ in enumerate(self._gen_secs(nc)):
                    itot += 1
                    yield self.nc_vols[inc][1][isec] / self.nc_vols[inc][
                        0
                    ], inc, itot - 1

        # data is always a nX3 array:
        # col[0] is the ratio (dimensionless).
        # col[1] is the row at which it should be added in the sparse matrix
        # col[2] is the col at which it should be added in the sparse matrix
        data = np.array([i for i in gen_data()])

        if len(data) > 0 and data.shape[1] > 0:
            ans = sparse.csr_matrix(
                (data[:, 0], (data[:, 1], data[:, 2])),
                shape=(len(self.ncs), data.shape[0]),
            )
        else:
            ans = sparse.csr_matrix((len(self.ncs), data.shape[0]))

        return ans

    def update(self, conn_m):
        """Update removing GIDs and connection matrices.

        This method updates the removal of GIDs and the corresponding connection matrices. It is essential to pass the connection manager to update the connection matrices.

        Args:
            conn_m: The connection manager to update connection matrices.

        Returns:
            None
        """

        def gen_to_be_removed_segs():
            i = 0
            for nc in self.ncs:
                for sec in self._gen_secs(nc):
                    for seg in self._gen_segs(sec):
                        i += 1
                        if int(nc.CCell.gid) in self.removed_gids.keys():
                            yield i - 1

        to_be_removed = [isegs for isegs in gen_to_be_removed_segs()]
        if conn_m is not None:
            conn_m.delete_rows("nsegXtetMat", to_be_removed)
            conn_m.delete_cols("nXnsegMatBool", to_be_removed)

        to_be_removed = [
            inc
            for inc, nc in enumerate(self.ncs)
            if int(nc.CCell.gid) in self.removed_gids.keys()
        ]

        if conn_m is not None:
            conn_m.delete_rows("nXtetMat", to_be_removed)
            conn_m.delete_rows("nXnsegMatBool", to_be_removed)

        # keep this as last
        for i in ["ncs", "nc_vols", "nc_areas"]:
            self._update_attr(i, to_be_removed)

        self.check_neuron_removal_status()

    def check_neuron_removal_status(self):
        """
        Check the status of neuron removal and raise an exception if all neurons were removed.

        This method checks the number of neurons and removed GIDs and provides status messages and warnings.
        If all neurons were removed, it aborts the simulation.

        """
        removed_gids = comm.gather(self.removed_gids, root=0)
        num_neurons = self.num_neurons_per_rank

        if rank == 0:
            working_gids = [
                total - len(broken) for broken, total in zip(removed_gids, num_neurons)
            ]

            def rr(t, w):
                r = f"{w}/{t}"
                if t == 0 or w > 0:
                    return r
                else:
                    return f"\033[1;31m{r}\033[m"

            ratios = [
                rr(total, working)
                for working, total in zip(working_gids, num_neurons)
                if total
            ]

            logging.info(f"Working GIDs to Total GIDs Ratio:\n{', '.join(ratios)}")

            logging.info("GIDs that failed:")
            for r, removal_reasons in enumerate(removed_gids):
                for gid, reason in removal_reasons.items():
                    logging.info(f"Rank {r}, GID {gid}: {reason}")

            if sum(working_gids) == 0:
                print(
                    "All the neurons were removed! There is probably something fishy going on here",
                    flush=True,
                )
                comm.Abort(1)

    def _update_attr(self, v, to_be_removed):
        """Update a class attribute based on the list of indices to be removed.

        This method updates a class attribute by removing elements at specific indices.

        Args:
            v: The class attribute to update (assumed to be a list).
            to_be_removed: A list of indices to remove from the attribute.

        Returns:
            None
        """
        if not hasattr(self, v):
            return

        attr = getattr(self, v)
        setattr(
            self,
            v,
            np.array([i for idx, i in enumerate(attr) if idx not in to_be_removed]),
        )

    def get_var(self, var, weight, filter=[]):
        """Get variable values per segment weighted by a specific factor (e.g., area or volume).

        This method retrieves variable values per segment, weighted by a specific factor (e.g., area or volume).

        Args:
            var: The variable to retrieve.
            weight: The factor for weighting (e.g., "area" or "volume").
            filter: A list of attributes to filter segments by.

        Returns:
            np.ndarray: An array containing the variable values per segment weighted by the specified factor.
        """

        def f(seg):
            w = 1
            if isinstance(weight, str):
                w = getattr(seg, weight)()

            return getattr(seg, var) * w

        ans = np.array(
            [
                f(seg)
                for nc in self.ncs
                for sec in self._gen_secs(nc, filter=filter)
                for seg in self._gen_segs(sec)
            ]
        )

        return ans

    def set_var(self, var, val, filter=[]):
        """Set a variable to a specific value for all segments.

        This method sets a variable to a specific value for all segments based on the provided filter.

        Args:
            var: The variable to set.
            val: The value to set the variable to.
            filter: A list of attributes to filter segments by.

        Returns:
            None
        """
        for inc, nc in enumerate(self.ncs):
            for sec in self._gen_secs(nc, filter=filter):
                for seg in self._gen_segs(sec):
                    setattr(seg, var, val[inc])

    def get_vasculature_path(self):
        """Get the path to the vasculature generated by VascCouplingB.

        Returns:
            str: The path to the vasculature.
        """
        return self.glio_vascular_manager.circuit_conf["VasculaturePath"]

    @utils.logs_decorator
    def get_vasc_radii(self):
        """Get vasculature radii generated by VascCouplingB.

        This method retrieves vasculature radii generated by VascCouplingB.

        Returns:
            Tuple: A tuple containing lists of vasculature IDs and radii.
        """
        manager = self.glio_vascular_manager
        astro_ids = manager._astro_ids

        def gen_vasc_ids(astro_id):
            # libsonata is 0 based
            endfeet = manager._gliovascular.afferent_edges(astro_id - 1)
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]

            if astrocyte.endfeet is None:
                return

            for i in manager._gliovascular.source_nodes(endfeet):
                if i is None:
                    continue
                # neurodamus is 1 based
                yield int(i + 1)

        def gen_radii(astro_id):
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
            if astrocyte.endfeet is None:
                return
            for sec in astrocyte.endfeet:
                yield sec(0.5).vascouplingB.Rad

        vasc_ids = [id for astro_id in astro_ids for id in gen_vasc_ids(astro_id)]
        radii = [r for astro_id in astro_ids for r in gen_radii(astro_id)]
        assert len(vasc_ids) == len(radii), f"{len(vasc_ids)} == {len(radii)}"

        return vasc_ids, radii
