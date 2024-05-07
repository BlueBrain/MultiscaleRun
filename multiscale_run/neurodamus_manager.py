import logging

import neurodamus
import numpy as np
from scipy import sparse

from . import utils


class MsrNeurodamusManager:
    """Handles neurodamus and keeps track of what neurons are working."""

    def __init__(self, config):
        """Initialize the MsrNeurodamusManager with the given configuration.

        Args:
            config: The configuration for the neurodamus manager.
        """
        logging.info("instantiate ndam")
        self.ndamus = neurodamus.Neurodamus(
            str(config.config_path),
            logging_level=config.multiscale_run.logging_level,
            enable_coord_mapping=True,
            cleanup_atexit=False,
            lb_mode="RoundRobin",
        )
        logging.info("ndam sim init")
        self.ndamus.sim_init()
        logging.info("ndam is ready")

        self.set_managers()
        # list[neurodamus.metype.Cell_V6]
        self.ncs = np.array([nc for nc in self.neuron_manager.cells])
        # useful for reporting
        self.num_neurons_per_rank = utils.comm().gather(len(self.ncs), root=0)
        self.init_ncs_len = len(self.ncs)
        self.acs = np.array([nc for nc in self.astrocyte_manager.cells])
        self.nc_weights = {
            k: self._cumulate_nc_sec_quantity(k) for k in ["volume", "area"]
        }
        self.removed_gids = {}

    @property
    def gids(self):
        """Convenience function to get the gids from ncs"""
        return [int(nc.CCell.gid) for nc in self.ncs]

    def set_managers(self):
        """Find useful node managers for neurons, astrocytes, and glio-vascular management.

        This method sets the neuron_manager, astrocyte_manager, and glio_vascular_manager attributes based on available node managers.

        Returns
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

    def gen_secs(self, nc: neurodamus.metype.Cell_V6, filter=None):
        """Generator of filtered sections for a neuron.

        This method generates filtered sections for a neuron based on the provided filter.

        Args:
            nc: A neuron to generate sections from.
            filter: An optional list of attributes to filter sections by.

        Yields:
            Filtered sections for the neuron.
        """
        filter = filter or []
        for sec in nc.CellRef.all:
            if not all(hasattr(sec, i) for i in filter):
                continue
            if not sec.n3d():
                continue
            yield sec

    def gen_segs(self, sec):
        """Generator of segments for a neuron section.

        This method generates segments for a neuron section.

        Args:
            sec: A neuron section.

        Yields:
            Segments in the neuron section.
        """
        for seg in sec:
            yield seg

    def _cumulate_nc_sec_quantity(self, f: str) -> tuple:
        """Calculate cumulative quantity for neuron sections.

        This method calculates a cumulative quantity for neuron sections, such as volume or area.

        Args:
            f: The quantity to calculate (e.g., "volume" or "area").

        Returns:
            tuple: A tuple containing two numpy arrays:
                np.ndarray: An array of sums for each neuron compartment.
                np.ndarray: An array of arrays containing individual values for each neuron compartment.
        """
        v = [
            np.array(
                [
                    sum([getattr(seg, f)() for seg in self.gen_segs(sec=sec)])
                    for sec in self.gen_secs(nc=nc)
                ],
                dtype=float,
            )
            for nc in self.ncs
        ]

        return np.array([sum(i) for i in v], dtype=float), np.array(v, dtype=object)

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
            for sec in self.gen_secs(nc=nc)
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

        Returns
            sparse.csr_matrix: The nXsecMat matrix.
        """

        def gen_data():
            itot = 0
            for inc, nc in enumerate(self.ncs):
                for isec, _ in enumerate(self.gen_secs(nc=nc)):
                    itot += 1
                    yield (
                        self.nc_weights["volume"][1][inc][isec]
                        / self.nc_weights["volume"][0][inc],
                        inc,
                        itot - 1,
                    )

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

    def gen_to_be_removed_segs(self):
        """Generate the segments ids that need to be removed."""
        i = 0
        for nc in self.ncs:
            for sec in self.gen_secs(nc=nc):
                for seg in self.gen_segs(sec=sec):
                    i += 1
                    if int(nc.CCell.gid) in self.removed_gids.keys():
                        # ndam 2 sonata off-by-one correction
                        yield i - 1

    def check_neuron_removal_status(self):
        """Check the status of neuron removal and raise an exception if all neurons were removed.

        This method checks the number of neurons and removed GIDs and provides status messages and warnings.
        If all neurons were removed, it aborts the simulation.

        """
        removed_gids = utils.comm().gather(self.removed_gids, root=0)
        num_neurons = self.num_neurons_per_rank

        if utils.rank0():
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
                raise utils.MsrException(
                    "All the neurons were removed! There is probably something fishy going on here"
                )

    def get_var(self, var: str, weight: str = None, filter=None):
        """Get variable values per segment weighted by a specific factor (e.g., area or volume).

        This method retrieves variable values per segment, weighted by a specific factor (e.g., area or volume).

        Args:
            var: The variable to retrieve.
            weight: The factor for weighting (e.g., "area" or "volume").
            filter: An optional list of attributes to filter segments by.

        Returns:
            np.ndarray: An array containing the variable values per segment weighted by the specified factor.
        """
        if weight is not None and weight not in self.nc_weights:
            raise KeyError(
                f"Unknown weight: {weight}. Possible keys: {', '.join(self.nc_weights)}"
            )

        def f(seg):
            w = 1
            if weight is not None:
                w = getattr(seg, weight)()

            return getattr(seg, var) * w

        ans = np.array(
            [
                f(seg)
                for nc in self.ncs
                for sec in self.gen_secs(nc=nc, filter=filter)
                for seg in self.gen_segs(sec=sec)
            ]
        )

        return ans

    def set_var(self, var: str, vals: list[float], filter=None):
        """Set a variable to a specific value for all segments.

        This method sets a variable to a specific value for all segments based on the provided filter.

        Args:
            var: The variable to set.
            val: The value to set the variable to. 1 value per neuron. It should be
                the average.
            filter: An optional list of attributes to filter segments by.

        Returns:
            None
        """
        for inc, nc in enumerate(self.ncs):
            for sec in self.gen_secs(nc=nc, filter=filter):
                for seg in self.gen_segs(sec=sec):
                    setattr(seg, var, vals[inc])

    def get_vasculature_path(self):
        """Get the path to the vasculature generated by VascCouplingB.

        Returns
            str: The path to the vasculature.
        """
        return self.glio_vascular_manager.circuit_conf["VasculaturePath"]

    @utils.logs_decorator
    def get_vasc_radii(self):
        """Get vasculature radii generated by VascCouplingB.

        This method retrieves vasculature radii generated by VascCouplingB.

        Returns
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
