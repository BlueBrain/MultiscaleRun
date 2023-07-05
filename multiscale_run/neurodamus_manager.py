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



class MsrNeurodamusManager:
    """Handles neurodamus and keep track of what neurons are working"""

    def __init__(self, sonata_path):
        logging.info("instantiate ndam")
        self.ndamus = neurodamus.Neurodamus(sonata_path,logging_level=None,enable_coord_mapping=True,cleanup_atexit=False,)
        logging.info("ndam sim init")
        self.ndamus.sim_init()
        logging.info("ndam is ready")

        self.set_managers()
        self.ncs = np.array([nc for nc in self.neuron_manager.cells])
        self.acs = np.array([nc for nc in self.astrocyte_manager.cells])
        self.nc_vols = self._cumulate_nc_sec_quantity("volume")
        self.nc_areas = self._cumulate_nc_sec_quantity("area")
        self.removed_gids = {}

    def remove_gids(self, failed_cells, conn_m):
        """Add gids to the removed_gids list"""

        s = "\n".join(f"{k}: {v}" for k, v in failed_cells.items())
        if len(s):
            utils.rank_print("failed gids:\n" + s)

        self.removed_gids |= failed_cells

        self.update(conn_m=conn_m)

    def dt(self):
        return self.ndamus._run_conf["Dt"]

    def duration(self):
        return self.ndamus._run_conf["Duration"]

    def set_managers(self):
        """
        Find useful node managers.
        Names are unreliable so this is my best effort to have something not ad-hoc
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
            if isinstance(i, neurodamus.ngv.AstrocyteManager)
            and i.total_cells > 0
        ][0]

        self.glio_vascular_manager = self.ndamus.circuits.get_edge_manager(
            "vasculature", "astrocytes", neurodamus.ngv.GlioVascularManager
        )

    @staticmethod
    def _gen_secs(nc, filter=[]):
        """Generator of filtered sections"""
        for sec in nc.CellRef.all:
            if not all(hasattr(sec, i) for i in filter):
                continue
            if not sec.n3d():
                continue
            yield sec

    @staticmethod
    def _gen_segs(sec):
        """Generator of segments"""
        for seg in sec.allseg():
            yield seg

    def _cumulate_nc_sec_quantity(self, f):
        """Returns a list of tuples [sum, l] where sum is the sum of the values in l and l is the sum of the function
        required on the segments f"""
        v = [
            [
                sum([getattr(seg, f)() for seg in self._gen_segs(sec)])
                for sec in self._gen_secs(nc)
            ]
            for nc in self.ncs
        ]
        return np.array([(sum(i), i) for i in v], dtype=object)

    def get_seg_points(self, scale):
        """get the extremes of every neuron segment

        Every rank owns the same structure at the end

        output: pts[rank] (list: nranks)
                    [gid] (dict: nneurons)
                        [isec] (list: nsecs)
                            points (nparray: sec.nseg+3, 3)

        check the relative check function for the reasoning behind these lengths
        """

        def get_seg_extremes(sec, loc2glob, scale):
            """Get extremes and roto-translate in global coordinates"""

            def get_local_seg_extremes(nseg, pp):
                """Compute the position of beginning and end of each compartment in a section

                Assumption: all the compartments have the same length

                Inputs:
                    - nseg: number of compartments. Notice that there are 2 "joint" compartments at the extremes of a section
                    used for connection with vol and area == 0
                    - pp is a nX4 matrix of positions of points. The first col give the relative position, (x in neuron) along the
                    axis. It is in the interval [0, 1]. The other 3 columns give x,y,z triplets of the points in a global system of
                    reference.

                Outputs:
                - a matrix 3Xn of the position of the extremes of every proper compartment (not the extremes)
                """

                def extend_extremes(v):
                    """This takes care of the extreme joint compartments"""
                    if len(v):
                        return [v[0], *v, v[-1]]
                    else:
                        return []

                pp = np.array(pp)
                x_rel, xp, yp, zp = pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3]
                x = np.linspace(0, 1, nseg + 1)
                xp_seg = extend_extremes(np.interp(x, x_rel, xp))
                yp_seg = extend_extremes(np.interp(x, x_rel, yp))
                zp_seg = extend_extremes(np.interp(x, x_rel, zp))

                return np.transpose([xp_seg, yp_seg, zp_seg])

            ll = [
                [sec.arc3d(i) / sec.L, sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                for i in range(sec.n3d())
            ]
            ans = get_local_seg_extremes(
                sec.nseg,
                ll,
            )
            ans = (
                np.array(
                    loc2glob(ans),
                    dtype=float,
                    order="C",
                )
                * scale
            )
            return ans

        return [
            get_seg_extremes(sec, nc.local_to_global_coord_mapping, scale)
            for nc in self.ncs
            for sec in self._gen_secs(nc)
        ]

    def get_nsecXnsegMat(self, pts):
        """nsecXnsegMat is a matrix that gives the fraction of neuron section in a tet"""

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
        """nsecXnsegMat is a matrix that gives the fraction of neuron in a tet"""

        def gen_data():
            itot = 0
            for inc, nc in enumerate(self.ncs):
                for isec, _ in enumerate(self._gen_secs(nc)):
                    itot += 1
                    yield self.nc_vols[inc][1][isec] / self.nc_vols[inc][0], inc, itot - 1

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
        """Update removing gids

        We also need to pass the connection manager to remove rows there too.
        We can also pass None to skip this part but it needs to be put to None explicitly to make sure that
        the operator knows what they are doing
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

        if len(self.ncs) == 0 and len(self.removed_gids):
            raise utils.MrException(
                f"All the gids of this rank were removed. This probably requires attention\nRemoved gids: {', '.join(str(i) for i in self.removed_gids)}"
            )
        

    def _update_attr(self, v, to_be_removed):
        """update class attribute assuming it is a list"""
        if not hasattr(self, v):
            return

        attr = getattr(self, v)
        setattr(
            self,
            v,
            np.array([i for idx, i in enumerate(attr) if idx not in to_be_removed]),
        )

    def get_var(self, var, weight, filter=[]):
        """get variable per segment weighted by weight (area, volume)"""

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
        """set var as val for all the segments"""
        for inc, nc in enumerate(self.ncs):
            for sec in self._gen_secs(nc, filter=filter):
                for seg in self._gen_segs(sec):
                    setattr(seg, var, val[inc])

    def get_vasculature_path(self):
        return self.glio_vascular_manager.circuit_conf["VasculaturePath"]

    @utils.logs_decorator
    def get_vasc_radii(self):
        """get vasculature radii generated by VascCouplingB"""
        manager = self.glio_vascular_manager
        astro_ids = manager._astro_ids

        def gen_vasc_ids(astro_id):
            endfeet = manager._gliovascular.afferent_edges(astro_id)
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]

            if astrocyte.endfeet is None:
                return
            for i in manager._gliovascular.source_nodes(endfeet):
                if i is None:
                    continue
                yield i

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
