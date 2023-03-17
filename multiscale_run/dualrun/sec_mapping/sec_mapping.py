from mpi4py import MPI
import numpy as np
import logging

import steps


def seg_extremes(nseg, pp):
    """ Compute the position of beginning and end of each compartment in a section

    Assumption: all the compartments have the same length

    Inputs:
        - nseg: number of compartments. Notice that there are 2 "joint" compartments at the extremes of a section
        used for connection with vol and area == 0
        - pp is a nX4 matrix of positions of points. The first col give the relative position, (x in neuron) along the
        axis. It is in the interval [0, 1]. The other 3 columns give x,y,z triplets of the points in a global system of
        reference.

    Outputs:
    - a matrix 3Xn of the position of the extremes of every proper compartment (not the extremes) """

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


# returns a list of tet mappings for the neurons
# each segment mapping is a list of (tetnum, fraction of segment)
def get_sec_mapping_collective_single_mesh(ndamus, mesh, attr1):
    comm = MPI.COMM_WORLD

    neurSecmap = []

    micrometer2meter = 1e-6
    rank = comm.Get_rank()

    # Init bounding box
    n_bbox_min = np.array([np.Inf, np.Inf, np.Inf], dtype=float)
    n_bbox_max = np.array([-np.Inf, -np.Inf, -np.Inf], dtype=float)

    cell_manager = ndamus.circuits.base_cell_manager

    for c in cell_manager.cells:
        # Get local to global coordinates transform

        # print("DIR_LOCAL_NODES: ", dir(cell_manager.local_nodes) )
        # print("DIR_META: ", dir(cell_manager.local_nodes.meta) )
        # print("LENGTH META",len(cell_manager.local_nodes._gid_info))
        # print("DIR_GID: ", cell_manager.local_nodes._gid_info.get(c.gid,"No c.gid found in meta!!!")  )

        # FIXED in latest neurodamus
        # To test do: export PYTHONPATH=$PWD/dev/neurodamus-py:$PYTHONPATH
        loc_2_glob_tsf = c.local_to_global_coord_mapping

        secs = (sec for sec in c.CCell.all if hasattr(sec, attr1))
        # Our returned struct is #neurons long list of
        # [(sec1, nparray([pt1, pt2, ...])), (sec2, npa...]
        secmap = []
        for sec in secs:
            npts = sec.n3d()

            if not npts:
                # logging.warning("Sec %s doesnt have 3d points", sec)
                continue

            pts = seg_extremes(
                sec.nseg,
                [
                    [sec.arc3d(i) / sec.L, sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                    for i in range(npts)
                ],
            )

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = (
                np.array(loc_2_glob_tsf(pts), dtype=float, order="C") * micrometer2meter
            )

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)

            # store map for each section
            secmap.append(
                (
                    sec,
                    [
                        [(tet.idx, rat) for tet, rat in seg]
                        for seg in mesh.intersect(pts_abs_coo)
                    ],
                )
            )

        neurSecmap.append(secmap)

    # Reduce Neuron bbox
    n_bbox_min_glo = np.empty((3), dtype=float)
    n_bbox_max_glo = np.empty((3), dtype=float)
    comm.Reduce(
        [n_bbox_min, 3, MPI.DOUBLE], [n_bbox_min_glo, 3, MPI.DOUBLE], op=MPI.MIN, root=0
    )
    comm.Reduce(
        [n_bbox_max, 3, MPI.DOUBLE], [n_bbox_max_glo, 3, MPI.DOUBLE], op=MPI.MAX, root=0
    )

    # Check bounding boxes
    if rank == 0:
        # no need to have "with asLocal():" since it is a single mesh
        s_bbox_min, s_bbox_max = mesh.bbox.min, mesh.bbox.max

        print(
            "bounding box Neuron [um] : ",
            n_bbox_min_glo / micrometer2meter,
            n_bbox_max_glo / micrometer2meter,
            flush=True,
        )
        print(
            "bounding box STEPS [um] : ",
            np.array(s_bbox_min) / micrometer2meter,
            np.array(s_bbox_max) / micrometer2meter,
            flush=True,
        )

        # Should add tolerance to check bounding box
        if (
            np.less(n_bbox_min_glo, s_bbox_min).any()
            or np.greater(n_bbox_max_glo, s_bbox_max).any()
        ):
            logging.warning("STEPS mesh does not overlap with all neurons")

    return neurSecmap


def get_sec_mapping_collective_partitioned_mesh(ndamus, mesh, attr1):
    comm = MPI.COMM_WORLD

    neurSecmap = []

    micrometer2meter = 1e-6
    rank = comm.Get_rank()
    nhosts = comm.Get_size()

    # Init bounding box
    n_bbox_min = np.array([np.Inf, np.Inf, np.Inf], dtype=float)
    n_bbox_max = np.array([-np.Inf, -np.Inf, -np.Inf], dtype=float)

    pts_abs_coo_glo = (
        []
    )  # list of numpy arrays (each array holds the 3D points of a section)
    cell_manager = ndamus.circuits.base_cell_manager

    for c in cell_manager.cells:
        # Get local to global coordinates transform
        loc_2_glob_tsf = c.local_to_global_coord_mapping
        secs = (sec for sec in c.CCell.all if hasattr(sec, attr1))
        for sec in secs:
            npts = sec.n3d()

            if not npts:
                # logging.warning("Sec %s doesnt have 3d points", sec)
                continue

            pts = seg_extremes(
                sec.nseg,
                [
                    [sec.arc3d(i) / sec.L, sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                    for i in range(npts)
                ],
            )

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = (
                np.array(loc_2_glob_tsf(pts), dtype=float, order="C") * micrometer2meter
            )
            pts_abs_coo_glo.append(pts_abs_coo)

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)

    # Check bounding boxes (local -> Mesh is distributed across ranks!)

    with mesh.asLocal():
        s_bbox_min, s_bbox_max = mesh.bbox.min, mesh.bbox.max

    # For small networks, some MPI tasks may get zero neurons. Therefore, the BB seems to span from -inf to inf
    print(f"{rank} bounding box Neuron [um] : ",
    n_bbox_min / micrometer2meter,
    n_bbox_max / micrometer2meter,
    flush=True,
    )
    print(f"{rank} bounding box STEPS [um] (local, per rank): ",
    np.array(s_bbox_min) / micrometer2meter,
    np.array(s_bbox_max) / micrometer2meter,
    flush=True,
    )

    # all MPI tasks process all the points/segments (STEPS side - naive approach as a first step)
    # list of lists: External list refers to the task, internal is the list created above
    pts_abs_coo_glo = comm.allgather(pts_abs_coo_glo)
    # after this point, pts_abs_coo_glo is exactly the same for each rank

    # list of intersections: all sections (from all the ranks) are checked for intersection with the mesh of this rank!

    intersect_struct = []
    for task in pts_abs_coo_glo:
        for sec in task:
            intersect_struct.append(
                [[(tet.idx, rat) for tet, rat in seg] for seg in mesh.intersect(sec)]
            )
    # The total size of this list is len(pts_abs_coo_glo:task[0]) + ... + len(task[nhosts-1]) (where task, check the inner loop above)
    # -> same length for every rank (same structure, not same content, though)

    # list of lists: External list refers to the task, internal is the list created above
    intersect_struct = comm.allgather(intersect_struct)
    # after this point, intersect_struct is exactly the same for each rank

    for c in cell_manager.cells:
        secs = (sec for sec in c.CCell.all if hasattr(sec, attr1))
        secmap = []
        isec = 0
        for sec in secs:
            npts = sec.n3d()
            if not npts:
                continue

            # The sections that belong to this rank (neurodamus side) were intersected with all meshes (STEPS side)
            # We need to find for every rank where this processing happened in the intersect_struct container
            start_from = 0
            for task in range(rank):
                start_from += len(pts_abs_coo_glo[task])

            for task in range(nhosts):
                # store map for each section
                secmap.append((sec, intersect_struct[task][start_from + isec]))

            isec += 1

        neurSecmap.append(secmap)

    return neurSecmap


def get_sec_mapping_collective(ndamus, mesh, attr1):
    if isinstance(mesh, steps.API_2.geom.DistMesh) and MPI.COMM_WORLD.Get_size() > 1:
        return get_sec_mapping_collective_partitioned_mesh(ndamus, mesh, attr1)
    else:
        return get_sec_mapping_collective_single_mesh(ndamus, mesh, attr1)


# compute the currents arising from each segment into each of the tets
def fract_collective(neurSecmap, ntets, global_inds):
    """
    global_inds : maps the global indices from 0 to ntet-1 (https://github.com/CNS-OIST/HBP_STEPS/issues/890)
    """
    tet_currents = np.zeros((ntets,), dtype=float)

    for secmap in neurSecmap:
        for sec, tet2fraction_map in secmap:
            for seg, tet2fraction in zip(sec.allseg(), tet2fraction_map):
                if tet2fraction:
                    for tet, fract in tet2fraction:
                        # there are 1e8 µm2 in a cm2, final output in mA
                        tet_currents[global_inds[tet]] += (
                            seg.ina * seg.area() * 1e-8 * fract
                        )
    return tet_currents
