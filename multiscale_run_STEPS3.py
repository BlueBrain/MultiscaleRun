#!/bin/env python
# inspired from http://steps.sourceforge.net/manual/diffusion.html
# to run: srun -Aproj40  special -mpi -python dualrun6.py

from __future__ import print_function
import logging
import numpy as np
import textwrap
import os
import steps
import steps.mpi as smpi
import steps.model as smodel
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as meshio
import steps.utilities.geom_decompose as gd
from neurodamus import Neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from mpi4py import MPI
comm = MPI.COMM_WORLD
import dualrun.timer.mpi as mt

# Memory tracking
import psutil

np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = 1.602176634e-19

REPORT_FLAG = False

TET_INDEX_DTYPE = np.uint32

which_mesh = os.getenv('which_mesh')
# The meshes for STEPS3 found in the split_* folders are for the API 2.
# Here, we are using STEPS API 1.
mesh_file_ = f'./steps_meshes/{which_mesh}/{which_mesh}.msh'

#################################################
# Model Specification
#################################################

dt_nrn2dt_steps: int = 4

class Geom:
    meshfile = mesh_file_
    compname = 'extra'


class Na:
    name = 'Na'
    conc_0 = 140  # (mM/L)
    diffname = 'diff_Na'
    diffcst = 2e-9
    current_var = 'ina'
    charge = 1 * ELEM_CHARGE


class Volsys0:
    name = 'extraNa'
    specs = (Na,)


#################################################
# Model Build
#################################################

def gen_model():
    mdl = smodel.Model()
    vsys = smodel.Volsys(Volsys0.name, mdl)
    na_spec = smodel.Spec(Na.name, mdl)
    diff = smodel.Diff(Na.diffname, vsys, na_spec)
    diff.setDcst(Na.diffcst)
    return mdl


def gen_geom():
    mesh = meshio.importGmsh(Geom.meshfile, scale=1)[0]
    ntets = mesh.countTets()
    comp = stetmesh.TmComp(Geom.compname, mesh, range(ntets))
    comp.addVolsys(Volsys0.name)
    return mesh, ntets


def init_solver(model, geom):
    rng = srng.create('mt19937', 512)
    rng.initialize(2903)

    import steps.mpi.solver as psolver
    tet2host = gd.linearPartition(geom, [1, 1, steps.mpi.nhosts])
    if steps.mpi.rank == 0:
        print("Number of tets: ", geom.ntets)
        # gd.validatePartition(geom, tet2host)
        # gd.printPartitionStat(tet2host)
    return psolver.TetOpSplit(model, geom, rng, psolver.EF_NONE, tet2host), tet2host


# returns a list of tet mappings for the neurons
# each segment mapping is a list of (tetnum, fraction of segment)
def get_sec_mapping(ndamus, mesh):
    neurSecmap = []

    micrometer2meter = 1e-6
    rank = comm.Get_rank()

    # Init bounding box
    inf: float = 1e15
    n_bbox_min = np.array([inf, inf, inf], dtype=float)
    n_bbox_max = np.array([-inf, -inf, -inf], dtype=float)

    cell_manager = ndamus.circuits.base_cell_manager

    for c in cell_manager.cells:
        # Get local to global coordinates transform

        #print("DIR_LOCAL_NODES: ", dir(cell_manager.local_nodes) )
        #print("DIR_META: ", dir(cell_manager.local_nodes.meta) )
        #print("LENGTH META",len(cell_manager.local_nodes._gid_info))
        #print("DIR_GID: ", cell_manager.local_nodes._gid_info.get(c.gid,"No c.gid found in meta!!!")  )

        # FIXED in latest neurodamus
        # To test do: export PYTHONPATH=$PWD/dev/neurodamus-py:$PYTHONPATH
        loc_2_glob_tsf = c.local_to_global_coord_mapping

        secs = [sec for sec in c.CCell.all if hasattr(sec, Na.current_var)]
        # Our returned struct is #neurons long list of
        # [(sec1, nparray([pt1, pt2, ...])), (sec2, npa...]
        secmap = []
        for sec in secs:
            npts = sec.n3d()

            if not npts:
                # logging.warning("Sec %s doesnt have 3d points", sec)
                continue
            fractmap = []

            # Store points in section
            pts = np.empty((npts, 3), dtype=float)

            for i in range(npts):
                pts[i] = np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]) * micrometer2meter

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = np.array(loc_2_glob_tsf(pts), dtype=float, order='C')

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)

            # get tet to fraction map, batch processing all points (segments)
            tet2fraction_map = mesh.intersect(pts_abs_coo)

            # store map for each section
            secmap.append((sec, tet2fraction_map))

        neurSecmap.append(secmap)

    # Reduce Neuron bbox
    n_bbox_min_glo = np.empty((3), dtype=float)
    n_bbox_max_glo = np.empty((3), dtype=float)
    comm.Reduce([n_bbox_min, 3, MPI.DOUBLE], [n_bbox_min_glo, 3, MPI.DOUBLE], \
        op=MPI.MIN, root=0)
    comm.Reduce([n_bbox_max, 3, MPI.DOUBLE], [n_bbox_max_glo, 3, MPI.DOUBLE], \
        op=MPI.MAX, root=0)

    # Check bounding boxes
    if rank == 0:
        s_bbox_min = mesh.getBoundMin()
        s_bbox_max = mesh.getBoundMax()

        print("bounding box Neuron:", n_bbox_min_glo, n_bbox_max_glo)
        print("bounding box STEPS:", s_bbox_min, s_bbox_max)

        # Should add tolerance to check bounding box
        if np.less(n_bbox_min_glo, s_bbox_min).any() or np.greater(n_bbox_max_glo, s_bbox_max).any():
            logging.warning("STEPS mesh does not overlap with all neurons")

    return neurSecmap


##############################################
# Runtime
##############################################

def timesteps(end: float, step: float):
    return ((i+1) * step for i in range(int(end/step)))


def main():
    rss = [] # Memory tracking

    ndamus = Neurodamus("BlueConfig", enable_reports=False, logging_level=None, enable_coord_mapping=True)

    # Simulate one molecule each 10e9

    # Times are in ms
    DT = ndamus._run_conf['Dt']
    SIM_END = ndamus._run_conf['Duration']
    DT_s = DT * 1e3 * dt_nrn2dt_steps

    # In steps use M/L and apply the SIM_REAL ratio
    CONC_FACTOR = 1e-9

    AVOGADRO = 6.02e23
    COULOMB = 6.24e18
    CA = COULOMB/AVOGADRO*CONC_FACTOR*DT_s


    log_stage("Initializing steps model and geom...")
    model = gen_model()
    tmgeom, ntets = gen_geom()

    logging.info("Computing segments per tet...")
    neurSecmap = get_sec_mapping(ndamus, tmgeom)


    with mt.timer.region('init_sims'):
        logging.info("Initializing simulations...")
        ndamus.sim_init()
        steps_sim, tet2host = init_solver(model, tmgeom)
        steps_sim.reset()
        # there are 0.001 M/mM
        steps_sim.setCompSpecConc(Geom.compname, Na.name, 1e-21 * Na.conc_0 * CONC_FACTOR)
        tetVol = np.array([tmgeom.getTetVol(x) for x in range(ntets)], dtype=float)

    log_stage("===============================================")
    log_stage("Running both STEPS and Neuron simultaneously...")

    rank: int = comm.Get_rank()
    if rank == 0 and REPORT_FLAG:
        f = open("tetConcs.txt", "w+")

    def fract(neurSecmap):
        # compute the currents arising from each segment into each of the tets
        tet_currents = np.zeros((ntets,), dtype=float)
        for secmap in neurSecmap:
            for sec, tet2fraction_map in secmap:
                for seg, tet2fraction in zip(sec.allseg(), tet2fraction_map):
                    if tet2fraction:
                        tet: int
                        fract: float
                        for tet, fract in tet2fraction:
                            # there are 1e8 µm2 in a cm2, final output in mA
                            tet_currents[tet] += seg.ina * seg.area() * 1e-8 * fract
        return tet_currents

    index = np.array(range(ntets), dtype=TET_INDEX_DTYPE)
    tetConcs = np.zeros((ntets,), dtype=float)

    # allreduce comm buffer
    tet_currents_all = np.zeros((ntets,), dtype=float)

    steps = 0
    for t in ProgressBar(int(SIM_END / DT))(timesteps(SIM_END, DT)):
        steps += 1

        with mt.timer.region('neuron_cum'):
            ndamus.solve(t)

        if steps % dt_nrn2dt_steps == 0:
            with mt.timer.region('steps_cum'):
                steps_sim.run(t / 1000)  # ms to sec


            with mt.timer.region('processing'):
                tet_currents = fract(neurSecmap)

                with mt.timer.region('comm_allred_currents'):
                    comm.Allreduce(tet_currents, tet_currents_all, op=MPI.SUM)

                # update the tet concentrations according to the currents
                steps_sim.getBatchTetConcsNP(index, Na.name, tetConcs) # in this method there is an allreduce which I think is unnecessary
                # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
                tetConcs = tetConcs + tet_currents_all * CA * tetVol
                tetConcs[tetConcs<0] = 0
                steps_sim.setBatchTetConcsNP(index, Na.name, tetConcs)
                if rank == 0 and REPORT_FLAG:
                    f.write(" ".join(("%e" % x for x in tetConcs)))
        
        rss.append(psutil.Process().memory_info().rss / 1024**2) # memory consumption in MB
    
    if rank == 0 and REPORT_FLAG:
        f.close()

    mt.timer.print()

    rss = np.mean(rss)
    avg_rss = comm.allreduce(rss, MPI.SUM) / smpi.nhosts
    if rank == 0:
        print(f"average (across ranks) memory consumption [MB]: {avg_rss}")

if __name__ == "__main__":
    main()
    exit() # needed to avoid hanging

