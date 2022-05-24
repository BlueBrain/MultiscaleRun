# HIGHLIGHT:
# Here we are using STEPS4, i.e. the tet mesh is distributed across the MPI tasks/ranks.
# The difficulty for the coupling of Neurodamus and STEPS4 comes from the fact that
# the neurons and the tets (for the same task) may not overlap (spatially). This is because
# the load balancing of Neurodamus and STEPS is completely decoupled.

import sys
import os
import time
import logging
import numpy as np

# STEPS related
import steps.interface
from steps.model import *
from steps.geom import *
from steps.sim import *
from steps.saving import *
from steps.rng import *

# Neurodamus related
from neurodamus import Neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage

# Dualrun related
import dualrun.timer.mpi as mt

from mpi4py import MPI as mpi
comm = mpi.COMM_WORLD

# Memory tracking
import psutil

np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = 1.602176634e-19

REPORT_FLAG = False

TET_INDEX_DTYPE = np.int64 # Global indices from Omega_H

which_STEPS = int(os.getenv('which_STEPS'))
which_mesh = os.getenv('which_mesh')
mesh_file_ = f'./steps_meshes/{which_mesh}/split_{MPI.nhosts}/steps{which_STEPS}/{which_mesh}'

#################################################
# Model Specification
#################################################

dt_nrn2dt_steps: int = 4

class Geom:
    meshfile = mesh_file_


class Na:
    conc_0 = 140  # (mM/L)
    diffcst = 2e-9
    current_var = 'ina'
    charge = 1 * ELEM_CHARGE


# returns a list of tet mappings for the neurons
# each segment mapping is a list of (tetnum, fraction of segment)
def get_sec_mapping(ndamus, mesh):
    neurSecmap = []

    micrometer2meter = 1e-6
    rank = comm.Get_rank()

    # Init bounding box
    n_bbox_min = np.array([np.Inf, np.Inf, np.Inf], dtype=float)
    n_bbox_max = np.array([-np.Inf, -np.Inf, -np.Inf], dtype=float)

    pts_abs_coo_glo = [] # list of numpy arrays (each array holds the 3D points of a section)
    cell_manager = ndamus.circuits.base_cell_manager
    
    for c in cell_manager.cells:
        # Get local to global coordinates transform
        loc_2_glob_tsf = c.local_to_global_coord_mapping
        secs = [sec for sec in c.CCell.all if hasattr(sec, Na.current_var)]
        for sec in secs:
            npts = sec.n3d()

            if not npts:
                # logging.warning("Sec %s doesnt have 3d points", sec)
                continue

            pts = np.empty((npts, 3), dtype=float)
            for i in range(npts):
                pts[i] = np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]) * micrometer2meter

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = np.array(loc_2_glob_tsf(pts), dtype=float, order='C')
            pts_abs_coo_glo.append(pts_abs_coo)

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)
   
    # Check bounding boxes (local -> Mesh is distributed across ranks!)
    s_bbox_min = mesh.getBoundMin(local=True)
    s_bbox_max = mesh.getBoundMax(local=True)

    print(f"{rank} bounding box Neuron:", n_bbox_min, n_bbox_max)
    print(f"{rank} bounding box STEPS:", s_bbox_min, s_bbox_max)

    # all MPI tasks process all the points/segments (STEPS side - naive approach as a first step)
    # list of lists: External list refers to the task, internal is the list created above
    pts_abs_coo_glo = comm.allgather(pts_abs_coo_glo)
    # after this point, pts_abs_coo_glo is exactly the same for each rank

    # list of intersections: all sections (from all the ranks) are checked for intersection with the mesh of this rank!    
    intersect_struct = []
    for task in pts_abs_coo_glo:
        for sec in task:
            intersect_struct.append(mesh.intersect(sec))
    # The total size of this list is len(pts_abs_coo_glo:task[0]) + ... + len(task[nhosts-1]) (where task, check the inner loop above)
    # -> same length for every rank (same structure, not same content, though)

    # list of lists: External list refers to the task, internal is the list created above
    intersect_struct = comm.allgather(intersect_struct)
    # after this point, intersect_struct is exactly the same for each rank

    for c in cell_manager.cells:
        secs = [sec for sec in c.CCell.all if hasattr(sec, Na.current_var)]
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
            
            for task in range(MPI.nhosts):
                # store map for each section
                secmap.append((sec, intersect_struct[task][start_from + isec]))
            
            isec += 1
        
        neurSecmap.append(secmap)

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
    ########################### BIOCHEMICAL MODEL ###############################
    model = Model()
    with model:
        SNA = Species.Create()
        # Vol system
        vsys = VolumeSystem.Create()
        with vsys:
            # The diffusion rule
            DNA = Diffusion.Create(SNA, Na.diffcst)

    ########### MESH & COMPARTMENTALIZATION #################
    mesh = DistMesh(Geom.meshfile, scale=1)
    with mesh:
        compartment = Compartment.Create(vsys, tetLst=mesh.tets)
        ntets_tot = mesh.stepsMesh.total_num_elems # TOTAL Number of elements/tets (across all ranks)
        ntets_loc = mesh.stepsMesh.num_elems # Number of elements/tets owned by this process
        print(f"rank:{comm.Get_rank()}, ntets_tot:{ntets_tot} - ntets_loc:{ntets_loc}", flush=True)

    logging.info("Computing segments per tet...")
    neurSecmap = get_sec_mapping(ndamus, mesh.stepsMesh)

    with mt.timer.region('init_sims'):
        logging.info("Initializing simulations...")
        # Neurodamus
        ndamus.sim_init()
        # STEPS
        rng = RNG('mt19937', 512, int(time.time()%4294967295))
        steps_sim = Simulation('DistTetOpSplit', model, mesh, rng, searchMethod=NextEventSearchMethod.GIBSON_BRUCK)
        steps_sim.newRun()
        steps_sim.solver.setCompSpecConc("compartment", "SNA",  1e-21 * Na.conc_0 * CONC_FACTOR)
        tetVol = np.array([mesh.stepsMesh.getTetVol(TET_INDEX_DTYPE(x), local=False) for x in range(ntets_tot)], dtype=float)

    log_stage("===============================================")
    log_stage("Running both STEPS and Neuron simultaneously...")

    rank: int = comm.Get_rank()
    if rank == 0 and REPORT_FLAG:
        f = open("tetConcs.txt", "w+")

    def fract(neurSecmap):
        # compute the currents arising from each segment into each of the tets
        tet_currents = np.zeros((ntets_tot,), dtype=float)
        for secmap in neurSecmap:
            for sec, tet2fraction_map in secmap:
                for seg, tet2fraction in zip(sec.allseg(), tet2fraction_map):
                    if tet2fraction:
                        # KEEP IN MIND: tet is a global index (GO from STEPS)!
                        for tet, fract in tet2fraction:
                            # there are 1e8 µm2 in a cm2, final output in mA
                            tet_currents[tet] += seg.ina * seg.area() * 1e-8 * fract
        return tet_currents

    index = np.array(range(ntets_tot), dtype=TET_INDEX_DTYPE)
    tetConcs = np.zeros((ntets_tot,), dtype=float)

    # allreduce comm buffer
    tet_currents_all = np.zeros((ntets_tot,), dtype=float)

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
                    comm.Allreduce(tet_currents, tet_currents_all, op=mpi.SUM)

                # update the tet concentrations according to the currents
                steps_sim.solver.getBatchTetSpecConcsNP(index, "SNA", tetConcs) # in this method there is an allreduce which I think is unnecessary
                # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
                tetConcs = tetConcs + tet_currents_all * CA * tetVol
                tetConcs[tetConcs<0] = 0
                steps_sim.solver.setBatchTetSpecConcsNP(index, "SNA", tetConcs)
                if rank == 0 and REPORT_FLAG:
                    f.write(" ".join(("%e" % x for x in tetConcs)))

        rss.append(psutil.Process().memory_info().rss / 1024**2) # memory consumption in MB
    
    if rank == 0 and REPORT_FLAG:
        f.close()

    mt.timer.print()

    rss = np.mean(rss)
    avg_rss = comm.allreduce(rss, mpi.SUM) / MPI.nhosts
    if MPI.rank == 0:
        print(f"average (across ranks) memory consumption [MB]: {avg_rss}")


if __name__ == "__main__":
    main()
    exit() # needed to avoid hanging
