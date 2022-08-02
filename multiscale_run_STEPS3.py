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
import dualrun.sec_mapping.sec_mapping as sec_mapping

# Memory tracking
import psutil

# triple-run related
import time
from collections import defaultdict
import collections
from contextlib import contextmanager
import pickle
import math
import h5py
import csv
import re
from neurodamus.connection_manager import SynapseRuleManager

np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = 1.602176634e-19

REPORT_FLAG = False

TET_INDEX_DTYPE = np.uint32

which_mesh = os.getenv('which_mesh')
# The meshes for STEPS3 found in the split_* folders are for the API 2.
# Here, we are using STEPS API 1.
mesh_file_ = f'./steps_meshes/{which_mesh}/{which_mesh}.msh'

dualrun_env = int(os.getenv('dualrun'))
triplerun_env = int(os.getenv('triplerun'))

#################################################
# Model Specification
#################################################

# ndam : neurodamus
dt_nrn2dt_steps: int = 100 # steps-ndam coupling
dt_nrn2dt_jl: int = 100 # jl-ndam coupling

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
    e_var = 'ena'
    nai_var = 'nai'

class K:
    name = 'K'
    conc_0 = 2 #3 it was 3 in Dan's example  # (mM/L)
    base_conc= 2 #3 it was 3 in Dan's example #sum here is 6 which is probably too high according to Magistretti #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_K'
    diffcst = 2e-9
    current_var = 'ik'
    ki_var = 'ki'
    charge = 1 * ELEM_CHARGE

'''
class ATP:
    name = 'ATP'
    conc_0 = 0.1
    base_conc= 1.4 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_ATP'
    diffcst = 2e-9
    charge = -3 * ELEM_CHARGE
    atpi_var = 'atpi'

class ADP:
    name = 'ADP'
    conc_0 = 0.0001
    base_conc= 0.03 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_ADP'
    diffcst = 2e-9
    charge = -2 * ELEM_CHARGE
    adpi_var = 'adpi'
'''

class Ca:
    name = 'Ca'
    conc_0 = 1e-5
    base_conc=4e-5
    diffname = 'diff_Ca'
    diffcst = 2e-9
    current_var = 'ica'
    charge = 2 * ELEM_CHARGE
    e_var = 'eca'
    cai_var = 'cai'


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
    # STEPS default length scale is m
    # NEURON default length scale is um
    mesh = meshio.importGmsh(Geom.meshfile, scale=1e-6)[0]
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


##############################################
# Triplerun related
##############################################

#for out file names
timestr = time.strftime("%Y%m%d%H")

# paths
path_to_results = "./RESULTS/"
path_to_metab_jl = "./metabolismndam/sim/metabolism_unit_models/"

#files

julia_code_file = path_to_metab_jl + "julia_gen_18feb2021.jl" 
u0_file = path_to_metab_jl + "u0forNdam/u0_Calv_ATP1p2_Nai10.txt"

ins_glut_file_output = path_to_results + f"dis_ins_r_glut_{timestr}.txt"
ins_gaba_file_output = path_to_results + f"dis_ins_r_gaba_{timestr}.txt"
outs_glut_file_output = path_to_results + f"dis_outs_r_glut_{timestr}.txt"
outs_gaba_file_output = path_to_results + f"dis_outs_r_gaba_{timestr}.txt"

param_out_file = path_to_results + f"dis_param_{timestr}.txt"
um_out_file = path_to_results + f"dis_um_{timestr}.txt"


#####
test_counter_seg_file = path_to_results + f"dis_test_counter_seg0_{timestr}.txt"
wrong_gids_testing_file = path_to_results + f"dis_wrong_gid_errors_{timestr}.txt"
err_solver_output = path_to_results + f"dis_solver_errors_{timestr}.txt"
#####
#voltages_per_gid_f = "./metabolismndam/in_data/voltages_per_gid.txt"
target_gids = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0.txt") # hex0 gids
exc_target_gids = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_exc_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0exc.txt")
inh_target_gids = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_inh_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0inh.txt")
target_gids_L1 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L1_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l1.txt")
target_gids_L2 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L2_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l2.txt")
target_gids_L3 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L3_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l3.txt")
target_gids_L4 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L4_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l4.txt")
target_gids_L5 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L5_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l5.txt")
target_gids_L6 = np.loadtxt("./metabolismndam/sim/gids_sets/O1_20190912_spines/mc2_L6_gids.txt") #for sscx: np.loadtxt("./metabolismndam/sim/gids_sets/hex0l6.txt")

########################################################################

#with open(voltages_per_gid_f,'r') as infile:
#    voltages_l = infile.readlines()

#voltages_per_gids = {}
#for line in voltages_l:
#    idx,v = line.split("\t")
#    voltages_per_gids[int(idx)] = float(v)

########################################################################

# MPI summation for dict
# https://stackoverflow.com/questions/31388465/summing-python-objects-with-mpis-allreduce 

def addCounter(counter1, counter2, datatype):
    for item in counter2:
        if item in counter1:
            counter1[item] += counter2[item]
        else:
            counter1[item] = counter2[item]
    return counter1

def addDict(d1, d2, datatype):  # d1 and d2 are from different ranks
    for s in d2:
        d1.setdefault(s, {})
        for t in d2[s]:
            d1[s][t] = d2[s][t] #10jan2021 #d1[s].get(t, 0.0) + d2[s][t]
    return d1

def joinDict(d1, d2, datatype):
    d1.update(d2)
    return d1

#################################################
# METABOLISM Model Build
#################################################

from diffeqpy import de
logging.info("diffeqpy imported")
from julia import Main ##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy
logging.info("main jl imported")
#from julia import Sundials
#import jl2py
#metabolism = jl2py.gen_metabolism_model(julia_code_file)
def gen_metabolism_model():
    '''import jl metabolism diff eq system code to py'''
    with open(julia_code_file, "r") as f:
        julia_code = f.read()
    metabolism = Main.eval(julia_code)
    return metabolism
logging.info("metabolism model read from file")

##############################################
# Runtime
##############################################

def timesteps(end: float, step: float):
    return ((i+1) * step for i in range(int(end/step)))


def main():
    rank: int = comm.Get_rank()

    rss = [] # Memory tracking

    ndamus = Neurodamus("BlueConfig", enable_reports=False, logging_level=None, enable_coord_mapping=True)

    # Simulate one molecule each 10e9

    # Times are in ms (for NEURON, because STEPS works with SI)
    DT = ndamus._run_conf['Dt'] #0.025  #ms i.e. = 25 usec which is timstep of ndam
    SIM_END = ndamus._run_conf['Duration'] #500.0 #10.0 #1000.0 #ms
    DT_s = DT * 1e3 * dt_nrn2dt_steps
    SIM_END_coupling_interval = DT*dt_nrn2dt_jl

    # In steps use M/L and apply the SIM_REAL ratio
    CONC_FACTOR = 1e-9

    AVOGADRO = 6.02e23
    COULOMB = 6.24e18
    CA = COULOMB/AVOGADRO*CONC_FACTOR*DT_s

    if triplerun_env:
        dictAddOp = MPI.Op.Create(addDict, commute=True)
        dictJoinOp = MPI.Op.Create(joinDict, commute=True)

        um = {}
        with open(u0_file,'r') as u0file:
            u0fromFile = [float(line.replace(" ","").split("=")[1].split("#")[0].strip()) for line in u0file if ((not line.startswith("#")) and (len(line.replace(" ","")) > 2 )) ]
        
        mito_volume_fraction = [0.0459, 0.0522, 0.064, 0.0774, 0.0575, 0.0403]
        mito_volume_fraction_scaled = []
        for mvfi in mito_volume_fraction:
            mito_volume_fraction_scaled.append(mvfi/max(mito_volume_fraction))

        glycogen_au = [128.0, 100.0, 100.0, 90.0, 80.0, 75.0]
        glycogen_scaled = []
        for glsi in glycogen_au:
            glycogen_scaled.append(glsi/max(glycogen_au))

        cells_volumes = {}
        logging.info("get volumes")

        for i, nc in enumerate(ndamus.circuits.base_cell_manager.cells):

            cells_volumes[int(nc.CCell.gid)] = 0

            secs_all = [sec for sec in nc.CCell.all]
            if len(secs_all) == 0:
                print("len_secs_all volumes: ", len(secs_all))

            for j, sec_elem in enumerate(secs_all):
                seg_all = sec_elem.allseg()
                for k, seg in enumerate(seg_all):
                    cells_volumes[int(nc.CCell.gid)] += seg.volume()

            #del secs_all

        gid_to_cell = {}
        for i, nc in enumerate(ndamus.circuits.base_cell_manager.cells):
            gid_to_cell[int(nc.CCell.gid)] = nc

        all_needs = comm.reduce({rank: set([int(i) for i in gid_to_cell.keys()])}, op=dictJoinOp, root=0)
        if rank == 0:
            all_needs.pop(0)

    with mt.timer.region('init_sims'):
        logging.info("Initializing simulations...")
        
        ndamus.sim_init()
        if dualrun_env:
            log_stage("Initializing steps model and geom...")
            model = gen_model()
            tmgeom, ntets = gen_geom()

            # In STEPS4 the global indices are not sequential and they can be over ntets.
            # This dictionary does a mapping from 0 to ntets. In STEPS3, there is no need,
            # and this is why we go with the identity map.
            global_inds = {i:i for i in range(ntets)}

            logging.info("Computing segments per tet...")
            neurSecmap = sec_mapping.get_sec_mapping_collective_STEPS3(ndamus, tmgeom, Na.current_var)

            steps_sim, tet2host = init_solver(model, tmgeom)
            steps_sim.reset()
            # there are 0.001 M/mM
            steps_sim.setCompSpecConc(Geom.compname, Na.name, 1e-3 * Na.conc_0 * CONC_FACTOR)
            tetVol = np.array([tmgeom.getTetVol(x) for x in range(ntets)], dtype=float)

    log_stage("===============================================")
    log_stage("Running both STEPS and Neuron simultaneously...")

    if rank == 0 and REPORT_FLAG:
        f = open("tetConcs.txt", "w+")

    if dualrun_env:
        index = np.array(range(ntets), dtype=TET_INDEX_DTYPE)
        tetConcs = np.zeros((ntets,), dtype=float)
        # allreduce comm buffer
        tet_currents_all = np.zeros((ntets,), dtype=float)

    steps = 0
    idxm = 0
    for t in ProgressBar(int(SIM_END / DT))(timesteps(SIM_END, DT)):
        steps += 1

        with mt.timer.region('neuron_cum'):
            ndamus.solve(t)

        if steps % dt_nrn2dt_steps == 0 and dualrun_env:
            with mt.timer.region('steps_cum'):
                steps_sim.run(t / 1000)  # ms to sec


            with mt.timer.region('processing'):
                tet_currents = sec_mapping.fract_collective(neurSecmap, ntets, global_inds)

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
        

        if (steps % dt_nrn2dt_jl == 0) and triplerun_env:
            outs_r_glu = {}
            outs_r_gaba = {}    
            
            collected_num_releases_glutamate = {}
            collected_num_releases_gaba = {}

            synapse_managers = [ manager for manager in ndamus.circuits.base_cell_manager.connection_managers.values() if isinstance(manager, SynapseRuleManager)  ]
            #print("synapse_managers DONE",len(synapse_managers))
            for syn_manager in synapse_managers:
                #print("SYNMAN")
                for conn in syn_manager.all_connections():
                    num_releases_glutamate = 0 # 12jan2021
                    num_releases_gaba = 0 # 12jan2021
                    if conn.sgid in target_gids: # 12jan2021
                        collected_num_releases_glutamate.setdefault(conn.sgid, {})
                        collected_num_releases_gaba.setdefault(conn.sgid, {})
                    
                        for syn in conn._synapses:
                            if hasattr(syn, 'A_AMPA_step'):
                                num_releases_glutamate += syn.release_accumulator
                                syn.release_accumulator = 0.0
                            elif hasattr(syn, 'A_GABAA_step'):
                                num_releases_gaba += syn.release_accumulator
                                syn.release_accumulator = 0.0
                        if isinstance(conn.sgid, float) or isinstance(conn.tgid, float):
                            raise Exception(f"Rank {rank} ids {conn.sgid} {conn.tgid} have floats!")

                        collected_num_releases_glutamate[conn.sgid][conn.tgid]=num_releases_glutamate
                        collected_num_releases_gaba[conn.sgid][conn.tgid]=num_releases_gaba
                        
                #del conn
                #del syn
                #del num_releases_glutamate # 26jan2021
                #del num_releases_gaba # 26jan2021

            ##################### METABOLISM RUN NOW!
            comm.Barrier()
            sum_t = {}
            for s in collected_num_releases_glutamate:
                for t in collected_num_releases_glutamate[s]:
                    sum_t.setdefault(t, 0.0)
                    sum_t[t] += collected_num_releases_glutamate[s][t]
            #if idxm % 10 == 0:
            with open(ins_glut_file_output, 'a') as f:
                for t in sum_t:
                    f.write(f"{idxm}\t{rank}\t{t}\t{sum_t[t]}\n")
            #del sum_t

            comm.Barrier()

            sum_t = {}
            for s in collected_num_releases_gaba:
                for t in collected_num_releases_gaba[s]:
                    sum_t.setdefault(t, 0.0)
                    sum_t[t] += collected_num_releases_gaba[s][t]

            #if idxm % 10 == 0:
            with open(ins_gaba_file_output, 'a') as f:
                for t in sum_t:
                    f.write(f"{idxm}\t{rank}\t{t}\t{sum_t[t]}\n")
            #del sum_t


            logging.info("barrier before start all_events")       
            comm.Barrier()
                        
            logging.info("start all_events glu")
            all_events_glu = comm.reduce(collected_num_releases_glutamate, op=dictAddOp, root=0)
            logging.info("start on rank 0 Glu")
            if rank == 0:
                for r, needs in all_needs.items():
                    events = {s: all_events_glu[s]  for s in needs if s in all_events_glu}
                    comm.send(events, dest=r)
                received_events_glu = {s: all_events_glu[s] for s in gid_to_cell if s in all_events_glu}
            else:
                received_events_glu = comm.recv(source=0)
            comm.Barrier()

            if rank == 0:  
                all_outs_r_glu = {}
                for sgid, tv in all_events_glu.items():
                    for tgid, v in tv.items():
                        all_outs_r_glu[sgid] = all_outs_r_glu.get(sgid, 0.0) + v
                #if idxm % 10 == 0:
                with open(outs_glut_file_output, 'a') as f:
                    for sgid, v in all_outs_r_glu.items():
                        f.write(f"{idxm}\t{sgid}\t{v}\n")
                #del all_outs_r_glu
            #del all_events_glu

            for s, tv in received_events_glu.items():
                collected_num_releases_glutamate.setdefault(s, {})
                for t, v in tv.items():
                    if t in gid_to_cell:
                        continue
                    collected_num_releases_glutamate[s][t] = v
            comm.Barrier()
            #del received_events_glu

            for sgid, tv in collected_num_releases_glutamate.items():
                for tgid, v in tv.items():
                    outs_r_glu[sgid] = outs_r_glu.get(sgid, 0.0) + v
            comm.Barrier()
            #del collected_num_releases_glutamate # 23jan2021


            all_events_gaba = comm.reduce(collected_num_releases_gaba, op=dictAddOp, root=0)
            logging.info("start on rank 0 GABA")
            if rank == 0:
                for r, needs in all_needs.items():
                    events = {s: all_events_gaba[s]  for s in needs if s in all_events_gaba}
                    comm.send(events, dest=r)

                received_events_gaba = {s: all_events_gaba[s] for s in gid_to_cell if s in all_events_gaba}
            else:
                received_events_gaba = comm.recv(source=0)
            comm.Barrier()

            if rank == 0:  
                all_outs_r_gaba = {}
                for sgid, tv in all_events_gaba.items():
                    for tgid, v in tv.items():
                        all_outs_r_gaba[sgid] = all_outs_r_gaba.get(sgid, 0.0) + v
                #if idxm % 10 == 0:
                with open(outs_gaba_file_output, 'a') as f:
                    for sgid, v in all_outs_r_gaba.items():
                        f.write(f"{idxm}\t{sgid}\t{v}\n")
                #del all_outs_r_gaba
            #del all_events_gaba

            for s, tv in received_events_gaba.items():
                collected_num_releases_gaba.setdefault(s, {})
                for t, v in tv.items():
                    if t in gid_to_cell:
                        continue
                    collected_num_releases_gaba[s][t] = v
            comm.Barrier()
            #del received_events_gaba

            for sgid, tv in collected_num_releases_gaba.items():
                for tgid, v in tv.items():
                    outs_r_gaba[sgid] = outs_r_gaba.get(sgid, 0.0) + v
            comm.Barrier()
            #del collected_num_releases_gaba # 23jan2021

            comm.Barrier()


            #logging.info("get ions from ndam")

            nais_mean = {}
            ina_density = {}
            kis_mean = {}
            ik_density = {}
            cais_mean = {}

            #atpi_mean = {}
            #adpi_mean = {}
            #cells_areas = {}
            #cells_volumes = {}

            current_ina = {}
            current_ik = {}
            #current_ica = {}

            nais = {}
            cais = {}
            kis = {}
            #atpi = {}
            #adpi = {}

            for c_gid, nc in gid_to_cell.items():
                if c_gid not in target_gids:
                    print("gid_not_in_list")
                    continue

                #counter_seg = {}
                counter_seg_Na = {}
                counter_seg_K = {}
                counter_seg_Ca = {}
                #counter_seg_ATP = {}
                #counter_seg_ADP = {}

                cells_volumes_Na = {}
                cells_areas_Na = {}
                cells_volumes_K = {}
                cells_areas_K = {}
                cells_volumes_Ca = {}
                #cells_volumes_ATP = {}
                #cells_volumes_ADP = {}

                #counter_seg.setdefault(c_gid, 0.0)
                counter_seg_Na.setdefault(c_gid, 0.0)
                counter_seg_K.setdefault(c_gid, 0.0)
                counter_seg_Ca.setdefault(c_gid, 0.0)
                #counter_seg_ATP.setdefault(c_gid, 0.0)
                #counter_seg_ADP.setdefault(c_gid, 0.0)

                cells_volumes_Na.setdefault(c_gid, 0.0)
                cells_areas_Na.setdefault(c_gid, 0.0)
                cells_volumes_K.setdefault(c_gid, 0.0)
                cells_areas_K.setdefault(c_gid, 0.0)
                cells_volumes_Ca.setdefault(c_gid, 0.0)
                #cells_volumes_ATP.setdefault(c_gid, 0.0)
                #cells_volumes_ADP.setdefault(c_gid, 0.0)

                nais.setdefault(c_gid, 0.0)
                cais.setdefault(c_gid, 0.0)
                kis.setdefault(c_gid, 0.0)
                #atpi.setdefault(c_gid, 0.0)
                #adpi.setdefault(c_gid, 0.0)

                #cells_areas.setdefault(c_gid, 0.0)
                #cells_volumes.setdefault(c_gid, 0.0)

                current_ina.setdefault(c_gid, 0.0)
                current_ik.setdefault(c_gid, 0.0)

                #secs_all = [sec for sec in nc.CCell.all if (hasattr(sec, Na.current_var) and (hasattr(sec, K.current_var)) and (hasattr(sec, ATP.atpi_var)) and (hasattr(sec, ADP.adpi_var)) and hasattr(sec, Ca.current_var)  )]
                secs_all = [sec for sec in nc.CCell.all if (hasattr(sec, Na.current_var) and (hasattr(sec, K.current_var)) and hasattr(sec, Ca.current_var)  )]

                if len(secs_all)==0:
                    print("len_secs all: ",len(secs_all))
                for sec_elem in secs_all:
                    seg_all = sec_elem.allseg()
                    for seg in seg_all:
                    #for seg in sec_elem:
                        #In order to loop through only the middle segment in the soma (as neuron does)
    #                    if ((isinstance(seg.nai, int)) or (isinstance(seg.nai, float))):
                        counter_seg_Na[c_gid] += 1.0
                        nais[c_gid] += seg.nai * 1e-3 * AVOGADRO * (seg.volume() * 1e-15) # number of molecules
                        cells_volumes_Na[c_gid] += seg.volume()
                        cells_areas_Na[c_gid] += seg.area()
                        current_ina[c_gid] += seg.ina * seg.area() / 100 # nA
                        current_ik[c_gid] += seg.ik * seg.area() / 100 # nA
                        kis[c_gid] += seg.ki * 1e-3 * AVOGADRO * (seg.volume() * 1e-15)
                        cais[c_gid] += seg.cai * 1e-3 * AVOGADRO * (seg.volume() * 1e-15)
                        #atpi[c_gid] += seg.atpi * 1e-3 * AVOGADRO * (seg.volume() * 1e-15)
                        #adpi[c_gid] += seg.adpi * 1e-3 * AVOGADRO * (seg.volume() * 1e-15)
                if counter_seg_Na[c_gid] == 0.0:
                    print("counter_seg_Na nai 0")
                    with open(test_counter_seg_file, "a") as param_outputfile:
                        param_outputfile.write(c_gid)
                        param_outputfile.write("\n")

                    cells_volumes_Na.pop(c_gid, None)
                    nais.pop(c_gid, None)
                    cells_areas_Na.pop(c_gid, None)
                    current_ina.pop(c_gid, None)
                    current_ik.pop(c_gid, None)
                    kis.pop(c_gid, None)
                    cais.pop(c_gid, None)
                    #atpi.pop(c_gid, None)
                    #adpi.pop(c_gid, None)
                    continue
                nais_mean[c_gid] = nais[c_gid] * 1e3 / ( AVOGADRO * cells_volumes_Na[c_gid] * 1e-15 ) #mM #/ counter_seg[c_gid]
                ina_density[c_gid] = current_ina[c_gid] / cells_areas_Na[c_gid] * 100 
                ik_density[c_gid] = current_ik[c_gid] / cells_areas_Na[c_gid] * 100
                kis_mean[c_gid]= kis[c_gid] * 1e3 / ( AVOGADRO * cells_volumes_Na[c_gid] * 1e-15 ) #mM  #/ counter_seg[c_gid]
                cais_mean[c_gid]= cais[c_gid] * 1e3 / ( AVOGADRO * cells_volumes_Na[c_gid] * 1e-15 ) #mM  #/ counter_seg[c_gid]
                #atpi_mean[c_gid]= atpi[c_gid] * 1e3 / ( AVOGADRO * cells_volumes_Na[c_gid] * 1e-15 ) #mM #/ counter_seg[c_gid]
                #adpi_mean[c_gid] = adpi[c_gid] * 1e3 / ( AVOGADRO * cells_volumes_Na[c_gid] * 1e-15 ) #mM #/ counter_seg[c_gid]
                #del secs_Na

    #            log_stage("nais test for nan")
    #            if (np.isnan(nais_mean[c_gid]) or math.isnan(nais_mean[c_gid]) or  (not isinstance(nais_mean[c_gid], float))):
    #                if rank == 0:
    #                    print("nai_nan_found at idxm:",idxm)
    #                    print("c_gid:",c_gid)
    #                    print("value:",nais_mean[c_gid])
    #                    print("rank:",rank)
    #                raise Exception("param_nan nais")
                
            comm.Barrier()
            #del cells_volumes_Na
            #del cells_areas_Na
            #del cells_volumes_K
            #del cells_areas_K
            #del cells_volumes_Ca
            #del cells_volumes_ATP
            #del cells_volumes_ADP

            #del counter_seg_Na
            #del counter_seg_K
            #del counter_seg_Ca
            #del counter_seg_ATP
            #del counter_seg_ADP



            error_solver = None
            failed_cells = []

            #outs_r_to_met = {}
            for c_gid, nc in gid_to_cell.items():
                if c_gid not in target_gids:
                    continue
    
                if c_gid in exc_target_gids:
                    outs_r_to_met = 4000.0 * outs_r_glu.get(c_gid,0.0) * 1e3 / ( AVOGADRO * cells_volumes[c_gid] * 1e-15 ) / SIM_END_coupling_interval  #mM/ms
                    glutamatergic_gaba_scaling = 0.1


                    if c_gid in target_gids_L1:
                        GLY_a = glycogen_scaled[0]*5.0
                        mito_scale = mito_volume_fraction_scaled[0]

                    elif c_gid in target_gids_L2:
                        GLY_a = glycogen_scaled[1]*5.0
                        mito_scale = mito_volume_fraction_scaled[1]

                    elif c_gid in target_gids_L3:
                        GLY_a = glycogen_scaled[2]*5.0
                        mito_scale = mito_volume_fraction_scaled[2]

                    elif c_gid in target_gids_L4:
                        GLY_a = glycogen_scaled[3]*5.0
                        mito_scale = mito_volume_fraction_scaled[3]

                    elif c_gid in target_gids_L5:
                        GLY_a = glycogen_scaled[4]*5.0
                        mito_scale = mito_volume_fraction_scaled[4]

                    elif c_gid in target_gids_L6:
                        GLY_a = glycogen_scaled[5]*5.0
                        mito_scale = mito_volume_fraction_scaled[5]

                elif c_gid in inh_target_gids:
                    outs_r_to_met = 4000.0 * outs_r_gaba.get(c_gid,0.0) * 1e3 / ( AVOGADRO * cells_volumes[c_gid] * 1e-15 ) / SIM_END_coupling_interval  #mM/ms

                    glutamatergic_gaba_scaling = 1.0

                    if c_gid in target_gids_L1:
                        GLY_a = glycogen_scaled[0]*5.0
                        mito_scale = mito_volume_fraction_scaled[0]

                    elif c_gid in target_gids_L2:
                        GLY_a = glycogen_scaled[1]*5.0
                        mito_scale = mito_volume_fraction_scaled[1]

                    elif c_gid in target_gids_L3:
                        GLY_a = glycogen_scaled[2]*5.0
                        mito_scale = mito_volume_fraction_scaled[2]

                    elif c_gid in target_gids_L4:
                        GLY_a = glycogen_scaled[3]*5.0
                        mito_scale = mito_volume_fraction_scaled[3]

                    elif c_gid in target_gids_L5:
                        GLY_a = glycogen_scaled[4]*5.0
                        mito_scale = mito_volume_fraction_scaled[4]

                    elif c_gid in target_gids_L6:
                        GLY_a = glycogen_scaled[5]*5.0
                        mito_scale = mito_volume_fraction_scaled[5]

                else:
                    with open(wrong_gids_testing_file, "a") as f:
                        f.write(f"{rank}\t{c_gid}\n")

    #            del outs_r_glu
    #            del outs_r_gaba


                #VNeu0 = voltages_per_gids[c_gid] #voltage_mean[c_gid] #changed7jan2021
                m0 =  0.1*(-65.0 + 30.0)/(1.0-np.exp(-0.1*(-65.0 + 30.0))) / (  0.1*(-65.0 + 30.0)/(1.0-np.exp(-0.1*(-65.0 + 30.0)))    +     4.0*np.exp(-(-65.0 + 55.0)/18.0) )  # (alpha_m + beta_m)
                
                u0 = [-65.0,m0] + u0fromFile
                #u0 = [VNeu0,m0,h0,n0,Conc_Cl_out,Conc_Cl_in, Na0in,K0out,Glc_b,Lac_b,O2_b,Q0,Glc_ecs,Lac_ecs,O2_ecs,O2_n,O2_a,Glc_n,Glc_a,Lac_n,Lac_a,Pyr_n,Pyr_a,PCr_n,PCr_a,Cr_n,Cr_a,ATP_n,ATP_a,ADP_n,ADP_a,NADH_n,NADH_a,NAD_n,NAD_a,ksi0,ksi0]

    #            if rank == 0:
    #                print("u0: ",len(u0))
    #                print("u027: ",u0[27])
                
                metabolism = gen_metabolism_model()  

                tspan_m = (1e-3*float(idxm)*SIM_END_coupling_interval,1e-3*(float(idxm)+1.0)*SIM_END_coupling_interval)  #tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0)) 
                um[(0,c_gid)] = u0

                vm=um[(idxm,c_gid)]

                #vm[161] = vm[161] - outs_r_glu.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)
                #vm[165] = vm[165] - outs_r_gaba.get(c_gid, 0.0)*4000.0/(6e23*1.5e-12)

                #comm.Barrier()

                vm[6] = nais_mean[c_gid]
                vm[7] = u0[7] - 1.33 * (kis_mean[c_gid] - 140.0 ) # Kout #changed7jan2021
    #            vm[8] = 2.255 # Glc_b
                # TO CHECK (2nd August 2022)
                #vm[27] = 0.5*1.2 + 0.5*atpi_mean[c_gid] #commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model
                #vm[29] = 0.5*6.3e-3 + 0.5*adpi_mean[c_gid] #commented on 13jan2021 because ATPase is in model, so if uncomment, the ATPase effects will be counted twice for metab model

                #param = [current_ina[c_gid], 0.06, voltage_mean[c_gid],nais_mean[c_gid],kis_mean[c_gid], current_ik[c_gid], 4.4, pAKTPFK2, atpi_mean[c_gid],vm[27],cais_mean[c_gid],mito_scale,glutamatergic_gaba_scaling] 

                #comm.Barrier()

    #            param = [current_ina[c_gid], 0.06, voltage_mean[c_gid], nais_mean[c_gid], kis_mean[c_gid], current_ik[c_gid], 4.1, pAKTPFK2, atpi_mean[c_gid],vm[27],cais_mean[c_gid],mito_scale,glutamatergic_gaba_scaling, outs_r_to_met[c_gid]] 
                #!!! 1000* in param is to have current_ina and current_ik units = uA/cm2 same as in Calvetti

                #param = [ina_density[c_gid], 0.06, -65.0, nais_mean[c_gid], kis_mean[c_gid], ik_density[c_gid], 4.1, 0.17, atpi_mean[c_gid],vm[27],cais_mean[c_gid],mito_scale,glutamatergic_gaba_scaling, outs_r_to_met]
                # TO CHECK (2nd August 2022) - compared to line above
                param = [ina_density[c_gid], 0.06, -65.0, nais_mean[c_gid], kis_mean[c_gid], ik_density[c_gid], 4.1, 0.17, vm[27],vm[27],cais_mean[c_gid],mito_scale,glutamatergic_gaba_scaling, outs_r_to_met]
                prob_metabo = de.ODEProblem(metabolism,vm,tspan_m,param)
                log_stage("solve metabolism")
                #with timer('julia'):
                #if idxm % 10 == 0:
                with open(param_out_file, "a") as param_outputfile:
                    out_data = [c_gid]
                    out_data.append(rank)
                    out_data.append(idxm)
                    out_data.extend(param)
                    #out_data.append(current_ina[c_gid])
                    #out_data.append(outs_r[idxm].get(c_gid, 0.0))
                    out_data.append(cells_volumes[c_gid])
                    #out_data.append(cells_areas[c_gid])
                    param_outputfile.write("\t".join([str(p) for p in out_data]))
                    param_outputfile.write("\n")
                    out_data = None

                if ((any([np.isnan(p) for p in param])) or  (any([math.isnan(p) for p in param])) or (not (isinstance(sum(param), float)) )):
                    print("param_nan_found at idxm: ",idxm)
                    failed_cells.append(c_gid)
                    #gid_to_cell.pop(c_gid) 

                    continue

                else:
                    
                    log_stage("solve metabolism")



            #  sol = de.solve(prob_metabo, de.Rodas5(),reltol=1e-8,abstol=1e-8,save_everystep=False )
                sol = None
                error_solver = None

                for i in range(5):
                    if i ==5:
                        print("metab solver attempt 10")
                    try:
                        sol = de.solve(prob_metabo, de.Rodas4P() ,reltol=1e-6,abstol=1e-6,maxiters=1e4,save_everystep=False) #de.Rodas4P ,autodiff=False (deprecated ?)
                        #sol = de.solve(prob_metabo, de.Rosenbrock23(),autodiff=False ,reltol=1e-8,abstol=1e-8,maxiters=1e4,save_everystep=False) #de.Rodas4P

                        #sol = de.solve(prob_metabo, de.Tsit5(),reltol=1e-4,abstol=1e-4,maxiters=1e4,save_everystep=False)
                        #sol = de.solve(prob_metabo, de.AutoTsit5(de.Rosenbrock23()),reltol=1e-4,abstol=1e-6,maxiters=1e4,save_everystep=False)
                        #sol = de.solve(prob_metabo, de.Tsit5(),reltol=1e-6,abstol=1e-6,maxiters=1e4,save_everystep=False)

                        #with open(f"/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/solver_good_{timestr}.txt", "a") as f:
                        #    f.write(f"{rank}\t{c_gid}\n")    

                        if sol.retcode != "Success":
                            print(f"sol.retcode: {sol.retcode}")
                        #else:
                        #    print(f"success sol.retcode: {sol.retcode}")


                        break
                    except Exception as e:
                        with open(err_solver_output, "a") as f:
                            f.write(f"{rank}\t{c_gid}\n")
                        error_solver = e
                        failed_cells.append(c_gid)
                if sol is None:
                    raise error_solver

                um[(idxm+1,c_gid)] = sol.u[-1]


                #um[(idxm+1,c_gid)] = sol.u[-1]
    #            logging.info("um_to_output")
    #            if idxm % 10 == 0:
                with open(um_out_file, "a") as test_outputfile:
                    um_out_data = [c_gid]
                    um_out_data.append(rank)
                    um_out_data.append(idxm)
                    um_out_data.extend(sol.u[-1])
                    test_outputfile.write("\t".join([str(p) for p in um_out_data]))
                    test_outputfile.write("\n")
                    um_out_data = None

                sol = None

                #atpi_weighted_mean = 0.5*1.2 + 0.5*um[(idxm+1,c_gid)][27] #um[(idxm+1,c_gid)][27]
                #adpi_weighted_mean = 0.5*6.3e-3 + 0.5*um[(idxm+1,c_gid)][29]  #um[(idxm+1,c_gid)][29]

                nao_weighted_mean = 0.5*140.0 + 0.5*(140.0 - 1.33*(um[(idxm+1,c_gid)][6] - 10.0)) #140.0 - 1.33*(param[3] - 10.0) #14jan2021  # or 140.0 - .. # 144  # param[3] because pyhton indexing is 0,1,2.. julia is 1,2,..
                ko_weighted_mean = 0.5*5.0 + 0.5*um[(idxm+1,c_gid)][7] #um[(idxm+1,c_gid)][7] 
                nai_weighted_mean = 0.5*10.0 + 0.5*um[(idxm+1,c_gid)][6] #0.5*10.0 + 0.5*um[(idxm+1,c_gid)][6] #um[(idxm+1,c_gid)][6]
                ki_weighted_mean = 0.5*140.0 + 0.5*param[4] #14jan2021
                #feedback loop to constrain ndamus by metabolism output

    #            print("size_of_um: ",getsizeof(um)," bytes ","idxm: ",idxm,"rank: ",rank) # accumulates with idxm, but shouldn't
                um[(idxm,c_gid)] = None
                
                #del vm
                #del param



                log_stage("feedback")
                #secs_all = [sec for sec in nc.CCell.all if (hasattr(sec, Na.current_var) and (hasattr(sec, K.current_var)) and (hasattr(sec, ATP.atpi_var)) and (hasattr(sec, ADP.adpi_var)) and hasattr(sec, Ca.current_var)  )]
                secs_all = [sec for sec in nc.CCell.all if (hasattr(sec, Na.current_var) and (hasattr(sec, K.current_var)) and hasattr(sec, Ca.current_var)  )]

                for sec_elem in secs_all:
                    seg_all = sec_elem.allseg()
                    for seg in seg_all:
                    #for sec_elem in secs_Na:
                    #    for seg in sec_elem:
                        seg.nao = nao_weighted_mean #140
                        seg.nai = nai_weighted_mean #10
                        seg.ko = ko_weighted_mean #5
                        seg.ki = ki_weighted_mean #140
                        #seg.atpi = atpi_weighted_mean #1.4
                        #seg.adpi = adpi_weighted_mean #0.03
                #        seg.v = -65.0

    #            secs_v = [sec for sec in nc.CCell.all if (hasattr(sec, "v") )]
    #            for sec_elem in secs_v:
    #                seg_all = sec_elem.allseg()
    #                for seg in seg_all:
    #                #for sec_elem in secs_v:
    #                #for seg in sec_elem:
    #                    seg.v = -65.0
    #            #del secs_v

            #del secs_all
            comm.Barrier()

            #print("size_of_um: ",getsizeof(um)," bytes ","idxm: ",idxm,"rank: ",rank) # accumulates with idxm, but shouldn't
    #        process = psutil.Process(os.getpid())
    #        print("memory_info_rss: ",process.memory_info().rss / 1073741824 ," Gbytes ","idxm: ",idxm,"rank: ",rank) #in bytes

    #        logging.info("pop_failed_cells")
            #if error_solver is not None:
            #    raise error_solver
            for i in failed_cells:
                print("failed_cells:",i,"at idxm: ",idxm)
                gid_to_cell.pop(i)

            idxm += 1
        
        rss.append(psutil.Process().memory_info().rss / 1024**2) # memory consumption in MB
    
    if rank == 0 and REPORT_FLAG:
        f.close()

    ndamus.spike2file("out.dat")

    mt.timer.print()

    #logging.info(textwrap.dedent("""\
    #    Simulation finished. Timings:
	#   - Neuron: {neuron_cum:g}
	#   - curr2conc: {curr2conc:g}
	#   - setBatchTetConcs: {setBatchTetConcs:g}
	#   - Julia: {julia:g}""".format_map(_timings)))

    rss = np.mean(rss)
    avg_rss = comm.allreduce(rss, MPI.SUM) / smpi.nhosts
    if rank == 0:
        print(f"average (across ranks) memory consumption [MB]: {avg_rss}")

if __name__ == "__main__":
    main()
    exit() # needed to avoid hanging

