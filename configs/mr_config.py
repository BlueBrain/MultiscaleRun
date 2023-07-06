import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from multiscale_run import utils

import time

import numpy as np
from scipy import constants as spc

from mpi4py import MPI as MPI4PY

MPI_COMM = MPI4PY.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

##############################################
# Dualrun related
##############################################


import os
import time

import numpy as np
from scipy import constants as spc

import logging


class ConfigError(Exception):
    pass


##############################################
# base flags and paths
##############################################

mr_debug = utils.load_from_env("mr_debug", False)
logging.basicConfig(level=logging.DEBUG if mr_debug else logging.INFO)

with_steps = utils.load_from_env("with_steps", True)
with_metabolism = utils.load_from_env("with_metabolism", True)
with_bloodflow = utils.load_from_env("with_bloodflow", True)

results_path = utils.load_from_env("results_path", "RESULTS")

# If you want to change paths, use bash: "export sonata_path=mypath"
# and put mesh and mr config file in the same folder. Otherwise the program
# looks in the parent folder. If not found it fails.

# Do not change this directly, pls
sonata_path = utils.get_sonata_path()

# Do not change this directly, pls
config_path = utils.get_config_path()


# You can change this considering that we are searcing in the folder of the 
# sonata_path first and its parent if not found

# in case of unsplit mesh it auto-splits
steps_mesh_path = utils.search_path("mesh/mc2c/mc2c.msh")

##############################################
# caching
##############################################

cache_path = os.path.join(os.path.split(sonata_path)[0], "cache")
cache_save = True
cache_load = True

##############################################
# Dualrun related
##############################################

# for out file names
timestr = time.strftime("%Y%m%d%H")
np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = spc.physical_constants["elementary charge"][0]
AVOGADRO = spc.N_A

# prevent steps overflowing
CONC_FACTOR = 1e-9
OUTS_R_TO_MET_FACTOR = 4000.0 * 1e3 / (AVOGADRO * 1e-15)

# these dts are multiples of ndam dt. To get the full dt you need to multiply by DT
n_DT_steps_per_update = {}
if with_steps:
    n_DT_steps_per_update["steps"] = 100

if with_metabolism:
    n_DT_steps_per_update["metab"] = 10 if mr_debug else 4000

if with_bloodflow:
    n_DT_steps_per_update["bf"] = 10 if mr_debug else 4000

# the multiscale run update happens with the largest DT that syncs everything else
n_DT_steps_per_update["mr"] = np.gcd.reduce(list(n_DT_steps_per_update.values()))


class Mesh:
    compname = "extra"


class Na:
    name = "Na"
    conc_0 = 140  # (mM/L)
    diffname = "diff_Na"
    diffcst = 2e-9
    current_var = "ina"
    charge = 1 * ELEM_CHARGE
    e_var = "ena"
    nai_var = "nai"


class KK:
    """Potassium specs. It is not just K because it is reserved in steps"""

    name = "KK"
    conc_0 = 3  # it was 3 in Dan's example  # (mM/L)
    diffname = "diff_KK"
    diffcst = 2e-9
    current_var = "ik"
    ki_var = "ki"
    ko_var = "ko"
    charge = 1 * ELEM_CHARGE


class ATP:
    name = "ATP"
    conc_0 = 0.1
    diffname = "diff_ATP"
    diffcst = 2e-9
    charge = -3 * ELEM_CHARGE
    atpi_var = "atpi"


class ADP:
    name = "ADP"
    conc_0 = 0.0001
    diffname = "diff_ADP"
    diffcst = 2e-9
    charge = -2 * ELEM_CHARGE
    adpi_var = "adpi"


class Ca:
    name = "Ca"
    conc_0 = 1e-5
    diffname = "diff_Ca"
    diffcst = 2e-9
    current_var = "ica"
    charge = 2 * ELEM_CHARGE
    e_var = "eca"
    cai_var = "cai"


class Volsys:
    name = "extra_volsys"
    specs = [Na, KK]


##############################################
# Triplerun related
##############################################
metabolism_type = ["cns", "main"][0]  # CHOOSE HERE FOR METABOLISM VERSION

# paths
metabolism_path = "metabolismndam_reduced"
path_to_metab_jl = os.path.join(metabolism_path, "sim/metabolism_unit_models")

# files
match metabolism_type:
    case "cns":
        julia_code_file_name = "met4cns.jl"  # < reduced for CNS
        u0_file = os.path.join(path_to_metab_jl, "u0_Calv_ATP_1p4_Nai10.csv")
        metab_vm_indexes = {"atpn": 28, "adpn": 30, "nai": 7, "ko": 8}
    case "main":
        julia_code_file_name = "metabolismWithSBBFinput_ndamAdapted_opt_sys_young_202302210826_2stim.jl"  # <main met
        u0_file = os.path.join(
            path_to_metab_jl, "u0steady_22nov22.csv"
        )  # file created from /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/optimiz_unit/enzymes/enzymes_preBigg/COMBO/MODEL_17Nov22.ipynb
        metab_vm_indexes = {"atpn": 22, "adpn": 23, "nai": 98, "ko": 95}


julia_code_file = os.path.join(
    path_to_metab_jl, julia_code_file_name
)  # file created based on /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/optimiz_unit/enzymes/enzymes_preBigg/COMBO/MODEL_17Nov22.ipynb


ins_glut_file_output = f"dis_ins_r_glut_{timestr}.csv"
ins_gaba_file_output = f"dis_ins_r_gaba_{timestr}.csv"
outs_glut_file_output = f"dis_outs_r_glut_{timestr}.csv"
outs_gaba_file_output = f"dis_outs_r_gaba_{timestr}.csv"

param_out_file = f"dis_param_{timestr}.txt"
um_out_file = f"dis_um_{timestr}.txt"


#####
test_counter_seg_file = f"dis_test_counter_seg0_{timestr}.txt"
wrong_gids_testing_file = f"dis_wrong_gid_errors_{timestr}.txt"
err_solver_output = f"dis_solver_errors_{timestr}.txt"

#####
gids_lists_dir = "metabolismndam_reduced/sim/gids_sets"
# mrci stems from testNGVSSCX. However, the layer subdivision is totally random and there is no
# correlation with real-life data
target_gids = set(list(np.loadtxt(os.path.join(gids_lists_dir, "mrci_gids.txt"))))

target_gids_L = []
for i in range(1, 7):
    list_temp = np.loadtxt(os.path.join(gids_lists_dir, f"mrci_L{i}_gids.txt")).tolist()
    target_gids_L.append(list_temp if isinstance(list_temp, list) else [list_temp])


mito_volume_fraction = [
    0.0459,
    0.0522,
    0.064,
    0.0774,
    0.0575,
    0.0403,
]  # 6 Layers of the circuit
mito_volume_fraction_scaled = [
    i / max(mito_volume_fraction) for i in mito_volume_fraction
]

glycogen_au = [
    128.0,
    100.0,
    100.0,
    90.0,
    80.0,
    75.0,
]  # 6 Layers of the circuit
glycogen_scaled = [i / max(glycogen_au) for i in glycogen_au]


def get_GLY_a_and_mito_vol_frac(c_gid):
    for idx, tgt in enumerate(target_gids_L):
        if c_gid in tgt:
            return glycogen_scaled[idx] * 14, mito_volume_fraction_scaled[idx]

    return None, None


##############################################
# Quadrun related
##############################################
bloodflow_path = utils.load_from_env("BLOODFLOW_PATH", "bloodflow_src")
bloodflow_params = {
    # paths
    "output_folder": "RESULTS/bloodflow",
    "n_workers": 1,
    "compliance": 4.05,  # total arterial compliance in µl/(g.µm^-1.s^-2)^-1
    # vascular_resistance: 5.60e-8 # total systemic vascular resistance in g.(µm.s.µl)^-1
    "blood_density": 1.024e-12,  # plasma density in g.µm^-3
    "blood_viscosity": 1.2e-6,  # plasma viscosity in g.µm^-1.s^-1
    "depth_ratio": 0.05,  # 0.05 Portion of the vasculature where there are inputs, corresponds to the y-axis.
    "max_nb_inputs": 3,  # maximum number of inputs to inject flow/pressure into vasculature. Should be > 1.
    "min_subgraph_size": 100,  # number of connected nodes to filter sub-graph for input with enough nodes
    "max_capillaries_diameter": 7.0,  # 7., 4.9 mean, 2 for example and 3.5 for the biggest vasculature (µm)
    "edge_scale": 2.0,
    "node_scale": 20.0,
    "p_min": 1.0e-10,
    "input_v": 3.5e4,  # input velocity. The input flow depends on the area
    "vasc_axis": 1,  # vasculature axis corresponding to x, y, or z. Should be set to 0, 1, or 2.
}


def print_config():
    if MPI_RANK == 0:
        print(
            f"""
    -----------------------------------------------------
    --- MSR CONFIG ---
    You can override any of the following variables by setting the omonim environment variable
    mr_debug: {mr_debug}
    
    current folder: {os.getcwd()}
    with_steps: {with_steps}
    with_metabolism: {with_metabolism} 
    with_bloodflow: {with_bloodflow}

    results_path: {results_path}

    config_path: {config_path}
    sonata_path: {sonata_path}
    steps_mesh_path: {steps_mesh_path}

    BLOODFLOW_PATH: {bloodflow_path}
    --- MSR CONFIG ---
    -----------------------------------------------------
    """,
            flush=True,
        )
