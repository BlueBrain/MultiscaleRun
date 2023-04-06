import os
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


class ConfigError(Exception):
    pass


def load_from_env(env_name, default):
    """Either load from environment or set with default (env takes priority)"""
    var = os.getenv(env_name)

    if isinstance(default, bool):
        return bool(int(var)) if var is not None else default
    elif isinstance(default, int):
        return int(var) if var is not None else default
    else:
        return var if var is not None else default


##############################################
# base flags and paths
##############################################

with_steps = load_from_env("with_steps", True)
with_metabolism = load_from_env("with_metabolism", True)
with_bloodflow = load_from_env("with_bloodflow", True)

results_path = load_from_env("results_path", "RESULTS/STEPS4")

sonata_path = load_from_env("sonata_path", "simulation_config.json")

steps_version = load_from_env("steps_version", 4)
steps_mesh_path = load_from_env("steps_mesh_path", "steps_meshes/mc2c/mc2c.msh")

##############################################
# Dualrun related
##############################################


if steps_version not in [3, 4]:
    raise ConfigError(f"Steps number: {steps_version} is not 3 or 4")

# for out file names
timestr = time.strftime("%Y%m%d%H")
np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = spc.physical_constants["elementary charge"][0]
AVOGADRO = spc.N_A
COULOMB = spc.physical_constants["joule-electron volt relationship"][0]

CONC_FACTOR = 1e-9
OUTS_R_TO_MET_FACTOR = 4000.0 * 1e3 / (AVOGADRO * 1e-15)

dt_nrn2dt_steps: int = (
    100  # 100 steps-ndam coupling. NOT SECONDS NOT MS, IT'S NUMBER OF NDAM DT
)
dt_nrn2dt_jl: int = 4000  # 40000  # metabolism (julia)-ndam coupling. NOT SECONDS NOT MS, IT'S NUMBER OF NDAM DT
dt_nrn2dt_bf: int = dt_nrn2dt_jl  # TODO decide when to sync with bloodflow


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
    conc_0 = 0  # 3 it was 3 in Dan's example  # (mM/L)
    base_conc = 0  # 3 it was 3 in Dan's example #sum here is 6 which is probably too high according to Magistretti #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = "diff_KK"
    diffcst = 2e-9
    current_var = "ik"
    ki_var = "ki"
    charge = 1 * ELEM_CHARGE


class ATP:
    name = "ATP"
    conc_0 = 0.1
    base_conc = 1.4  # base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = "diff_ATP"
    diffcst = 2e-9
    charge = -3 * ELEM_CHARGE
    atpi_var = "atpi"


class ADP:
    name = "ADP"
    conc_0 = 0.0001
    base_conc = 0.03  # base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = "diff_ADP"
    diffcst = 2e-9
    charge = -2 * ELEM_CHARGE
    adpi_var = "adpi"


class Ca:
    name = "Ca"
    conc_0 = 1e-5
    base_conc = 4e-5
    diffname = "diff_Ca"
    diffcst = 2e-9
    current_var = "ica"
    charge = 2 * ELEM_CHARGE
    e_var = "eca"
    cai_var = "cai"


class Volsys:
    name = "extra_volsys"
    specs = (Na, KK)


specNames = [Na.name, KK.name]


##############################################
# Triplerun related
##############################################


# paths
metabolism_path = "metabolismndam_reduced"
path_to_metab_jl = os.path.join(metabolism_path, "sim/metabolism_unit_models")

# files
julia_code_file_name = "metabolism_model_21nov22_withEphysCurrNdam_noSB.jl" #"metabolism_model_21nov22_noEphys_noSB.jl","metabolism_model_21nov22_withEphysNoCurrNdam_noSB.jl",
julia_code_file = os.path.join(
    path_to_metab_jl, julia_code_file_name
)  # file created based on /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/optimiz_unit/enzymes/enzymes_preBigg/COMBO/MODEL_17Nov22.ipynb
u0_file = os.path.join(
    path_to_metab_jl, "u0steady_22nov22.csv"
)  # file created from /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/optimiz_unit/enzymes/enzymes_preBigg/COMBO/MODEL_17Nov22.ipynb


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

moles_current_output = "Moles_Current.csv"
comp_counts_output = "CompCount.csv"

#####
gids_lists_dir = "metabolismndam_reduced/sim/gids_sets"
# mrci stems from testNGVSSCX. However, the layer subdivision is totally random and there is no
# correlation with real-life data
target_gids = set(list(
    np.loadtxt(os.path.join(gids_lists_dir, "mrci_gids.txt"))
))

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


# paths
bloodflow_path = load_from_env("BLOODFLOW_PATH", "bloodflow_src")
bloodflow_params_path = os.path.join(os.getcwd(), "bloodflow_params.yaml")


def print_config():
    if MPI_RANK == 0:
        print(
            f"""
    -----------------------------------------------------
    --- MSR CONFIG ---
    You can override any of the following variables by setting the omonim environment variable
    current folder: {os.getcwd()}
    with_steps: {with_steps}
    with_metabolism: {with_metabolism}
    with_bloodflow: {with_bloodflow}

    results_path: {results_path}

    sonata_path: {sonata_path}

    steps_version: {steps_version}
    steps_mesh_path: {steps_mesh_path}

    BLOODFLOW_PATH: {bloodflow_path}
    --- MSR CONFIG ---
    -----------------------------------------------------
    """,
            flush=True,
        )


##############################################
# Conversion factors
##############################################


micrometer2meter = 1e-6
