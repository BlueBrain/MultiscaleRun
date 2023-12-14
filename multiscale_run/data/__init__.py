"""
This module provides an API on top of the data files shipped
with this Python package available in this directory.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


DATA_DIR = Path(__file__).parent.resolve()

CONFIG_DIR = DATA_DIR / "config"
CIRCUITS_DIRS = [
    CONFIG_DIR / "rat_sscxS1HL_V6",
    CONFIG_DIR / "rat_sscxS1HL_V10_all_valid_cells",
]
DEFAULT_CIRCUIT = CONFIG_DIR / "rat_sscxS1HL_V6"

MSR_CONFIG_JSON = CONFIG_DIR / "msr_config.json"
MSR_POSTPROC = DATA_DIR / "postproc.ipynb"
METABOLISM_MODEL = DATA_DIR / "metabolismndam_reduced"

_jinja_env = Environment(loader=FileSystemLoader(DATA_DIR))

SBATCH_TEMPLATE = _jinja_env.get_template("simulation.sbatch.jinja")

BB5_JULIA_ENV = Path(
    "/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment/latest"
)


def circuit_path(name):
    path = CONFIG_DIR / name
    if not path.is_dir():
        raise IOError(f"Unknown circuit '{name}'")
    return path
