"""This module provides an API on top of the data files shipped
with this Python package available in this directory.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

DATA_DIR = Path(__file__).parent.resolve()

CONFIG_DIR = DATA_DIR / "config"
MSR_CONFIG_JSON = "simulation_config.json"
MSR_SCHEMA_JSON = CONFIG_DIR / "msr.schema.json"
MSR_POSTPROC = DATA_DIR / "postproc.ipynb"
METABOLISM_MODEL = DATA_DIR / "metabolismndam_reduced"

_jinja_env = Environment(loader=FileSystemLoader(DATA_DIR))

SBATCH_TEMPLATE = _jinja_env.get_template("simulation.sbatch.jinja")

BB5_JULIA_ENV = Path(
    "/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment/latest"
)
