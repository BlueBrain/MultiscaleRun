"""
This module provides the command line interface of the multiscale-run
console-scripts entrypoint of the Python package (see setup.py)

The CLI provides the required commands to create a run simulations.
"""

import argparse
import contextlib
import copy
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import diffeqpy

from .simulation import MsrSimulation
from . import __version__
from .data import (
    BB5_JULIA_ENV,
    CIRCUITS_DIRS,
    DEFAULT_CIRCUIT,
    circuit_path,
    SBATCH_TEMPLATE,
    MSR_CONFIG_JSON,
    MSR_POSTPROC,
)


def _cli_logger():
    logger = logging.getLogger("cli")
    logger.setLevel(logging.WARN)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("࿋ %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


LOGGER = _cli_logger()
del _cli_logger


def replace_in_file(file, old, new, count=-1):
    """
    In-place string replacement in a file

    Params:
      file: the file path to replace
      old: the string to be replaced
      new: the replacement string
      count: maximum number of occurrences to replace within the file.
            -1 (the default value) means replace all occurrences.
    """
    with open(file) as istr:
        content = istr.read()
    new_content = content.replace(old, new, count)
    if new_content != content:
        with open(file, "w") as ostr:
            ostr.write(new_content)


def julia_env(func):
    """Decorator to define the required Julia environment variables"""

    def wrap(**kwargs):
        julia_depot = Path(".julia")
        julia_project = Path("julia_environment")
        os.environ.setdefault("JULIA_DEPOT_PATH", str(julia_depot.resolve()))
        os.environ.setdefault("JULIA_PROJECT", str(julia_project.resolve()))
        return func(**kwargs)

    return wrap


def command(func):
    """Decorator for every command functions"""

    def wrap(directory=None, **kwargs):
        directory = Path(directory or ".")
        directory.mkdir(parents=True, exist_ok=True)
        with pushd(directory):
            return func(directory=directory, **kwargs)

    return wrap


def julia_cmd(*instructions):
    """Execute a set of Julia instructions in a dedicated Julia process"""
    julia_cmds = ["using Pkg"]
    for instruction in instructions:
        julia_cmds.append(instruction)
    cmd = ["julia", "-e", ";".join(julia_cmds)]
    subprocess.check_call(cmd)


def julia_pkg(command, package):
    """Execute a Julia Pkg command in a dedicated process

    Args:
      command: the Pkg function
      package: the package to install, either as a string or a dictionary
    """
    instruction = f'Pkg.{command}("{package}")'
    julia_cmd(instruction)


@contextlib.contextmanager
def pushd(path):
    """Change the current working directory within the scope of a Python `with` statement

    Args:
      path: the directory to jump into
    """
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(cwd)


@julia_env
@command
def julia(args=None, **kwargs):
    command = ["julia"]
    if args is not None:
        command += args
    subprocess.check_call(command)


@command
def init(directory, circuit, julia="shared", force=False):
    if not force and next(Path(".").iterdir(), None) is not None:
        raise IOError(
            f"Directory '{directory}' is not empty. "
            "Use option '-f' to overwrite the content of the directory."
        )
    assert julia in ["shared", "create", "no"]

    circuit = circuit_path(circuit)

    SBATCH_CIRCUITS_PARAMS = dict(
        rat_sscxS1HL_V6=dict(
            job_name="msr_ratV6",
            nodes=1,
            time="01:00:00",
        ),
        rat_sscxS1HL_V10_all_valid_cells=dict(
            job_name="msr_ratV10",
            nodes=64,
            time="10:00:00",
        ),
    )
    sbatch_params = copy.copy(SBATCH_CIRCUITS_PARAMS[circuit.name])
    loaded_modules = filter(
        lambda s: len(s) > 0, os.environ.get("LOADEDMODULES", "").split(":")
    )
    sbatch_params["loaded_modules"] = loaded_modules
    SBATCH_TEMPLATE.stream(sbatch_params).dump("simulation.sbatch")
    shutil.copy(MSR_CONFIG_JSON, MSR_CONFIG_JSON.name)
    shutil.copy(MSR_POSTPROC, MSR_POSTPROC.name)

    shutil.copytree(
        str(circuit),
        "config",
        ignore=shutil.ignore_patterns("cache"),
        dirs_exist_ok=True,
    )
    if julia == "no":
        replace_in_file(
            MSR_CONFIG_JSON.name,
            '"with_metabolism": true,',
            '"with_metabolism": false,',
        )
    else:
        if julia == "shared" and not BB5_JULIA_ENV.exists():
            LOGGER.warning("Cannot find shared Julia environment at %s", BB5_JULIA_ENV)
            LOGGER.warning("Creating a new one")
            julia = "create"
        if julia == "shared":
            LOGGER.warning("Reusing shared Julia environment at %s", BB5_JULIA_ENV)
            os.symlink(BB5_JULIA_ENV / ".julia", ".julia")
            os.symlink(BB5_JULIA_ENV / "julia_environment", "julia_environment")
        elif julia == "create":
            Path(".julia").mkdir(exist_ok=True)
            Path("julia_environment").mkdir(exist_ok=True)
            LOGGER.warning("Installing Julia package 'IJulia'")
            julia_pkg("add", "IJulia")
            LOGGER.warning(
                "Installing Julia packages required to solve differential equations"
            )
            diffeqpy.install()
            LOGGER.warning("Precompiling all Julia packages")
            julia_cmd("Pkg.instantiate(; verbose=true)")

            replace_in_file(
                MSR_CONFIG_JSON.name,
                '"with_metabolism": false,',
                '"with_metabolism": true,',
            )

        @julia_env
        def check_julia_env():
            LOGGER.warning("Checking installation of differential equations solver...")
            # noinspection PyUnresolvedReferences
            from diffeqpy import de  # noqa : F401

        check_julia_env()

    LOGGER.warning(
        "Preparation of the simulation configuration and environment succeeded"
    )
    LOGGER.warning(
        "The generated setup is already ready to compute "
        "with the command 'multiscale-run compute' or via the "
        "generated sbatch file. But feel free to browse and tweak "
        "the JSON configuration files at will!"
    )


@julia_env
@command
def compute(**kwargs):
    sim = MsrSimulation(Path("config"))
    sim.main()


@julia_env
@command
def check(**kwargs):
    LOGGER.warning("Running minimalist checks to test environment sanity")
    sim = MsrSimulation(Path("config"))
    sim.configure()
    LOGGER.warning("Checking installation of differential equations solver...")
    # noinspection PyUnresolvedReferences
    from diffeqpy import de  # noqa : F401

    LOGGER.warning("The simulation environment looks sane")


def main(**kwargs):
    """Package script entry-point for the multiscale-run CLI utility.

    Args:
        kwargs: optional arguments passed to the argument parser.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--verbose", "-v", action="count", default=0)
    subparsers = ap.add_subparsers(title="commands")

    parser_init = subparsers.add_parser("init", help="Setup a new simulation")
    parser_init.set_defaults(func=init)
    parser_init.add_argument(
        "--circuit",
        choices=[p.name for p in CIRCUITS_DIRS],
        default=DEFAULT_CIRCUIT.name,
    )
    parser_init.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force files creations if directory already exists",
    )
    parser_init.add_argument(
        "--julia",
        choices=["shared", "create", "no"],
        default="shared",
        help="Choose Julia installation. "
        "'shared' (the default) to reuse an existing env on BB5, "
        "'create' to construct a new env locally, "
        "'no' to skip Julia environment (typically if metabolism model is not required)",
    )
    parser_init.add_argument("directory", nargs="?")

    parser_check = subparsers.add_parser("check", help="Check environment sanity")
    parser_check.set_defaults(func=check)

    parser_compute = subparsers.add_parser("compute", help="Compute the simulation")
    parser_compute.set_defaults(func=compute)

    parser_julia = subparsers.add_parser(
        "julia", help="Run Julia within the simulation environment"
    )
    parser_julia.set_defaults(func=julia)
    parser_julia.add_argument(
        "args",
        nargs="*",
        metavar="ARGS",
        help="Optional arguments passed to Julia executable",
    )

    args = ap.parse_args(**kwargs)
    args = vars(args)

    verbosity = args.pop("verbose")
    log_level = logging.WARN
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity > 1:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    if callback := args.pop("func", None):
        try:
            callback(**args)
        except Exception as e:
            logging.error(e)
            sys.exit(1)
    else:
        ap.error("a subcommand is required.")
