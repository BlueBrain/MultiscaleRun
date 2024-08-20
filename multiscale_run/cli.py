"""This module provides the command line interface of the multiscale-run
console-scripts entrypoint of the Python package (see setup.py)

The CLI provides the required commands to create a run simulations.
"""

import argparse
import copy
import functools
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import diffeqpy
import scipy.sparse
from nbconvert.nbconvertapp import main as NbConvertApp

from . import __version__
from .config import (
    DEFAULT_CIRCUIT,
    NAMED_CIRCUITS,
)
from .data import (
    BB5_JULIA_ENV,
    MSR_CONFIG_JSON,
    MSR_POSTPROC,
    SBATCH_TEMPLATE,
)
from .simulation import MsrSimulation
from .utils import (
    MsrException,
    merge_dicts,
    pushd,
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


def julia_env(func):
    """Decorator to define the required Julia environment variables"""

    @functools.wraps(func)
    def wrap(**kwargs):
        julia_depot = Path(".julia")
        julia_project = Path(".julia_environment")
        os.environ.setdefault("JULIA_DEPOT_PATH", str(julia_depot.resolve()))
        os.environ.setdefault("JULIA_PROJECT", str(julia_project.resolve()))
        return func(**kwargs)

    return wrap


def command(func):
    """Decorator for every command functions"""
    # extract first line of function docstring, and use it
    # for --help description
    func_doc = func.__doc__ or ""
    func.__argparse_help__ = func_doc.split("\n", 1)[0]

    @functools.wraps(func)
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

    def print_os_var(var):
        LOGGER.warning(f"{var}: {os.environ.get(var, 'NOT_FOUND')}")

    print_os_var("JULIA_PROJECT")
    print_os_var("JULIA_DEPOT_PATH")

    LOGGER.warning(f"Running the command '{cmd}' in {Path.cwd()}")
    subprocess.check_call(cmd)


def julia_pkg(command, package):
    """Execute a Julia Pkg command in a dedicated process

    Args:
      command: the Pkg function
      package: the package to install, either as a string or a dictionary
    """
    instruction = f'Pkg.{command}("{package}")'
    julia_cmd(instruction)


@command
@julia_env
def julia(args=None, **kwargs):
    """Run Julia within the simulation environment"""
    command = ["julia"]
    if args is not None:
        command += args

    subprocess.check_call(command)


@command
def init(directory, circuit, julia="shared", check=True, force=False):
    """Setup a new simulation"""
    if not force:
        dir_content = set(Path(".").iterdir())
        dir_content.discard(Path(".logs"))
        if dir_content:
            raise MsrException(
                f"Directory '{directory}' is not empty. "
                "Use option '-f' to overwrite the content of the directory."
            )
    assert julia in ["shared", "create", "no"]

    circuit_def = NAMED_CIRCUITS[circuit]

    sbatch_parameters = copy.copy(circuit_def.sbatch_parameters)
    loaded_modules = filter(
        lambda s: len(s) > 0, os.environ.get("LOADEDMODULES", "").split(":")
    )
    sbatch_parameters["loaded_modules"] = loaded_modules
    SBATCH_TEMPLATE.stream(sbatch_parameters).dump("simulation.sbatch")
    shutil.copy(MSR_POSTPROC, MSR_POSTPROC.name)
    shutil.copytree(
        str(circuit_def.path),
        ".",
        ignore=shutil.ignore_patterns("cache", "msr*"),
        dirs_exist_ok=True,
    )

    rd = {"msr_version": str(__version__)}
    if julia == "no":
        rd["with_metabolism"] = False
    else:
        if julia == "shared" and not BB5_JULIA_ENV.exists():
            LOGGER.warning("Cannot find shared Julia environment at %s", BB5_JULIA_ENV)
            LOGGER.warning("Creating a new one")
            julia = "create"
        if julia == "shared":
            LOGGER.warning("Reusing shared Julia environment at %s", BB5_JULIA_ENV)
            for dir in [".julia", ".julia_environment"]:
                dir = Path(dir)
                if dir.is_symlink():
                    dir.unlink()
                elif dir.is_dir():
                    shutil.rmtree(dir)
                os.symlink(BB5_JULIA_ENV / dir.name[1:], dir)
        elif julia == "create":
            Path(".julia").mkdir(exist_ok=True)
            Path(".julia_environment").mkdir(exist_ok=True)

            @julia_env
            def create():
                LOGGER.warning("Installing Julia package 'IJulia'")
                julia_pkg("add", "IJulia")
                LOGGER.warning(
                    "Installing Julia packages required to solve differential equations"
                )
                diffeqpy.install()
                LOGGER.warning("Precompiling all Julia packages")
                julia_cmd("Pkg.instantiate(; verbose=true)")

                rd["with_metabolism"] = True

            create()

        @julia_env
        def check_julia_env():
            LOGGER.warning("Checking installation of differential equations solver...")
            # noinspection PyUnresolvedReferences
            from diffeqpy import de  # noqa : F401

        if check:
            check_julia_env()

    circuit_config = circuit_def.json()
    circuit_config = merge_dicts(child={"multiscale_run": rd}, parent=circuit_config)
    with open(MSR_CONFIG_JSON, "w") as ostr:
        json.dump(circuit_config, ostr, indent=4)

    LOGGER.warning(
        "Preparation of the simulation configuration and environment succeeded"
    )
    LOGGER.warning(
        "The generated setup is already ready to compute "
        "with the command 'multiscale-run compute' or via the "
        "generated sbatch file. But feel free to browse and tweak "
        "the JSON configuration files at will!"
    )
    return directory


@command
@julia_env
def compute(**kwargs):
    """Compute the simulation"""
    _check_local_neuron_mechanisms()
    sim = MsrSimulation()
    sim.main()


@command
@julia_env
def check(**kwargs):
    """Check environment sanity"""
    sane = True
    LOGGER.warning("Running minimalist checks to test environment sanity")
    sim = MsrSimulation()
    sim.configure()

    sane &= _check_local_neuron_mechanisms()

    if sim.conf.multiscale_run.with_metabolism:
        LOGGER.warning("Checking installation of differential equations solver...")
        # noinspection PyUnresolvedReferences
        from diffeqpy import de  # noqa : F401

    if sane:
        LOGGER.warning("The simulation environment looks sane")
    else:
        LOGGER.warning("Inconsistencies have been spotted")
        sys.exit(1)


@command
def post_processing(notebook: str, **kwargs) -> Path:
    """Execute a Jupyter notebook over the simulation results to generate an HTML document

    Returns
      Path to the created HTML report
    """
    from .config import MsrConfig

    conf = MsrConfig()
    results = str(conf.output.output_dir)
    NbConvertApp(
        ["--execute", "--to", "html", "--no-input", f"--output-dir={results}", notebook]
    )
    return Path(results) / (Path(notebook).stem + ".html")


def spack(*args, log=True):
    spack_executable = Path(os.environ["SPACK_ROOT"]) / "bin" / "spack"
    command = [str(spack_executable), "--color=never"] + list(args)
    env = os.environ.copy()
    # release space in the command-line to prevent saturation
    env.pop("PYTHONPATH", None)
    # run command outside any spack environment
    env.pop("SPACK_ENV", None)
    env.pop("SPACK_ENV_VENV", None)
    # remove it so that spack may export the correct path
    env.pop("HOC_LIBRARY_PATH", None)
    if log:
        print(" ".join(command))
    return subprocess.check_output(command, encoding="utf-8", env=env)


@command
def virtualenv(venv=".venv", spec="py-multiscale-run@develop", **kwargs):
    """Create a Python virtual environment to ease development of multiscale_run Python package

    Basically:
    1. capture output of `spack load --only dependencies --sh py-multiscale-run@develop`
    It contains PATH and PYTHON required to have everything required by multiscale_run but multiscale_run itself
    2. create a Python virtualenv in this context
    3. patch the virtualenv script `bin/activate` with shell script retrived (1)
    """
    # ensure the command is executed from multiscale_run working copy
    git_clone_error = MsrException(
        "this command must be executed from the root directory of multiscale-run git working-copy"
    )
    if not Path(".git").is_dir():
        raise git_clone_error
    pyproject_toml = Path("pyproject.toml")
    if not pyproject_toml.is_file():
        raise git_clone_error
    with open(pyproject_toml) as istr:
        content = istr.read()
        if 'name = "multiscale_run"' not in content:
            raise git_clone_error

    # ensure spack is loaded
    if "SPACK_ROOT" not in os.environ:
        if BB5_JULIA_ENV.exists():
            logging.warning(
                textwrap.dedent(
                    """\
                Cannot locate spack installation. The module is available on BB5:

                    module load unstable spack
            """
                )
            )
        raise MsrException("Cannot locate spack installation")

    spec_out = spack("spec", "--very-long", "--install-status", spec)
    # read output of `spack spec ...` command to extract:
    # - whether the spec is installed or not
    # - the hash of the spec
    sep = "\nConcretized\n--------------------------------\n"
    offset = spec_out.find(sep)
    assert offset != -1
    spec_out = spec_out[offset + len(sep) :]
    spec_out = spec_out.split("\n", 1)[0]
    pkg_name = spec.split("@", 1)[0]
    assert pkg_name in spec_out
    installed = spec_out.startswith("[+]")
    spec_hash = spec_out[3:].split()[0]
    # ------------------------------------------------------
    if not installed:
        spack("install", spec)
    dependencies_env = spack("load", "--only", "dependencies", "--sh", f"/{spec_hash}")

    # Generate a shell-script to create the virtualenv
    fd, installer_path = tempfile.mkstemp(
        suffix=".sh", prefix="msr-venv-installer", text=True
    )
    os.write(fd, b"#!/bin/sh\n")
    os.write(fd, dependencies_env.encode("utf-8"))
    spack_env = os.environ.get("SPACK_ENV")
    if spack_env:
        cleanup_environment = textwrap.dedent(
            f"""\
            # wipe spack environment from PATH and PYTHONPATH
            export PATH=$(echo $PATH | sed -e "s@{spack_env}[^:]*@@g")
            export PYTHONPATH=$(echo $PYTHONPATH | sed -e "s@{spack_env}[^:]*@@g")
        """
        )
        os.write(fd, cleanup_environment.encode("utf-8"))
    os.write(
        fd,
        textwrap.dedent(
            f"""\
        # remove previous references of multiscale-run in PYTHONPATH
        export PYTHONPATH=$(echo $PYTHONPATH | sed -e "s@/[^:]*multiscale-run[^:]*@@g")
        # cleanup previous build artifacts
        rm -rf build multiscale_run.egg-info
        python -m venv {venv}
        {venv}/bin/python -m pip --disable-pip-version-check install --editable .
    """
        ).encode("utf-8"),
    )
    os.close(fd)
    st = os.stat(installer_path)
    os.chmod(installer_path, st.st_mode | stat.S_IEXEC)
    print("Executing shell script: ", installer_path)
    subprocess.check_call(installer_path)

    # patch the virtualenv 'activate' script
    activate_f = Path(venv) / "bin" / "activate"
    with activate_f.open() as istr:
        activate_content = istr.read()
        pos = activate_content.find("# This file must be used with")
        if pos != 1:
            activate_content = activate_content[pos:]
    with activate_f.open("w") as ostr:
        ostr.write(
            textwrap.dedent(
                """\
            _LAST_MODIFIED=$(stat -c "%Y" ${BASH_SOURCE[0]})
            _NOW=$(date +%s)
            if [ $(($_NOW - $_LAST_MODIFIED)) -gt 14515200 ]; then
                >&2 echo "WARNING: This virtualenv is more than 1 week old."
                >&2 echo "WARNING: The spack packages it relies on may be out of date."
                >&2 echo "WARNING: You may consider removing it and creating a fresh one with the 'multiscale-run virtualenv' command."
            fi
            unset _LAST_MODIFIED _NOW
        """
            )
        )
        ostr.write(dependencies_env)
        if spack_env:
            ostr.write(cleanup_environment)
        ostr.write(
            'export PYTHONPATH=$(echo $PYTHONPATH | sed -e "s@/[^:]*multiscale-run[^:]*@@g")\n'
        )
        ostr.write(activate_content)
    print(
        textwrap.dedent(
            f"""\

        Setup is successful.
        multiscale_run has been installed in editable mode in the virtualenv {Path(venv).resolve()}
        The virtualenv has been patched to take into account the spack package dependencies.
        To have multiscale-run executable loaded in the PATH, simply execute the shell command:

            source {Path(venv).resolve()}/bin/activate
     """
        ),
        file=sys.stderr,
    )


def _check_local_neuron_mechanisms():
    """Perform sanity check in case the mod files library has been cloned in the
    simulation directory for local editing.

    Returns
       False if something smelly has been identified, True otherwise
    """
    sane = True
    mod_files = Path("mod")
    if not mod_files.exists():
        return sane

    mod_files_mtime = max(
        entry.stat()[stat.ST_MTIME] for entry in os.scandir(mod_files)
    )

    for ext in [".so", ".dll", ".dylib"]:
        nrn_mechanisms = Path(platform.machine()) / ("libnrnmech" + ext)
        if nrn_mechanisms.exists():
            break
    if (
        not nrn_mechanisms.exists()
        or nrn_mechanisms.stat()[stat.ST_MTIME] < mod_files_mtime
    ):
        sane = False
        LOGGER.warning(
            textwrap.dedent(
                f"""\
            Content of 'mod' directory is more recent that Neuron mechanisms file {nrn_mechanisms}.
            Consider rebuilding it with the following command:

                cd {os.getcwd()}
                build_neurodamus.sh mod
        """
            )
        )

    if nrn_mechanisms.exists():
        env_nrn_mechanisms = os.environ.get("NRNMECH_LIB_PATH")
        if env_nrn_mechanisms != str(nrn_mechanisms.resolve()):
            sane = False
            LOGGER.warning(
                textwrap.dedent(
                    f"""
                A custom version of Neuron mechanisms library have been detected
                    in the simulation directory: {nrn_mechanisms.resolve()}
                but the NRNMECH_LIB_PATH environment variable is not pointing it.
                To use it, you may define the environment variable as follow

                    export NRNMECH_LIB_PATH={nrn_mechanisms.resolve()}
            """
                )
            )
    return sane


@command
def edit_mod_files(**kwargs):
    """Clone the Neurodamus mod files library for local editing

    1. $ cp -r $NEURODAMUS_NEOCORTEX_ROOT ./mod
    2. $ build_neurodamus.sh
    3. Patch simulation.sbatch to override NEURODAMUS_NEOCORTEX_ROOT
    4. Write instructions to the console

    """
    if Path("mod").exists():
        raise MsrException(
            "Directory 'mod' already exists. Remove it first to reinitialize it"
        )

    if (ndam_root := os.environ.get("NEURODAMUS_NEOCORTEX_ROOT")) is None:
        raise MsrException(
            "Environment variable 'NEURODAMUS_NEOCORTEX_ROOT' is missing"
        )
    if os.environ.get("NRNMECH_LIB_PATH") is None:
        raise MsrException("Environment variable 'NRNMECH_LIB_PATH' is missing")
    ndam_mod = Path(ndam_root) / "share" / "neurodamus_neocortex" / "mod"
    if not ndam_mod.exists():
        raise MsrException(f"Directory '{ndam_mod}' does not exist")

    print("copying neocortex mod files library locally")
    shutil.copytree(ndam_mod, "mod")
    print("building local mod files")
    intel_compiler = (
        subprocess.run(
            "readelf -p .comment $NRNMECH_LIB_PATH | grep -q 'Intel(R) oneAPI'",
            shell=True,
        ).returncode
        == 0
    )
    build_cmd = f"build_neurodamus.sh '{ndam_mod}' --only-neuron"
    if BB5_JULIA_ENV.exists() and intel_compiler:
        build_cmd = "module load unstable intel-oneapi-compilers ; " + build_cmd
    subprocess.check_call(build_cmd, shell=True)
    nrn_mechanims = (Path(platform.machine()) / "libnrnmech.so").resolve()
    if not nrn_mechanims.exists():
        raise MsrException(
            f"Missing file expected to be built by 'build_neurodamus.sh': '{nrn_mechanims}'"
        )

    sbatch_script = Path("simulation.sbatch")
    if sbatch_script.exists():
        with sbatch_script.open() as istr:
            content = istr.read()
        if "export NRNMECH_LIB_PATH" not in content:
            print(
                f"Patching {sbatch_script} to take the new Neuron mechanisms into account"
            )
            content = content.replace(
                "-----\n\n",
                f"-----\n\n# Use local Neuron mechanisms\nexport NRNMECH_LIB_PATH={nrn_mechanims}\n\n",
            )
            with sbatch_script.open("w") as ostr:
                ostr.write(content)
    else:
        LOGGER.warning(f"Could not find '{sbatch_script}. Skip patching")

    print(
        textwrap.dedent(
            f"""\
        Neuron mechanisms have been successfully built.
        Define the following environment variable to take them into account during multiscale-run simulations:

            export NRNMECH_LIB_PATH={nrn_mechanims}

        Whenever you modify the mod files, launch this command to rebuild the mechanisms:

            cd {os.getcwd()}
            build_neurodamus.sh mod --only-neuron

        Happy hacking!
    """
        )
    )


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--verbose", "-v", action="count", default=0)
    subparsers = ap.add_subparsers(title="commands")

    parser_init = subparsers.add_parser("init", help=init.__argparse_help__)
    parser_init.set_defaults(func=init)
    parser_init.add_argument(
        "--circuit",
        choices=sorted(list(NAMED_CIRCUITS)),
        default=DEFAULT_CIRCUIT,
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
    parser_init.add_argument(
        "--no-check",
        action="store_const",
        const=False,
        default=True,
        dest="check",
        help="Do not verify simulation health",
    )
    parser_init.add_argument("directory", nargs="?")

    parser_check = subparsers.add_parser("check", help=check.__argparse_help__)
    parser_check.set_defaults(func=check)
    parser_check.add_argument("directory", nargs="?")

    parser_compute = subparsers.add_parser("compute", help=compute.__argparse_help__)
    parser_compute.set_defaults(func=compute)
    parser_compute.add_argument("directory", nargs="?")

    parser_postproc = subparsers.add_parser(
        "post-processing", help=post_processing.__argparse_help__
    )
    parser_postproc.set_defaults(func=post_processing)
    parser_postproc.add_argument(
        "--notebook",
        default=MSR_POSTPROC.name,
        help="path to the Jupyter notebook to execute. Default is %(default)s",
    )
    parser_postproc.add_argument("directory", nargs="?")

    parser_julia = subparsers.add_parser("julia", help=julia.__argparse_help__)
    parser_julia.set_defaults(func=julia)
    parser_julia.add_argument(
        "args",
        nargs="*",
        metavar="ARGS",
        help="Optional arguments passed to Julia executable",
    )

    parser_virtualenv = subparsers.add_parser(
        "virtualenv", help=virtualenv.__argparse_help__
    )
    parser_virtualenv.set_defaults(func=virtualenv)
    parser_virtualenv.add_argument(
        "--venv",
        default=".venv",
        help="Name of the virtual environment. Default is %(default)s",
    )
    parser_virtualenv.add_argument(
        "--spec",
        default="py-multiscale-run@develop",
        help="Spack spec to determine the list of dependencies to install",
    )

    parser_edit_mod_files = subparsers.add_parser(
        "edit-mod-files", help=edit_mod_files.__argparse_help__
    )
    parser_edit_mod_files.set_defaults(func=edit_mod_files)
    parser_edit_mod_files.add_argument("directory", nargs="?")

    return ap


def main(**kwargs):
    """Package script entry-point for the multiscale-run CLI utility.

    Args:
        kwargs: optional arguments passed to the argument parser.
    """
    from .utils import comm, size

    ap = argument_parser()
    args = ap.parse_args(**kwargs)
    args = vars(args)

    warnings.filterwarnings("ignore", category=scipy.sparse.SparseEfficiencyWarning)

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
            LOGGER.error(str(e))
            if size() > 1:
                comm().Abort(errorcode=1)
            else:
                sys.exit(1)
    else:
        ap.error("a subcommand is required.")
