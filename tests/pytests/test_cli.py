import json
import os
from pathlib import Path
import platform
import subprocess
import stat
import textwrap

import pytest

from multiscale_run import utils
from multiscale_run.cli import argument_parser, main
from multiscale_run.data import BB5_JULIA_ENV


def test_init_without_metabolism(tmp_path):
    sim_path = tmp_path / "sim"
    main(args=["init", "--julia=no", str(sim_path)])

    assert sim_path.is_dir()

    msr_config_f = sim_path / "msr_config.json"
    assert msr_config_f.is_file(), "top JSON config is missing"
    with open(msr_config_f) as istr:
        msr_config = json.load(istr)
    assert "with_metabolism" in msr_config, "JSON config is missing a mandatory key"
    assert msr_config["with_metabolism"] is False, "Metabolism should be disabled"


@pytest.mark.skipif(not BB5_JULIA_ENV.exists(), reason="BB5 resources required")
def test_init_twice(tmp_path):
    sim_path = tmp_path / "sim"

    # Prepare a new simulation with "shared" Julia setup
    main(args=["init", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()

    # Recreate the simulation at the same location
    main(args=["init", "--force", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()

    # mess a little the Julia setup
    (sim_path / ".julia").unlink()
    (sim_path / ".julia").mkdir()

    # Recreate the simulation at the same location
    main(args=["init", "--force", "--no-check", str(sim_path)])
    assert (sim_path / ".julia").is_symlink()
    assert (sim_path / ".julia_environment").is_symlink()


def test_valid_commands(tmp_path):
    path = str(tmp_path)
    ap = argument_parser()
    try:
        ap.parse_args(["init"])
        ap.parse_args(["check"])
        ap.parse_args(["compute"])
        ap.parse_args(["julia"])

        ap.parse_args(["-v", "init"])
        ap.parse_args(["-vv", "init"])

        ap.parse_args(["init", path])
        ap.parse_args(["init", "--force", path])
        ap.parse_args(["init", "--julia", "no", path])
        ap.parse_args(["init", "--julia=no", path])
        ap.parse_args(["init", "--circuit=rat_sscxS1HL_V6", path])

        ap.parse_args(["check", path])
        ap.parse_args(["compute", path])
        ap.parse_args(["julia", 'print("Foo")', 'Pkg.install("Foo")'])
    except:
        pytest.fail("Parsing of valid commands failed")


def test_virtualenv():
    """Run virtualenv in the base multiscale run folder

    - move to multiscale run base folder
    - remove old venv folders
    - run the venv command
    - clean up
    """

    # # clean up before
    venv = Path.cwd() / "venv"
    venvdo = Path.cwd() / "venvdo.sh"
    utils.remove_path(venv)
    utils.remove_path(venvdo)

    subprocess.check_call(["multiscale-run", "virtualenv", "--venv", str(venv)])
    assert (venv / "bin" / "multiscale-run").exists()

    with venvdo.open("w") as ostr:
        ostr.write(
            textwrap.dedent(f"""\
            #!/bin/bash

            source {venv}/bin/activate

            if echo $PYTHONPATH | grep -q multiscale-run ; then
                echo "Error: multiscale-run should not be available in PYTHONPATH: " >&2
                echo $PYTHONPATH | tr : "\\n" | grep multiscale-run >&2
                exit 1
            fi

            $@
        """
            )
        )
    st = venvdo.stat()
    os.chmod(venvdo, st.st_mode | stat.S_IEXEC)

    subprocess.check_call([str(venvdo), "multiscale-run", "--version"])
    assert "multiscale_run" in subprocess.check_output(
        [str(venvdo.resolve()), "pip", "list", "-e"], cwd="..", encoding="utf-8"
    )
    # clean up after
    utils.remove_path(venv)
    utils.remove_path(venvdo)


def test_edit_mod_files(tmp_path):
    path = str(tmp_path)

    def msr(*args, check=True, **kwargs):
        command = ["multiscale-run"] + list(args)
        if check:
            return subprocess.check_call(command, **kwargs)
        return subprocess.call(command, **kwargs)

    msr("init", "--force", "--julia", "no", path)

    msr("edit-mod-files", path)
    mod_dir = tmp_path / "mod"
    libnrnmech = tmp_path / platform.machine() / "libnrnmech.so"
    assert mod_dir.is_dir()
    assert libnrnmech.is_file()

    # But NRNMECH_LIB_PATH is not pointing at new the new libnrnmech.so
    assert msr("check", path, check=False) == 1

    proper_env = os.environ.copy()
    proper_env["NRNMECH_LIB_PATH"] = str(libnrnmech.resolve())
    # this time, "check" doesn't complain
    msr("check", path, env=proper_env)

    # Let's make libnrnmech.so outdated by modifying one of the mod files
    (mod_dir / "Ca.mod").touch()
    # the "check" command is able to detect that and emit a warning about it
    assert msr("check", path, env=proper_env, check=False) == 1

    # let's update the mechanisms library
    intel_compiler = subprocess.run("readelf -p .comment $NRNMECH_LIB_PATH | grep -q 'Intel(R) oneAPI'", shell=True).returncode == 0
    build_cmd = "build_neurodamus.sh mod"
    if BB5_JULIA_ENV.exists and intel_compiler:
        build_cmd = "module load unstable intel-oneapi-compilers ; " + build_cmd
    subprocess.check_call(build_cmd, shell=True, cwd=path)

    # this time, "check" doesn't complain
    msr("check", path, env=proper_env)
