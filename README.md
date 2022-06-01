# multiscale_run

1. dualrun : Coupling Neuron & STEPS. The dualrun part of the script (STEPS3 version) is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).

2. triplerun : Coupling dualrun & metabolism. The triplerun part of the script (STEPS3 version) is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).

3. quadrun : Coupling triplerun & blood flow. [**WIP**]

The BlueConfig & user.target are based on the ones found in [this repo](https://bbpgitlab.epfl.ch/hpc/sim/blueconfigs/-/tree/main/ngv-v6).

The mod folder is based on the one found in [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/tree/main/triplerun/custom_ndam_special_v1/polina_mod) with the addition of the mod files found [here](https://bbpgitlab.epfl.ch/hpc/sim/models/common/-/tree/main/mod/ngv). For the `ProbAMPANMDA_EMS.mod` & `ProbGABAAB_EMS.mod`, we use the ones found in `custom_ndam_2021_02_22_archive202101` folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/tree/main/custom_ndam_2021_02_22_archive202101/mod).

The `multiscale_run_STEPS3.py` (STEPS3 compatible only) & `multiscale_run_STEPS4.py` (STEPS4 compatible only) scripts execute the various runs. Just by exporting the env var `which_run`, the user can decide which coupling wants to execute, e.g. `export which_run=2` realizes the dualrun and `export which_run=3` the triplerun.

## Environment Setup

1. [Set Spack up on BlueBrain5](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/setup_bb5.md). Do not rely on `module load spack` and instead set it up in your `$HOME` folder as instructed.
    * Add these lines in your `.bashrc`:
        ```
        . $HOME/spack/share/spack/setup-env.sh
        export SPACK_SYSTEM_CONFIG_PATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/config
        export SPACK_USER_CACHE_PATH=$HOME/spack_install
        ```
    * There is no need for `module load spack`
2. `salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 00:30:00 --cpus-per-task=2 --constraint=clx`
3. `source set_env.sh`
4. `exit`

## Run Simulation

1. **Environment Setup** first
2. `sbatch job_script`
