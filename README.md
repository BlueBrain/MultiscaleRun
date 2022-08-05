# multiscale_run

1. dualrun : Coupling Neuron & STEPS. The dualrun part of the script is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam). The official repo for dualrun is now the [following](https://bbpgitlab.epfl.ch/molsys/dualrun).

1. triplerun : Coupling dualrun & metabolism. The triplerun part of the script is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).

1. quadrun : Coupling triplerun & blood flow. [**WIP**]

The `multiscale_run_STEPS3.py` (STEPS3 compatible only) script executes the various runs. Just by exporting specific environment variables (see `job_script`), the user can decide which coupling wants to execute, i.e. dual/triple/quad-run.

## Environment Setup

1. [Set Spack up on BlueBrain5](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/setup_bb5.md). Do not rely on `module load spack` and instead set it up in your `$HOME` folder as instructed (main steps here, but follow the spack README for more):
    * `git clone -c feature.manyFiles=true https://github.com/BlueBrain/spack.git` (`HOME` dir)
    * Add these lines in your `.bashrc`:
        ```
        . $HOME/spack/share/spack/setup-env.sh
        export SPACK_SYSTEM_CONFIG_PATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/config
        export SPACK_USER_CACHE_PATH=$HOME/spack_install
        ```
    * There is no need for `module load spack`
1. `source .bashrc`
1. `salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 00:30:00 --cpus-per-task=2 --constraint=clx`
1. `source set_env.sh`
1. `exit`

## Run Simulation

1. **Environment Setup** first
1. `sbatch job_script`
