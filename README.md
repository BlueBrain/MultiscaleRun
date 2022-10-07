# multiscale_run

1. dualrun : Coupling Neuron & STEPS. The dualrun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam). The official repo for dualrun is now the [following](https://bbpgitlab.epfl.ch/molsys/dualrun).
1. triplerun : Coupling dualrun & metabolism. The triplerun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
1. quadrun : Coupling triplerun & blood flow. [**WIP**]

The `multiscale_run_STEPS3.py` (STEPS3 compatible only) script executes the various runs. Just by exporting specific environment variables (see `job_script`), the user can decide which coupling wants to execute or even mix them, i.e. dual/triple/quad-run.

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
1. `salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 02:00:00 --cpus-per-task=2 --constraint=clx`
1. `source set_env.sh`
1. `exit`

## Run Simulation

1. **Environment Setup** first
1. `sbatch job_script`

## Custom special

The special is being built in a fully automated way:
```
module load neurodamus-neocortex-multiscale_run
```
or 
```
spack install/load neurodamus-neocortex@develop+ngv+metabolism
```

The custom special is based on the mod files of the `custom_ndam_2021_02_22_archive202101` folder, found [here](https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/tree/main/custom_ndam_2021_02_22_archive202101). This folder has been curated and now resides in [neocortex repo](https://bbpgitlab.epfl.ch/hpc/sim/models/neocortex) in `mod/metabolism`.

After spack installation, all the gathered mod files can be found following the steps below:
1. ``` ndam_installation_dir=`spack find --paths neurodamus-neocortex@develop+ngv+metabolism | tail -n 1 | grep -o "/.*"` ```
1. mod files -> `$ndam_installation_dir/share/mod_full`

## Convert BlueConfig to SONATA compatible json file

Currently, there is a [merge request in bluepy-configfile repo](https://bbpgitlab.epfl.ch/nse/bluepy-configfile/-/merge_requests/11) that implements this converter.

1. Prepare your python virtual environment.
1. Clone the [bluepy-configfile repo](https://bbpgitlab.epfl.ch/nse/bluepy-configfile). If the above-mentioned MR is merged then proceed to the next step, if not just checkout to the relevant branch, i.e. `jblanco/convert_blueconfig`.
1. Go to the repo and do `pip install .`.
1. In the BlueConfig (BC) of interest comment out the `RunMode` field, the `Circuit GLIA` section, the `Projection NeuroGlia` section, and the the `Projection GlioVascular` section.
1. Go to the multiscale_run repo and run `blueconfig convert-to-sonata ngv.json ./BlueConfig`.
1. In the `ngv.json`, change the `network` field to point to the correct `circuit_config.json` file.
1. Now the `ngv.json` can be used from the jupyter notebook for the visualizations.
