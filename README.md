# multiscale_run

1. dualrun : Coupling Neuron & STEPS. The dualrun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam). The official repo for dualrun is now the [following](https://bbpgitlab.epfl.ch/molsys/dualrun).
1. triplerun : Coupling dualrun & metabolism. The triplerun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
1. quadrun : Coupling triplerun & blood flow. [**WIP**]

The `main.py` script executes the various runs. Just by exporting specific environment variables (see `job_script`), the user can decide which coupling wants to execute or even mix them, i.e. dual/triple/quad-run.

## Environment Setup

First of all, **update spack**. More info [here](https://github.com/BlueBrain/spack).

You also need access to [this](https://github.com/CNS-OIST/HBP_STEPS) git repository. Contact Alessandro Cattabiani (Katta)
in case you do not already have access rights.

The environment setup is pretty involved because there are a lot of different, contributing projects. Fortunately, it was all figured out!
Just `source setup.sh` before proceeding with the simulations. It sets up all the environment for you and download the repos, packages and modules that you need.
For problems contact Katta.

If it is the first time you call `setup.sh` it is suggested to allocate a node to not stress the login node. For example:
```
salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 02:00:00 --cpus-per-task=2 --constraint=clx
```

## Julia environment

Julia's enviroment is frozen (packages with specific versions) and completely defined by `Project.toml` and `Manifest.toml` found in `./julia_environment` folder.

If for some reason, you want to update the environment, either modify `Project.toml` and `Manifest.toml`, or delete `./julia_environment` (along with `.julia` folder) and re-run the setup script.
The latter approach will regenerate `./julia_environment` folder.

## Run Simulation

Run your job with sbatch. For example: `sbatch job_script`. 
The parameters are either in `params.py` and `job_script`. Check them for more info. 

## Spack

We now use environments! In this way we can keep our standard spack separated with the one needed for multiscale_run. More info [here](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md)


## Custom special

 
The special is being built in a fully automated way with `setup.sh` through spack environments. More info [here](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md).
Thus, a custom special is usually not needed. Conversely it is somewhat discouraged since it can lead to inconsistencies on what mod files are used). 
However, in the case you want to change some mod files and test them quickly, this is the workaround for you.
The custom special is based on the mod files of the `custom_ndam_2021_02_22_archive202101` folder, 
found [here](https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/tree/main/custom_ndam_2021_02_22_archive202101). 
This folder has been curated and now resides in [neocortex repo](https://bbpgitlab.epfl.ch/hpc/sim/models/neocortex) in
`mod/metabolism`.

After `setup.sh`, all the gathered mod files can be found following the steps below:
1. ``` ndam_installation_dir=`spack find --paths neurodamus-neocortex@develop+ngv+metabolism | tail -n 1 | grep -o "/.*"` ```
1. mod files -> `$ndam_installation_dir/share/mod_full`

Before running you should place all your mod files in the `mod` folder. After:

```bash
build_neurodamus.sh mod
```

`job_script` should pick up automatically the custom special (if the `x86_64` folder is there). In case you want to fix 
you can override this behavior in `job_script` itself by uncommenting `special_path`.


## Convert BlueConfig to SONATA compatible json file

1. Activate the python virtual environment, i.e. `source setup.sh`.
1. Clone the [bluepy-configfile repo](https://bbpgitlab.epfl.ch/nse/bluepy-configfile).
1. Go to the repo and do `pip install .`.
1. In the BlueConfig (BC) of interest comment out the `RunMode` field, the `Circuit GLIA` section, the `Projection NeuroGlia` section, and the the `Projection GlioVascular` section.
1. Go back to the multiscale_run repo and run `blueconfig convert-to-sonata /gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/circuit_config.json ngv.json user.target BlueConfig`. The first argument, i.e. circuit_config, should point to an existing SONATA circuit config file, which can be found by searching the folder defined in `CircuitPath` field of the BlueConfig. The second argument is the name of the SONATA simulation config file to be created from the BlueConfig. The third argument is the path to an existing SONATA nodesets file, and the fourth argument is the BlueConfig that we want to convert.
1. Now the `ngv.json` can be used from the jupyter notebook for the visualizations.

## Profile multiscale_run script with ARM MAP

In the `job_script`:
1. After the `source ./set_env.sh ...`, add the following line: `module load arm-forge`.
1. Execute `map --profile --output OUT.map srun special -mpi -python ${PYDRIVER}`.
1. Open `OUT.map` with the Arm Forge Client (locally).

For more on how to use ARM MAP check [here](https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?spaceKey=BBPHPC&title=How+to+use+Arm+MAP).

**REMARK**: In the past, at the very end of the multiscale_run script we were calling `exit() # needed to avoid hanging`. However, this `exit` is crashing ARM MAP because some processes exit before `MPI_Finalize` is called. In the latest versions of Neurodamus and STEPS, there is no need for using `exit()` and therefore it has been removed from the script.
