# multiscale_run

1. dualrun : Coupling Neuron & STEPS. The dualrun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam). The official repo for dualrun is now the [following](https://bbpgitlab.epfl.ch/molsys/dualrun).
1. triplerun : Coupling dualrun & metabolism. The triplerun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
1. quadrun : Coupling triplerun & blood flow. [**WIP**]

# Multiscale_run on BB5

You may not use multiscale-run on the login nodes bbpv1 and bbpv2 to either configure or run a simulation.
```
salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 02:00:00 --cpus-per-task=2 --constraint=clx
```

## As a module (recommended)

```
module load unstable py-multiscale-run
```

## As a spack package:

```
spack install py-multiscale-run@develop
spack load py-multiscale-run
```

Using spack environments is recommended so that you can work in an isolated environment with only the multiscale_run required spack packages.
More info [here](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md)

> :rainbow: **This may also work on your spack-powered machine!**


## Your own working-copy of multiscale_run

In this section we explain how to get your own working copy of multiscale run. Typically, these methods are employed by developers that want to customize multiscale run itself or one or more dependencies.

There are 2 ways to do that:

- **Multiscale run virtualenv**: <u>recommended</u> when you just need to change multiscale run code.
- **Spack environments**: all the other cases.

### Multiscale run virtualenv (python-venv)

First, you need one version (and only one) of `py-multiscale-run@develop` in your spack (you need this only once):

```
spack uninstall --all py-multiscale-run@develop
spack install
```

Then, you can install your Python virtual environment `.venv` with this simple series of commands:

```bash
git clone git@bbpgitlab.epfl.ch:molsys/multiscale_run.git /path/to/multiscale_run
cd /path/to/multiscale_run
module load unstable py-multiscale-run
multiscale-run virtualenv
```

Then activate the virtualenv environment to work with your working-copy: `.venv/bin/activate`

```bash
$ .env/bin/multiscale-run --version
multiscale-run 0.3.dev9+g6b660cb
```

> :rainbow: **This may also work on your spack-powered machine!**


### Spack environments

This section is a specialization of this [generic spack guide](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md). 

As a concrete example, let's say that we want to make some modifications in neurodamus and multiscale run and see how the code performs in rat V6. Let's also assume that we are on BB5 with a working spack. If it is not the case please check [here](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/setup_bb5.md) on how to install spack on BB5. 

It is **recommended to allocated a node** before starting. Compiling in the log in node is deprecated. 

Let's first clone the repositories:

```
git clone git@bbpgitlab.epfl.ch:molsys/multiscale_run.git
git clone git@github.com:BlueBrain/neurodamus.git
```

Our environemnt will be called `spackenv`. Let's create and activate it:

```
spack env create -d spackenv
spack env activate -d spackenv
```

Now, we should have 3 folders:

```
.
├ multiscale_run
├ neurodamus
└ spackenv
```

Let's add neurodamus and tell spack to use the source code that we cloned before:

```
spack add py-neurodamus@develop
spack develop -p ${PWD}/neurodamus --no-clone py-neurodamus@develop
```

And let's do the same for multiscale run:

```
spack add py-multiscale-run@develop
spack develop -p ${PWD}/multiscale_run --no-clone py-multiscale-run@develop
```

Now we can finally install:

```
spack install
```

In order to be sure that all changes have been in effect and `$PYTHONPATH` is populated properly (note that this is only needed when you set up the Spack environment the first time):

```
spack env deactivate
spack env activate -d spackenv
```

Now you are ready to test your multiscale run (follow the section **How to use the `multiscale-run` executable?**). If you use slurm you need to remove the py-multiscale-run and, instead, load the spackenv environment. In concrete terms, in `simulation.sbatch` you need to replace this line:

```
module load py-multiscale-run
```

with this line (assuming that your spackenv is in ~. Change the location accordingly):

```
spack env activate -d ~/spackenv
```

You may also need to load `gmsh`:

```
module load unstable gmsh
```

**Important:**

Remember that every time you add a modification to the code you need to call `spack install` before testing it.

> :rainbow: **This may also work on your spack-powered machine!**

# How to use the `multiscale-run` executable?

This program provides several commands to initialize, configure and execute simulations

## Setup a new simulation

```shell
multiscale-run init /path/to/my-sim
```

This command creates the following files in `/path/to/my-sim` providing both the circuit, the config files, and the runtime dependencies:

```
.
├── circuit_config.json
├── msr_config.json
├── node_sets.json
└── simulation_config.json
├── postproc.ipynb
└── simulation.sbatch
```

The generated setup is ready to compute, but feel free to browse and tweak the JSON configuration files at will!

> :ledger: **See `multiscale-run init --help` for more information**

## Verify a simulation configuration

This command performs a series of check to identify common mistakes in the configuration. It is recommended 
to perform this operation before starting a simulation.

```shell
multiscale-run check [/path/to/my/sim]
```

## Compute the simulation

### On the current machine / allocation

```shell
multiscale-run compute [/path/to/my-sim]
```

> :ledger: To use multiple ranks, use `srun -n X multiscale-run compute` where X is the number of ranks. Notice that steps requires this to be a power of 2.

> :ledger: **See `multiscale-run compute --help` for more information**

> :exclamation: **You may need to load the `intel-oneapi-mkl` module on BB5 if not already loaded**
> otherwise you will probably experience the following error when running the _compute_ phase: `libmkl_intel_thread.so.1: undefined symbol: omp_get_num_procs`

Three more folders will be created during the simulation: 
* `cache`: it keeps some cached communication matrices, useful for subsequent runs
* `mesh`: mesh files for steps. If the folder is missing a standard mesh will be generated Just In Time
* `RESULTS`: the results of the simulation. Various data are recorded here in hdf5 files. The variables are per-neuron based. Multiscale run mymics the structure of neurodamus result files so that they can be postprocessed with the same method

> :For efficiency reasons, multiscale run result files are filled with 0s at the beginning of the simulation. Thus, if the simulation dies early, these files will be full of 0s for the time steps after the simulation died

### On SLURM cluster:

```
sbatch simulation.sbatch
```

## Custom Neuron mechanisms

This operation clones the Neurodamus mod files library for local editing.
```shell
multiscale-run edit-mod-files [/path/to/my-sim]
```

# Other operations

## Convert BlueConfig to SONATA compatible json file

1. Setup a development environment with the `multiscale-run virtualenv` operation
1. Clone the [bluepy-configfile repository](https://bbpgitlab.epfl.ch/nse/bluepy-configfile)
1. Go to the repository and execute: `pip install .`
1. In the BlueConfig (BC) of interest comment out the `RunMode` field, the `Circuit GLIA` section, the `Projection NeuroGlia` section, and the the `Projection GlioVascular` section.
1. Go back to the multiscale_run repo and run `blueconfig convert-to-sonata /gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/circuit_config.json ngv.json user.target BlueConfig`. The first argument, i.e. circuit_config, should point to an existing SONATA circuit config file, which can be found by searching the folder defined in `CircuitPath` field of the BlueConfig. The second argument is the name of the SONATA simulation config file to be created from the BlueConfig. The third argument is the path to an existing SONATA nodesets file, and the fourth argument is the BlueConfig that we want to convert.
1. Now the `ngv.json` can be used from the Jupyter notebook for the visualizations.

## Profile multiscale_run script with ARM MAP

1. Load ARM MAP in the environment: `module load arm-forge`
1. Run the simulation: `map --profile --output OUT.map srun [...] multiscale-run compute [/path/to/my-sim]`
1. Open `OUT.map` with the Arm Forge Client (locally)

For more on how to use ARM MAP check [here](https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?spaceKey=BBPHPC&title=How+to+use+Arm+MAP).

# Release notes

## 0.5 - 2024-01-29

* Improved README
* `msr_config.json` is a template now. Prepared for the bbp workflow
* `base_path` is now specified in the main config file. This is breaking change. Add to your `msr_config.json` file a field: `"base_path": "."`
* `ndam` is now forced to use `RoundRobin`

## 0.4 - 2024-01-19

* New commands:
  * `edit-mod-files`: Clone the Neurodamus mod files library for local editing
  * `virtualenv`: Create a Python virtual environment to ease development of multiscale_run Python package
* Fix possible program hang due to sensitive MPI initialization
* Improved README
* Added a section for `spack environments`
* Increased standard mesh refinement for ratV10 to not have ranks without tets (omega_h cannot handle them)
* Align to bbp workflow

## 0.3 - 2024-01-10

* new `post-processing` command creates an HTML report based on simulation results. Usage: `multiscale-run post-processing [sim-path]`. Use `multiscale-run post-processing -h` for more information.
* `init` command:
  * new option `--no-check` to skip the tests of Julia environment which lasts several minutes
  * new keys in `msr_config.json`:
    * `config_format`: an integer, the version of this file structure
    * `msr_version`: a string indicating the version of multiscale-run that created this config
* `compute` command: now creates a Neurodamus success file at the end of the simulation
* Compatibility notes for simulations created with previous versions of multiscale-run
  * rename directory `julia_environment` to `.julia_environment`
  * it is not possible to override the JSON configuration keys with environment variables.
* GitLab CI on BB5 now relies on spack

## 0.2 - 2023-12-14

* Rework reporting [BBPP40-402, BBPP40-407, BBPP40-410, BBPP40-411, BBPP40-415]

## 0.1 - 2023-11-30

First release of the code shipped as a Python package
