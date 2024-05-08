# MultiscaleRun

1. dualrun : Coupling Neuron & STEPS. The dualrun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam). The official repo for dualrun is now the [following](https://bbpgitlab.epfl.ch/molsys/dualrun).
1. triplerun : Coupling dualrun & metabolism. The triplerun is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
1. quadrun : Coupling triplerun & blood flow. [**WIP**]

# MultiscaleRun on BB5

You may not use MultiscaleRun on the login nodes bbpv1 and bbpv2 to either configure or run a simulation.
```
salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 02:00:00 --cpus-per-task=2 --constraint=clx
```

## As a module (recommended)

```
module load unstable py-multiscale-run
```

## As a spack package

```
spack install py-multiscale-run@develop
spack load py-multiscale-run
```

Using spack environments is recommended so that you can work in an isolated environment with only the MultiscaleRun required spack packages.
More info [here](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md)

> :rainbow: **This may also work on your spack-powered machine!**


## Your own working-copy of MultiscaleRun

In this section we explain how to get your own working copy of MultiscaleRun. Typically, these methods are employed by developers that want to customize MultiscaleRun itself or one or more dependencies.

There are 2 ways to do that:

- **MultiscaleRun virtualenv**: <u>recommended</u> when you just need to change MultiscaleRun code.
- **Spack environments**: all the other cases.

### MultiscaleRun virtualenv (python-venv)

Create a Python virtual environment `.venv` with this simple series of commands:

```bash
git clone git@bbpgitlab.epfl.ch:molsys/multiscale_run.git /path/to/multiscale_run
cd /path/to/multiscale_run
module load unstable py-multiscale-run
multiscale-run virtualenv
```

Then activate the virtualenv environment to work with your working-copy: `source .venv/bin/activate`

```bash
$ .venv/bin/multiscale-run --version
multiscale-run 0.7
```

> :rainbow: **This may also work on your spack-powered machine!**


### Spack environments

This section is a specialization of this [generic spack guide](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md). 

As a concrete example, let's say that we want to make some modifications in Neurodamus and MultiscaleRun and see how the code performs in rat V6. Let's also assume that we are on BB5 with a working spack. If it is not the case please check [the spack documentation on BB5](https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/setup_bb5.md) on how to install spack on BB5. 

It is **recommended to allocated a node** before starting. Compiling in the log in node is deprecated. 

Let's first clone the repositories:

```
git clone git@bbpgitlab.epfl.ch:molsys/multiscale_run.git
git clone git@github.com:BlueBrain/neurodamus.git
```

Our environment will be called `spackenv`. Let's create and activate it:

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

Let's add Neurodamus and tell spack to use the source code that we cloned before:

```
spack add py-neurodamus@develop
spack develop -p ${PWD}/neurodamus --no-clone py-neurodamus@develop
```

And let's do the same for MultiscaleRun:

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

Now you are ready to test your version of MultiscaleRun (follow the section **How to use the `multiscale-run` executable?**). If you use SLURM you need to remove the py-multiscale-run and, instead, load the spackenv environment. In concrete terms, in `simulation.sbatch` you need to replace this line:

```
module load py-multiscale-run
```

with these lines (assuming that your spackenv is in ~. Change the location accordingly):

```
module load llvm
spack env activate -d ~/spackenv
```

If you want to run an interactive session instead you need the following modules too:

```
module load unstable llvm
```

**Important:**

Remember that every time you add a modification to the code you need to call `spack install` before testing it.

> :rainbow: **This may also work on your spack-powered machine!**

# How to use the MultiscaleRun executable?

This program provides several commands to initialize, configure and execute simulations

## Setup a new simulation

```shell
multiscale-run init /path/to/my-sim
```

This command creates the following files in `/path/to/my-sim` providing both the circuit, the configuration files, and the runtime dependencies:

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

> :exclamation: **You may need to load the `intel-oneapi-mkl` module on BB5 if not already loaded** otherwise you will probably experience the following error when running the _compute_ phase: `libmkl_intel_thread.so.1: undefined symbol: omp_get_num_procs`

Three more folders will be created during the simulation: 
* `cache`: it keeps some cached communication matrices, useful for subsequent runs
* `mesh`: mesh files for steps. If the folder is missing a standard mesh will be generated Just In Time
* `RESULTS`: the results of the simulation. Various data are recorded here in HDF5 files. The variables are per-neuron based. MultiscaleRun mimics the structure of Neurodamus result files so that they can be post-processed with the same method

> :For efficiency reasons, MultiscaleRun result files are filled with 0s at the beginning of the simulation. Thus, if the simulation dies early, these files will be full of 0s for the time steps after the simulation died

### On SLURM cluster:

```
sbatch simulation.sbatch
```

## Custom Neuron mechanisms

This operation clones the Neurodamus mod files library for local editing.

```shell
multiscale-run edit-mod-files [/path/to/my-sim]
```

### Troubleshooting

The command `build_neurodamus.sh mod` may fail with the following error:
```
=> LINKING shared library ./libnrnmech.so
/usr/bin/ld: /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/stage_applications/install_oneapi-2023.2.0-skylake/neuron-9.0.a15-lrspl6/lib/libnrniv.so: unable to initialize decompress status for section .debug_info
/usr/bin/ld: /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/stage_applications/install_oneapi-2023.2.0-skylake/neuron-9.0.a15-lrspl6/lib/libnrniv.so: unable to initialize decompress status for section .debug_info
/gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/stage_applications/install_oneapi-2023.2.0-skylake/neuron-9.0.a15-lrspl6/lib/libnrniv.so: file not recognized: File format not recognized
icpx: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [/gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/stage_applications/install_oneapi-2023.2.0-skylake/neuron-9.0.a15-lrspl6/bin/nrnmech_makefile:133: mech_lib_shared] Error 1
```
This happens because Neuron was built with Intel oneAPI compiler but the compiler is not available in the environment. Loading the proper module on BB5 may fix the issue `module load unstable  intel-oneapi-compilers`

# Other operations

## Convert BlueConfig to SONATA compatible JSON file

1. Setup a development environment with the `multiscale-run virtualenv` operation
1. Clone the [bluepy-configfile repository](https://bbpgitlab.epfl.ch/nse/bluepy-configfile)
1. Go to the repository and execute: `pip install .`
1. In the BlueConfig (BC) of interest comment out the `RunMode` field, the `Circuit GLIA` section, the `Projection NeuroGlia` section, and the the `Projection GlioVascular` section.
1. Go back to the MultiscaleRun repo and run `blueconfig convert-to-sonata /gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/circuit_config.json ngv.json user.target BlueConfig`. The first argument, i.e. circuit_config, should point to an existing SONATA circuit config file, which can be found by searching the folder defined in `CircuitPath` field of the BlueConfig. The second argument is the name of the SONATA simulation config file to be created from the BlueConfig. The third argument is the path to an existing SONATA nodesets file, and the fourth argument is the BlueConfig that we want to convert.
1. Now the `ngv.json` can be used from the Jupyter notebook for the visualizations.

## Profile MultiscaleRun script with ARM MAP

1. Load ARM MAP in the environment: `module load arm-forge`
1. Run the simulation: `map --profile --output OUT.map srun [...] multiscale-run compute [/path/to/my-sim]`
1. Open `OUT.map` with the Arm Forge Client (locally)

For more on how to use ARM MAP on BB5, please check [this page](https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?spaceKey=BBPHPC&title=How+to+use+Arm+MAP).

# Release notes

## Upcoming release

### Major Changes

* The configuration file has changed in the `connections` section:
  *  you can specify where the connection takes place and if you want to write the connection results (`vals`) in the source simulator too.
  * matrices are now part of the `op` operation and can be used throughout the code.
  * A `dest_simulator` key is now required for every connection as `connect_to` keys do not dictate which simulator is the destination simulator anymore.
  * Previous `simulation_config.json` files must be adapted since backward compatibility is not possible. However, changes should be small and self-explanatory if a new template is compared to an old version (v 0.7, `config_format`: 2) of the configuration file.

## 0.7 - 2024-04-10

### Major Changes

#### Rework of the config object and the config file

* The configuration file `msr_config.json` does not exist anymore. All the MultiscaleRun configurations have been merged into `simulation_config.json` in a new root section:`multiscale_run`. This is not yet SONATA. The configuration is in the same file but Neurodamus is not aware of it. [BBPP40-455].
* The configuration has changed significantly from 0.6. The [documentation](https://bbpteam.epfl.ch/documentation/projects/multiscale_run) has now a dedicated page to explain how to use it.
* All the `magic numbers` in metabolism have been moved to the configuration or eliminated. For this reason, simulation behavior changed. Currently, neurons develop an ATP deficiency that breaks the simulations after a certain amount of time. This requires studies that will be incorporated in a future release. 

#### Rework of metabolism

* All the `magic numbers` that were used as glue among the various simulators have been moved to the configuration file (`simulation_config.json`) or removed. 
* Value checking before a metabolism time step can now be changed from the config file. For example you can now say that if a neuron has an ATP that is lower than 0.25 it should be kicked out of the simulation and the simulation needs to continue without it. More information on how to do it is available in the configuration documentation.
* The metabolism model and metabolism object are not connected with fixed numbers anymore. Thus, if you want to change the model just a change in the configuration file is required.

#### Rework of the connections among simulators

* Connections among simulators can now be changed programmatically from the configuration. Just after a simulator (called `destination simulator`) does an `advance` the connection manager loops over its connections, picks the `source simulator` and performs the connections. Currently there are 3 types of connections:
  * `sum` sums the values from the `source` and the `destination` in the destination simulator.
  * `set` sets the values from `source` into `destination`.
  * `merge` merges the two discording values $V_{\text{source}}$ and $V_{\text{destination}}$ using the formula: $V_{\text{source}} + V_{\text{destination}} - V_{n-1} = V_n$. $V_n$ is then set in both `source` and `destination`. This is the only connection type that also changes the `source`. Notice that swapping `source` and `destination` is not exactly equivalent because the sync is performed after a `destination` `advance`.
  
#### Rework of the reporting

* The reporting structure has been changed significantly (mostly simplified and improved). However, these changes are mostly under the hood except for the changes in the configuration file. Consider reading the appropriate section in the [documentation](https://bbpteam.epfl.ch/documentation/projects/multiscale_run) to know more MultiscaleRun reports.

### Minor Changes
  
* Added documentation for the configuration file.
* Added docstrings in the code.
* `bf` -> `bloodflow` keyword change for simpler future development [BBPP40-440].
* Added type suggestions for function signatures in the code.
* Formatted docstrings for `sphinx` documentation.
* Updated `postproc` to handle the new configuration.
* Reworked the Python `import` statements and the initialization of the MPI library to fix possible hanging issues when `MPI_Finalize` was called.
* Add static analysis of the code in the continuous-integration process.

### Bug Fixes

* Fix a bug in the pytest CI that made it pass even if some tests were failing
* Fix hanging simulations when an error is thrown [BBPP40-427] when MPI is used

## 0.6 - 2024-03-04

* virtualenv command:
  * Improvement: the command now installs spack pkg if necessary. There is no need to run spack commands manually anymore. [!104]
  * Fix issue when MultiscaleRun was loaded from BB5 module `py-multiscale-run`. [BBPP40-430]
* edit-mod-file command: load intel compilers module if required (!104)
* improved the documentation of the Python API

## 0.5.1 - 2024-02-07

* Fix: init with julia create was not pointing to the correct location

## 0.5 - 2024-01-29

* Improved README
* `msr_config.json` is a template now. Prepared for the bbp workflow
* `base_path` is now specified in the main config file. This is breaking change. Add to your `msr_config.json` file a field: `"base_path": "."`
* `ndam` is now forced to use `RoundRobin`

## 0.4 - 2024-01-19

* New commands:
  * `edit-mod-files`: Clone the Neurodamus mod files library for local editing
  * `virtualenv`: Create a Python virtual environment to ease development of MultiscaleRun Python package
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
    * `msr_version`: a string indicating the version of MultiscaleRun that created this config
* `compute` command: now creates a Neurodamus success file at the end of the simulation
* Compatibility notes for simulations created with previous versions of MultiscaleRun
  * rename directory `julia_environment` to `.julia_environment`
  * it is not possible to override the JSON configuration keys with environment variables.
* GitLab CI on BB5 now relies on spack

## 0.2 - 2023-12-14

* Rework reporting [BBPP40-402, BBPP40-407, BBPP40-410, BBPP40-411, BBPP40-415]

## 0.1 - 2023-11-30

First release of the code shipped as a Python package

## Authors

Polina Shichkova, Alessandro Cattabiani, Christos Kotsalos, and Tristan Carel

## Funding and Acknowledgments

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2023-2024 Blue Brain Project/EPFL
