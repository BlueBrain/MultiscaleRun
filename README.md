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

## List of outdated mod files

So far all the mod files found in [neurodamus-core](https://bbpgitlab.epfl.ch/hpc/sim/neurodamus-core/-/tree/main/mod) & [common](https://bbpgitlab.epfl.ch/hpc/sim/models/common/-/tree/main/mod/ngv) are the latest. These mod files end up in the **mod folder**.

How to generate the list of outdated mod files:

1. `cp metabolismndam/custom_ndam_2021_02_22_archive202101/mod/* mod/`
1. ```for f in `ls -1 neurodamus-core/mod`; do rm mod/$f; done;```, i.e. remove from **mod folder** all the mod files that are already in **neurodamus-core**.

```
ampa.mod
Ca_HVA.mod
Ca_LVAst.mod
DetAMPANMDA.mod
DetGABAAB.mod
gap.mod
GluSynapse.mod
Ih.mod
Im.mod
internal_ions.mod
IonSynapse.mod
kcc2.mod
KdShu2007.mod
K_Pst.mod
K_Tst.mod
leak.mod
naclamp.mod
nakcc.mod
nakpump.mod
Nap_Et2_ionic.mod
NaTa_t_ionic.mod
NaTg.mod
NaTs2_t_ionic.mod
ProbAMPANMDA_EMS.mod
ProbGABAAB_EMS.mod
SK_E2.mod
SKv3_1.mod
StochKv3.mod
TTXDynamicsSwitch.mod
```

The latest mod files can be found in the installation folder of neurodamus, following these steps:

1. ``` ndam_installation_dir=`spack find --paths neurodamus-neocortex+ngv | tail -n 1 | grep -o "/.*"` ```
1. latest mod files are located in `$ndam_installation_dir/share/mod_full`
1. List of mod files that are not in the **mod_full**:
    ```
    ampa.mod
    internal_ions.mod
    IonSynapse.mod
    kcc2.mod
    leak.mod
    naclamp.mod
    nakcc.mod
    nakpump.mod
    Nap_Et2_ionic.mod
    NaTa_t_ionic.mod
    NaTs2_t_ionic.mod
    ```
    These mod files should be eventually added in neurodamus+ngv related mod folder.
