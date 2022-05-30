# multiscale_run

dualrun : Coupling Neuron & STEPS. The dualrun part of the script (STEPS3 version) is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
triplerun : Coupling dualrun & metabolism. The triplerun part of the script (STEPS3 version) is based on the one found in the **triplerun** folder of [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam).
quadrun : Coupling triplerun & blood flow. [**WIP**]

The BlueConfig & user.target are based on the ones found in [this repo](https://bbpgitlab.epfl.ch/hpc/sim/blueconfigs/-/tree/main/ngv-v6).
The mod folder is based on the one found in [this repo](https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/tree/main/triplerun/custom_ndam_special_v1/polina_mod) with the addition of the mod files found [here](https://bbpgitlab.epfl.ch/hpc/sim/models/common/-/tree/main/mod/ngv).

## Environment Setup

1. `salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 00:30:00 --cpus-per-task=2 --constraint=clx`
2. `source set_env.sh`
3. `exit`

## Run Simulation

1. **Environment Setup** first
2. `sbatch job_script`
