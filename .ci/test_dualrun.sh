#!/bin/bash

set -e

echo "! setup !"

module purge
module load unstable git python-dev py-neurodamus py-mpi4py
module load intel gcc hpe-mpi

echo "Cloning various repos."
echo "metabolismndam"
rm -rf metabolismndam/
git clone --quiet -b main --single-branch https://gitlab-ci-token:${CI_JOB_TOKEN}@bbpgitlab.epfl.ch/molsys/metabolismndam.git
echo "neurodamus-core"
rm -rf neurodamus-core/
git clone --quiet -b main --single-branch https://gitlab-ci-token:${CI_JOB_TOKEN}@bbpgitlab.epfl.ch/hpc/sim/neurodamus-core.git
echo "common"
rm -rf common/
git clone --quiet -b main --single-branch https://gitlab-ci-token:${CI_JOB_TOKEN}@bbpgitlab.epfl.ch/hpc/sim/models/common.git

echo "*******************************************************************************"
echo "STEPS_INSTALLED_HASH=${STEPS_INSTALLED_HASH}"
echo "NEURODAMUS_NEOCORTEX_INSTALLED_HASH=${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}"
echo "*******************************************************************************"
echo "Just loading already installed versions of steps & neurodamus."
spack load /${STEPS_INSTALLED_HASH}
spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}

echo "Setting up Python env."
rm -rf ./python-venv
python -m venv ./python-venv
source ./python-venv/bin/activate

pip install psutil

echo "[...] building custom special."
rm -rf x86_64

# legacy mod files for triplerun
cp metabolismndam/custom_ndam_2021_02_22_archive202101/mod/* mod/

# update the legacy ones with mod files from neurodamus-core
cp neurodamus-core/mod/* mod/
rm -rf neurodamus-core/

# additional mod files from common repo
cp -n common/mod/ngv/* mod/
rm -rf common/

build_neurodamus.sh mod
echo "[completed] building custom special."

# STEPS related
export OMP_NUM_THREADS=1

echo "! setup completed !"

export which_STEPS=3
PYDRIVER=multiscale_run_STEPS${which_STEPS}.py
export which_mesh=mc2c
export which_BlueConfig=BlueConfigCI
# Remove Stimulus Inject from BC (else sum of currents != 0)
sed -i 's/StimulusInject pInj/#StimulusInject pInj/g' $which_BlueConfig
# Like in C, the integers 0 (false) and 1 (true—in fact any nonzero integer) are used.
export dualrun=1
export triplerun=0
echo "*******************************************************************************"
echo " *** STEPS${which_STEPS} run *** "
echo "*******************************************************************************"
srun --overlap -n $bb5_ntasks dplace x86_64/special -mpi -python ${PYDRIVER}

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
pytest -v .ci/test_dualrun_results.py
