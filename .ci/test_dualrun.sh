#!/bin/bash

set -e

echo "! setup !"

module purge
module load unstable git python-dev py-neurodamus py-mpi4py
module load intel gcc hpe-mpi

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

# STEPS related
export OMP_NUM_THREADS=1

echo "! setup completed !"

# Remove Stimulus Inject from BC (else sum of currents != 0)
sed -i 's/StimulusInject pInj/#StimulusInject pInj/g' BlueConfigCI

export which_STEPS=3
PYDRIVER=multiscale_run_STEPS${which_STEPS}.py
export which_mesh=mc2c
export which_BlueConfig=BlueConfigCI
# Like in C, the integers 0 (false) and 1 (true—in fact any nonzero integer) are used.
export dualrun=1
export triplerun=0
echo "*******************************************************************************"
echo " *** STEPS${which_STEPS} run *** "
echo "*******************************************************************************"
srun --overlap -n $bb5_ntasks dplace special -mpi -python ${PYDRIVER}

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
pytest -v .ci/test_dualrun_results.py
