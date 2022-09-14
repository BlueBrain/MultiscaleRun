#!/bin/bash

set -e

module purge
module load unstable python-dev py-neurodamus hpe-mpi

echo "*******************************************************************************"
echo "STEPS_INSTALLED_HASH=${STEPS_INSTALLED_HASH}"
echo "NEURODAMUS_NEOCORTEX_INSTALLED_HASH=${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}"
echo "*******************************************************************************"
spack load /${STEPS_INSTALLED_HASH}
spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}

rm -rf ./python-venv
python -m venv ./python-venv
source ./python-venv/bin/activate
pip install psutil

export OMP_NUM_THREADS=1

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
srun --overlap -n 16 dplace special -mpi -python ${PYDRIVER}

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
pytest -v .ci/test_dualrun_results.py
