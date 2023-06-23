#!/bin/bash

set -e
# There are a lot of things to load. Better have a blueprint for that

. ${SPACK_ROOT}/share/spack/setup-env.sh

module load unstable
module load gcc 
module load intel-oneapi-compilers 
module load hpe-mpi

module load julia petsc-complex py-petsc4py-complex

spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
if [ "${STEPS_USE_MODULE}" -eq 1 ]
then
  echo "steps from module"
  module load steps-complex/5.0.0a
else
  echo "steps from CI"
  spack load /${STEPS_INSTALLED_HASH}
fi
spack env activate -d $SPACKENV_PATH
source $PYTHON_VENV_PATH/bin/activate

source .utils.sh

# Our approach is to map 1 core per MPI task.
# Therefore, this strategy is not compatible with multi-threading.
set_default OMP_NUM_THREADS 1
export JULIA_NUM_THREADS=1
