#!/bin/bash

set -e
# There are a lot of things to load. Better have a blueprint for that

echo
echo "   ### setup quick"
echo "   This is supposed to be used after a recent run of setup.sh (or in CI). If you ran setup.sh today it is probably fine."
echo


. ${SPACK_ROOT}/share/spack/setup-env.sh

source setup_env.sh

module load unstable
module load gcc 
module load intel-oneapi-compilers 
module load hpe-mpi

module load julia petsc py-petsc4py gmsh

if [ "${NEURODAMUS_NEOCORTEX_USE_MODULE}" -eq 1 ]; then
  echo "neurodamus neocortex from module"
  module load neurodamus-neocortex-multiscale
elif [ -n "${CI}" ]; then
  spack load "/${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}"
fi

if [ "${STEPS_USE_MODULE}" -eq 1 ]; then
  echo "steps from module"
  module load steps/5.0.0b
elif [ -n "${CI}" ]; then
  echo "steps from CI"
  spack load "/${STEPS_INSTALLED_HASH}"
fi

spack env activate -d $SPACKENV_PATH
source $PYTHON_VENV_PATH/bin/activate




