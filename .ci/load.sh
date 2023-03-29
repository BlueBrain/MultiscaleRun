#!/bin/bash

set -e
# There are a lot of things to load. Better have a blueprint for that

. ${SPACK_ROOT}/share/spack/setup-env.sh
module load unstable intel gcc hpe-mpi julia
module load petsc-complex
module load py-petsc4py-complex
module load steps-complex/5.0.0a
spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
spack env activate -d $SPACKENV_PATH
source $PYTHON_VENV_PATH/bin/activate
