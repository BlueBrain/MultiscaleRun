#!/bin/bash

# python venv setup. Necessary because there are a couple of modules that are not in spack.
# It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia needs the same location for the python interpreter).
# - after bloodflow: it needs to know from it what are the additional packages that we need to install
echo
echo "   ### python-venv"
echo

module load unstable python julia gcc hpe-mpi

export PYTHON_VENV_PATH=${PWD}/python-venv
if [ -d "python-venv" ]
then
  echo "python-venv already set"
  source ./python-venv/bin/activate
else
  python -m venv python-venv
  source python-venv/bin/activate
  python -m pip install --upgrade pip
  # patchelf to fix conflict between different PETSc-libs (STEPS & Blood Flow Solver)
  pip install patchelf
fi

pip install diffeqpy julia

# install blood flow solver
echo "   ### python-venv : Blood Flow Solver setup"
pushd $BLOODFLOW_PATH

export ARCHFLAGS="-arch "`uname -m`
export PETSC_CONFIGURE_OPTIONS='--with-scalar-type=complex'

# Run it every time to account for any changes we introduce to the solver
pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ -e .

# Backend solver/library for the linear systems [BFS : Blood Flow Solver]
# petsc or scipy
export BACKEND_SOLVER_BFS='scipy'

# Run the SciPy solver and compare the result with the PETSc one [which is the default]!
# 0 : False / 1 : True
export DEBUG_BFS=0

# Show PETSc progress or not
# 0 : False / 1 : True
export VERBOSE_BFS=0

popd

# The lines below solve the conflict between the PETSc-lib from STEPS (real number support) & 
# the PETSc-lib from the Blood Flow solver (complex number support).
# patchelf makes sure that petsc4py-lib points to the right PETSc-lib!

pushd `pip show petsc4py | grep Location: | grep -o "/.*"`/petsc4py/lib

# E.g. 3.18.5 -> pip_PETSC_MAJOR_VERSION=3 & pip_PETSC_MINOR_VERSION=18 & pip_PETSC_PATCH_VERSION=5
export pip_PETSC_MAJOR_VERSION=`pip show petsc | grep Version: | cut -d' ' -f2 | cut -d'.' -f1`
export pip_PETSC_MINOR_VERSION=`pip show petsc | grep Version: | cut -d' ' -f2 | cut -d'.' -f2`
export pip_PETSC_PATCH_VERSION=`pip show petsc | grep Version: | cut -d' ' -f2 | cut -d'.' -f3`

patchelf --replace-needed \
libpetsc.so.$pip_PETSC_MAJOR_VERSION.$pip_PETSC_MINOR_VERSION \
`pip show petsc | grep Location: | grep -o "/.*"`/petsc/lib/libpetsc.so.$pip_PETSC_MAJOR_VERSION.$pip_PETSC_MINOR_VERSION.$pip_PETSC_PATCH_VERSION \
PETSc.cpython-*.so

popd
