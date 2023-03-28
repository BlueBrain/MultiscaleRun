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

  # Blood Flow Solver-related
  # deactivate pip install petsc & petsc4py, use module instead!
  sed -i 's/"petsc"/#"petsc"/g' $BLOODFLOW_PATH/setup.py
  sed -i 's/"petsc4py"/#"petsc4py"/g' $BLOODFLOW_PATH/setup.py
fi

pip install diffeqpy julia

# install blood flow solver
echo "   ### python-venv : Blood Flow Solver setup"
pushd $BLOODFLOW_PATH

module load petsc-complex
module load py-petsc4py-complex

# Run it every time to account for any changes we introduce to the solver
pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ -e .

# Backend solver/library for the linear systems [BFS : Blood Flow Solver]
# petsc or scipy
export BACKEND_SOLVER_BFS='petsc'

# Run the SciPy solver and compare the result with the PETSc one [which is the default]!
# 0 : False / 1 : True
export DEBUG_BFS=0

# Show PETSc progress or not
# 0 : False / 1 : True
export VERBOSE_BFS=0

popd
