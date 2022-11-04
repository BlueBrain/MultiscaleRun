#!/bin/bash

# python venv setup. Necessary because there are a couple of modules that are not in spack.
# It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia needs the same location for the python interpreter).
echo
echo "   ### python-venv"
echo

module load unstable python julia

export PYTHON_VENV_PATH=${PWD}/python-venv
if [ -d "python-venv" ]
then
  echo "python-venv already set"
  source ./python-venv/bin/activate
else
  python -m venv python-venv
  source python-venv/bin/activate
  pip install diffeqpy julia
fi