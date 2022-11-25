#!/bin/bash

# python venv setup. Necessary because there are a couple of modules that are not in spack.
# It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia needs the same location for the python interpreter).
# - after bloodflow: it needs to know from it what are the additional packages that we need to install
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
  python -m pip install --upgrade pip
fi

# install bloodflow-related packages
pushd $bloodflow_path
pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ -e .
popd