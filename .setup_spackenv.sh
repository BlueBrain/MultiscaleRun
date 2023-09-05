#!/bin/bash

# Spack env setup. It is supposed to be called before the simulations at least once. With the CI we already have the
# hashes, we just load them. Standalone we add steps and neurodamus to the environment itself
# Constraints:
# - call it after steps setup
# - call it after neurodamus neocortex setup

echo
echo "   ### spack env and additional repos"
echo

module load unstable
module load gcc 
module load intel-oneapi-compilers 
module load hpe-mpi

if [[ -n "${CI}" ]]
then
  module load git
fi

if [ -d "spackenv" ]
then
  echo "spackenv already set"
  spack env activate -d spackenv
else
  echo "create spackenv"
  spack env create -d spackenv
  spack env activate -d spackenv
  spack config add concretizer:unify:true
fi
export SPACKENV_PATH=${PWD}/spackenv

if [[ -z "${CI}" ]]
then
  if [ "${PY_NEURODAMUS_USE_MODULE}" -eq 1 ]
  then
    echo "py-neurodamus from module"
    module load py-neurodamus
  else
    echo "add custom py-neurodamus"

    lazy_clone py-neurodamus git@bbpgitlab.epfl.ch:hpc/sim/neurodamus-py.git $PY_NEURODAMUS_BRANCH $UPDATE_NEURODAMUS
    # hack to remove the links to proj12 for the people that do not have access. Discussion in [BBPBGLIB-973]
    rm py-neurodamus/tests/simulations/v5_gapjunctions/gap_junctions
    spack add py-neurodamus@develop
    spack develop -p ${PWD}/py-neurodamus --no-clone py-neurodamus@develop
  fi

  echo "add ndam neocortex"
  spack add neurodamus-neocortex@develop+ngv+metabolism
fi


if [ "${NEURODAMUS_NEOCORTEX_USE_MODULE}" -eq 1 ]
then
  echo "neurodamus neocortex from module"
  module load neurodamus-neocortex-multiscale
else
  if [[ -n "${CI}" ]]
  then
    echo "neurodamus neocortex from CI"
    spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
  else
    if [ "${PY_NEURODAMUS_USE_MODULE}" -eq 1 ]
    then
      echo "py-neurodamus from module"
      module load py-neurodamus
    else
      echo "add custom py-neurodamus"
      lazy_clone py-neurodamus git@bbpgitlab.epfl.ch:hpc/sim/neurodamus-py.git $PY_NEURODAMUS_BRANCH $UPDATE_NEURODAMUS
      # hack to remove the links to proj12 for the people that do not have access. Discussion in [BBPBGLIB-973]
      rm py-neurodamus/tests/simulations/v5_gapjunctions/gap_junctions
      spack add py-neurodamus@develop
      spack develop -p ${PWD}/py-neurodamus --no-clone py-neurodamus@develop
    fi

    echo "add ndam neocortex"
    spack add neurodamus-neocortex@develop+ngv+metabolism
  fi
fi

if [ "${STEPS_USE_MODULE}" -eq 1 ]
then
  echo "steps from module"
  module load steps/5.0.0a
else
  if [[ -n "${CI}" ]]
  then
    echo "steps from CI"
    spack load /${STEPS_INSTALLED_HASH}
  else
    echo "custom build steps"
    lazy_clone HBP_STEPS git@github.com:CNS-OIST/HBP_STEPS.git $STEPS_BRANCH $UPDATE_STEPS
    spack add steps@develop ^petsc+int64+mpi
    spack develop -p ${PWD}/HBP_STEPS --no-clone steps@develop
  fi
fi

echo "additional software"
# TODO reactivate this after https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-1039 is solved
spack add py-psutil py-bluepysnap ^neuron py-pytest py-jupyterlab


spack install
spack env deactivate
spack env activate -d spackenv
