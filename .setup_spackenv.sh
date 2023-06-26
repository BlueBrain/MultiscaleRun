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
  echo "ndam neocortex from CI"
  spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
  

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
  echo "add ndam py"

  lazy_clone py-neurodamus git@bbpgitlab.epfl.ch:hpc/sim/neurodamus-py.git $PY_NEURODAMUS_BRANCH $UPDATE_NEURODAMUS
  # hack to remove the links to proj12 for the people that do not have access. Discussion in [BBPBGLIB-973]
  rm py-neurodamus/tests/simulations/v5_gapjunctions/gap_junctions
  spack add py-neurodamus@develop
  spack develop -p ${PWD}/py-neurodamus --no-clone py-neurodamus@develop

  echo "add ndam neocortex"
  # TODO reactivate this after https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-1039 is solved
  # spack add neurodamus-neocortex@develop+ngv+metabolism
  spack add neurodamus-neocortex@develop+ngv+metabolism%intel ^neuron%intel
  


fi

if [ "${STEPS_USE_MODULE}" -eq 1 ]
then
  echo "steps from module"
  module load steps-complex/5.0.0a
else
  if [[ -n "${CI}" ]]
  then
    echo "steps from CI"
    spack load /${STEPS_INSTALLED_HASH}
  else
    echo "custom build steps"
    lazy_clone HBP_STEPS git@github.com:CNS-OIST/HBP_STEPS.git $STEPS_BRANCH $UPDATE_STEPS
    spack add steps@develop ^petsc+complex+int64+mpi
    spack develop -p ${PWD}/HBP_STEPS --no-clone steps@develop
  fi
fi

echo "additional software"
# TODO reactivate this after https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-1039 is solved
spack add py-psutil py-bluepysnap ^neuron%intel py-pytest py-jupyterlab


spack install
spack env deactivate
spack env activate -d spackenv
