#!/bin/bash

# Spack env setup. It is supposed to be called before the simulations at least once. With the CI we already have the
# hashes, we just load them. Standalone we add steps and neurodamus to the environment itself
# Constraints:
# - call it after steps setup
# - call it after neurodamus neocortex setup

echo
echo "   ### spack env and additional repos"
echo

module load unstable intel-oneapi-compilers gcc hpe-mpi

if [[ -n "${CI}" ]]
then
  module load git
  echo "ndam from CI"
  spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
  echo "steps from CI"
  spack load /${STEPS_INSTALLED_HASH}
fi

if [ -d "spackenv" ]
then
  echo "spackenv already set"
else
  spack env create -d spackenv
  sed -i '6 i\  concretization: together' spackenv/spack.yaml
fi
export SPACKENV_PATH=${PWD}/spackenv
spack env activate -d spackenv

if [[ -z "${CI}" ]]
then
  echo "add ndam"
  spack add neurodamus-neocortex@develop+ngv+metabolism

  echo "add steps"
  if [[ -z "${STEPS_BRANCH}" ]]
  then
    export STEPS_BRANCH=master
  fi
  lazy_clone HBP_STEPS git@github.com:CNS-OIST/HBP_STEPS.git $STEPS_BRANCH
  spack add steps@develop
  spack develop -p ${PWD}/HBP_STEPS --no-clone steps@develop
fi

echo "additional software"
spack add py-psutil py-bluepysnap py-scipy py-neurodamus py-pytest

spack install
spack env deactivate
spack env activate -d spackenv