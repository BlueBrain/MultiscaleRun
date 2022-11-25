#!/bin/bash

# Spack env setup. It is supposed to be called before the simulations at least once. With the CI we already have the
# hashes, we just load them. Standalone we add steps and neurodamus to the environment itself
# Constraints:
# - call it after steps setup
# - call it after neurodamus neocortex setup

echo
echo "   ### spack env and additional repos"
echo

# Pramod's suggestion: `gcc` before `intel-oneapi-compilers`
module load unstable gcc intel-oneapi-compilers hpe-mpi

if [[ -n "${CI}" ]]
then
  module load git
  echo "ndam py from CI"
  spack load /${PY_NEURODAMUS_INSTALLED_HASH}
  echo "ndam neocortex from CI"
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
  echo "add ndam py"
  if [[ -z "${PY_NEURODAMUS_BRANCH}" ]]
  then
    export PY_NEURODAMUS_BRANCH=katta/init_vasccouplingB
  fi
  lazy_clone py-neurodamus git@bbpgitlab.epfl.ch:hpc/sim/neurodamus-py.git $PY_NEURODAMUS_BRANCH
  spack add py-neurodamus@develop
  spack develop -p ${PWD}/py-neurodamus --no-clone py-neurodamus@develop

  echo "add ndam neocortex"
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
spack add py-psutil py-bluepysnap py-scipy py-pytest

spack install
spack env deactivate
spack env activate -d spackenv