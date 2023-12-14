#!/bin/bash

# This should be called by setup.sh / quick_setup.sh

echo
echo "   ### setup env"
echo

export HTTP_PROXY="http://bbpproxy.epfl.ch:80/"
export HTTPS_PROXY="http://bbpproxy.epfl.ch:80/"
export http_proxy="http://bbpproxy.epfl.ch:80/"
export https_proxy="http://bbpproxy.epfl.ch:80/"

source .utils.sh

set_default BLOODFLOW_BRANCH main
set_default UPDATE_BLOODFLOW 1

set_default PY_NEURODAMUS_BRANCH main
set_default NEURODAMUS_NEOCORTEX_BRANCH main
set_default UPDATE_PY_NEURODAMUS 1
set_default PY_NEURODAMUS_USE_MODULE 0
set_default NEURODAMUS_NEOCORTEX_USE_MODULE 0

set_default STEPS_BRANCH master
set_default UPDATE_STEPS 1
set_default STEPS_USE_MODULE 1

set_default SPACKENV_PATH ${PWD}/spackenv
set_default PYTHON_VENV_PATH ${PWD}/python-venv
set_default JULIA_DEPOT_PATH ${PWD}/.julia
set_default JULIA_PROJECT ${PWD}/julia_environment
set_default BLOODFLOW_PATH ${PWD}/bloodflow_src


# Our approach is to map 1 core per MPI task.
# Therefore, this strategy is not compatible with multi-threading.
set_default OMP_NUM_THREADS 1
export JULIA_NUM_THREADS=1