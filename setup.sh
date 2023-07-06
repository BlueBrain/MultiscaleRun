#!/bin/bash

if [[ $IS_SPACK_UPDATED -eq 0 ]]
then
while true; do

read -p "Is your spack updated? (y/n) " yn

case $yn in 
	[yY] ) echo ok, we will proceed;
		break;;
	[nN] ) echo exiting...;
		return;;
	* ) echo invalid response;;
esac
done
fi

echo
echo "   ### setup"
echo "   It is suggested to allocate a node if it is the first time you run this script!"
echo

export HTTP_PROXY="http://bbpproxy.epfl.ch:80/"
export HTTPS_PROXY="http://bbpproxy.epfl.ch:80/"
export http_proxy="http://bbpproxy.epfl.ch:80/"
export https_proxy="http://bbpproxy.epfl.ch:80/"

module purge
git pull

source .utils.sh

set_default BLOODFLOW_BRANCH main
set_default UPDATE_BLOODFLOW 1

set_default PY_NEURODAMUS_BRANCH main
set_default  UPDATE_NEURODAMUS 1

# this is completely handled by spack. We keep it updated for now
set_default NEURODAMUS_NEOCORTEX_BRANCH main

set_default STEPS_BRANCH master
set_default UPDATE_STEPS 1
set_default STEPS_USE_MODULE 1

source .setup_bloodflow.sh
source .setup_spackenv.sh
source .setup_python_venv.sh
source .setup_julia.sh

# Our approach is to map 1 core per MPI task.
# Therefore, this strategy is not compatible with multi-threading.
set_default OMP_NUM_THREADS 1
export JULIA_NUM_THREADS=1

