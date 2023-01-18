echo
echo "   ### setup"
echo "   It is suggested to allocate a node if it is the first time you run this script!"
echo

module purge
git pull

source .utils.sh

export BLOODFLOW_BRANCH=main
export UPDATE_BLOODFLOW=1

export PY_NEURODAMUS_BRANCH=main
export UPDATE_NEURODAMUS=1

# this is completely handled by spack. We keep it updated for now
export NEURODAMUS_NEOCORTEX_BRANCH=main

export STEPS_BRANCH=master
export UPDATE_STEPS=1

source .setup_bloodflow.sh
source .setup_spackenv.sh
source .setup_python_venv.sh
source .setup_julia.sh

export OMP_NUM_THREADS=1

