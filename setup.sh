echo
echo "   ### setup"
echo "   It is suggested to allocate a node if it is the first time you run this script!"
echo

module purge

source .utils.sh

source .setup_bloodflow.sh
source .setup_spackenv.sh
source .setup_python_venv.sh
source .setup_julia.sh

export OMP_NUM_THREADS=1

