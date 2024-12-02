fatal_error() {
  RED='\033[0;31m'
  NC='\033[0m' # No Color
  >&2 printf "${RED}Error${NC}: $@\n${RED}Abort.${NC}\n"
  exit 1
}

set_test_environment() {
  if [ -d /gpfs/bbp.cscs.ch ] ;then
    module load unstable intel-oneapi-mkl py-pytest
    export JULIA_DEPOT_PATH=/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment/latest/julia
    export JULIA_PROJECT=/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment/latest/julia_environment
    if ! [ -e "$JULIA_DEPOT_PATH" ] ; then
      fatal_error "Julia depot does not exist: '$JULIA_DEPOT_PATH'"
    fi
  fi
  MPIRUN=${MPIRUN:-"srun --overlap"}
}

download_tiny_CI_neurodamus_data() {
    local url="https://github.com/BlueBrain/MultiscaleRun/releases/download/0.8.2/tiny_CI_neurodamus_release-v0.8.2.tar.gz"
    local filename="tiny_CI_neurodamus_release-v0.8.2.tar.gz"

    # Download the file
    wget -q "$url" -O "$filename"

    # Check if download was successful
    if [[ $? -eq 0 ]]; then
        # Extract the tar.gz file
        tar -xzf "$filename"
        echo "Download and extraction completed."
    else
        echo "Download failed."
    fi
}
