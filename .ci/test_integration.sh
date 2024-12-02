#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh
set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

download_tiny_CI_neurodamus_data
MULTISCALE_RUN_PATH00=$(pwd | sed -E 's|(/gpfs/bbp.cscs.ch/ssd/gitlab_map_jobs/bbpcimolsys/)[^/]+(/spack-build/spack-stage-)|\1\2|; s|/spack-src/|/lib/python3.11/site-packages/|')
ln -s "$(pwd)/tiny_CI_neurodamus" "$MULTISCALE_RUN_PATH00/templates/tiny_CI"

num_errors=0
count_errors() {
    ((num_errors++))
    local command="$BASH_COMMAND"
    echo "Error occurred while executing: $command (Trap: ERR)"
    echo "num_errors: $num_errors"
}
trap count_errors ERR

time $MPIRUN -n 2 python tests/integration/connect_neurodamus2steps.py
time $MPIRUN -n 2 python tests/integration/autogen_mesh.py
time $MPIRUN -n 2 python tests/integration/preprocessor.py

popd > /dev/null
exit $num_errors
