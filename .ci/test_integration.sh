#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh
set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

download_tiny_CI_neurodamus_data


# Commit hash stored in PY_MULTISCALE_RUN_COMMIT
commit_hash=$PY_MULTISCALE_RUN_COMMIT
MULTISCALE_RUN_PATH=""
# Loop through the directories in PYTHONPATH and find the one containing the commit hash
for path in $(echo $PYTHON_PATH | tr ':' '\n'); do
    if [[ "$path" == *"$PY_MULTISCALE_RUN_COMMIT"* ]]; then
        MULTISCALE_RUN_PATH=$path
        break
    fi
done
ln -s "$(pwd)/tiny_CI_neurodamus" "$MULTISCALE_RUN_PATH/multiscale_run/templates/tiny_CI"

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
