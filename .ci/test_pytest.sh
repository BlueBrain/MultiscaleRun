#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh

set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

download_tiny_CI_neurodamus_data
ln -s "$(pwd)/tiny_CI_neurodamus" "$(pwd)/multiscale_run/templates/tiny_CI"

num_errors=0
count_errors() {
    local command="$BASH_COMMAND"
    ((num_errors++))
    echo "Error occurred while executing: $command (Trap: ERR)"
    echo "num_errors: $num_errors"
}
trap count_errors ERR

if [ $# -eq 0 ] ;then
    python -mpytest tests/pytests
    $MPIRUN -n 4 python -mpytest -v tests/pytests/test_reporter.py
else
    python -mpytest $@
fi

popd >/dev/null
exit $num_errors
