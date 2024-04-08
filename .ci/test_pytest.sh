#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh

set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

num_errors=0
count_errors() {
    ((num_errors++))
    local command="$BASH_COMMAND"
    echo "Error occurred while executing: $command (Trap: ERR)"
    echo "num_errors: $num_errors"
}
trap count_errors ERR

python -mpytest tests/pytests
$MPIRUN -n 4 python -mpytest -v tests/pytests/test_reporter.py

popd >/dev/null
exit $num_errors
