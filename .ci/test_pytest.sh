#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh

set_test_environment

pushd "${SCRIPT_DIR}/.."

wget -q https://github.com/BlueBrain/MultiscaleRun/releases/download/0.8.2/tiny_CI_neurodamus_release-v0.8.2.tar.gz
tar -xzf tiny_CI_neurodamus_release-v0.8.2.tar.gz

echo "multiscale run location: $(pwd)"

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
