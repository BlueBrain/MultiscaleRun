#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh
set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

$MPIRUN -n 2 python tests/integration_tests/test_connect_neurodamus2steps.py
$MPIRUN -n 2 python tests/integration_tests/test_autogen_mesh.py
$MPIRUN -n 2 python tests/integration_tests/test_preprocessor.py

popd > /dev/null
