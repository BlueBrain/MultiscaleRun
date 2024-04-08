#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh
set_test_environment

pushd "${SCRIPT_DIR}/.." >/dev/null

time $MPIRUN -n 2 python tests/integration_tests/test_connect_neurodamus2steps.py
time $MPIRUN -n 2 python tests/integration_tests/test_autogen_mesh.py
time $MPIRUN -n 2 python tests/integration_tests/test_preprocessor.py

popd > /dev/null
