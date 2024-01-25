#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh
set_test_environment

pushd "${SCRIPT_DIR}/.."

srun --overlap -n 2 python tests/integration_tests/test_connect_ndam2steps.py
srun --overlap -n 2 python tests/integration_tests/test_ndam2steps_sync.py
srun --overlap -n 2 python tests/integration_tests/test_autogen_mesh.py
srun --overlap -n 2 python tests/integration_tests/test_preprocessor.py

popd
