#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source ${SCRIPT_DIR}/setup.sh

set_test_environment

#python -mpytest ${SPACK_SOURCE_DIR}/tests/pytests hangs!
# related issue in: https://bbpteam.epfl.ch/project/issues/browse/BBPP40-412
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_bloodflow.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_cli.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_metabolism.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_neurodamus.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_steps.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_utils.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_config.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_preprocessor.py
python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_reporter.py
srun --overlap -n 4 python -mpytest ${SCRIPT_DIR}/../tests/pytests/test_reporter.py
