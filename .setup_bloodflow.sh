#!/bin/bash

# Bloodflow setup. It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia module scrumbles certificates).
echo
echo "   ### bloodflow"
echo

source setup_env.sh

# it is important to not name this "bloodflow" otherwise python messes up the imports
export BLOODFLOW_PATH=${PWD}/bloodflow_src
lazy_clone $BLOODFLOW_PATH https://github.com/BlueBrain/AstroVascPy.git $BLOODFLOW_BRANCH $UPDATE_BLOODFLOW
