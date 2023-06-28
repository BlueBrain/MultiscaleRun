#!/bin/bash

# Bloodflow setup. It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia module scrumbles certificates).
echo
echo "   ### bloodflow"
echo

# it is important to not name this "bloodflow" otherwise python messes up the imports
folder=bloodflow_src
export BLOODFLOW_PATH=${PWD}/$folder
lazy_clone $folder https://github.com/BlueBrain/AstroVascPy.git $BLOODFLOW_BRANCH $UPDATE_BLOODFLOW
