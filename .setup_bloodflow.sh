#!/bin/bash

# Bloodflow setup. It is supposed to be called before the simulations at least once.
# Constraints:
# - call it before julia setup (julia module scrumbles certificates).
echo
echo "   ### bloodflow"
echo

export BLOODFLOW_PATH=${PWD}/bloodflow
lazy_clone bloodflow git@bbpgitlab.epfl.ch:molsys/bloodflow.git main