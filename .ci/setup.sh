#!/bin/bash

set -e
# blueprint for a setup step of the CI. Since we install stuff in the base pipeline directory, it is a little different
# from the general setup for running the program manually. For the sake of clarity, all of this is hidden in the .ci folder

# This works only in the CI

. ${SPACK_ROOT}/share/spack/setup-env.sh
source .utils.sh

export STARTING_DIR=${PWD}
echo "STARTING_DIR=${STARTING_DIR}"
ls -la

source $setup_file

echo "Put envvars variables in " $envfile.env
# this is a trick because gitlab ci does not support array variables
eval $envvars

for str in ${envvars[@]}; do
  eval temp=\$$str
  echo ${str}=${temp} >> ${envfile}.env
  echo ${str}=${temp}
done
echo "Done"

