#!/bin/bash

if [[ $nrun -eq 2 ]]
then
  sed -i 's/StimulusInject pInj/#StimulusInject pInj/g' $blueconfig_path
fi

source .utils.sh

ms_run 3
ms_run 4

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
if [[ $nrun -eq 2 ]]
then
  pytest -v .ci/test_dualrun_results.py
elif [[ $nrun -eq 3 ]]
then
  pytest -v .ci/test_triplerun_results.py
elif [[ $nrun -eq 4 ]]
then
  pytest -v .ci/test_triplerun_results.py
fi
