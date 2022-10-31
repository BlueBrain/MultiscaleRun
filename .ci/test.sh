#!/bin/bash

set -e

if [[ $nrun -eq 2 ]]
then
  sed -i 's/StimulusInject pInj/#StimulusInject pInj/g' $blueconfig_path
fi

run_steps () {
  RESULTS_PATH=./RESULTS/STEPS$1/
  echo "*******************************************************************************"
  echo " *** STEPS$1 run *** "
  echo "*******************************************************************************"
  srun --overlap -n $bb5_ntasks dplace special -mpi -python main.py $nrun $1 $RESULTS_PATH $blueconfig_path $mesh_path

  echo "*******************************************************************************"
  echo " *** Jupyter notebook *** "
  echo "*******************************************************************************"
  module load py-notebook

  # execute the jupyter notebook and save the output as html file
  jupyter-nbconvert \
  --execute \
  --to html \
  --no-input \
  --output-dir=$RESULTS_PATH \
  notebook.ipynb
}

run_steps 3
run_steps 4

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
if [[ $nrun -eq 2 ]]
then
  pytest -v .ci/test_dualrun_results.py
elif [[ $nrun -eq 3 ]]
then
  pytest -v .ci/test_triplerun_results.py
fi
