#!/bin/bash

set -e

export results_path=./RESULTS/
echo "*******************************************************************************"
echo " *** main.py run *** "
echo "*******************************************************************************"
srun --overlap -n $bb5_ntasks python main.py

echo "*******************************************************************************"
echo " *** Jupyter notebook *** "
echo "*******************************************************************************"
module load py-notebook

# execute the jupyter notebook and save the output as html file
jupyter-nbconvert \
--execute \
--to html \
--no-input \
--output-dir=$results_path \
postproc.ipynb

echo "*******************************************************************************"
echo " *** pytest *** "
echo "*******************************************************************************"
if [[ $nrun -eq 2 ]]
then
  pytest -mpytest -v tests/test_dualrun.py
elif [[ $nrun -eq 3 ]]
then
  python -mpytest -v tests/test_triplerun.py
elif [[ $nrun -eq 4 ]]
then
  pytest -mpytest -v tests/test_triplerun.py
fi
