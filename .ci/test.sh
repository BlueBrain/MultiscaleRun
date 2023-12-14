#!/bin/bash

set -e

echo "*******************************************************************************"
echo " *** main.py run *** "
echo "*******************************************************************************"
export results_path=RESULTS
srun --overlap -n $bb5_ntasks python main.py

if [[ $with_postproc -ne 1 ]]; then
    echo "no postproc required. with_postproc=$with_postproc"
    return
fi

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


if [[ "$with_steps" -eq 1 && "$with_metabolism" -eq 1 ]]; then
    echo "Running test_triplerun"
    python -mpytest -v tests/test_triplerun.py
fi