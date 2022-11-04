#!/bin/bash

# Collection of general utils. Source me if you want!

# Clone if there is no folder, pull otherwise
# inputs: folder, repo, branch
lazy_clone () {
folder=$1
repo=$2
branch=$3
if [ -d "$folder" ]
then
  echo "$folder already set. Just pulling latest changes"
  pushd $folder
  git pull
  git submodule update
  popd
else
  echo "clone $folder"
  git clone -b $branch --single-branch --recursive $repo $folder
fi

}

# run and record results (mainly for the CI)
ms_run () {
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