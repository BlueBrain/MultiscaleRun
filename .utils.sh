#!/bin/bash

# Collection of general utils. Source me if you want!

# Clone if there is no folder, pull otherwise
# inputs: folder, repo, branch
lazy_clone () {
folder=$1
repo=$2
branch=$3
update_branch=$4
if [ -d "$folder" ]
then
  echo "$folder already set"
  if [ $update_branch -eq 0 ]
  then
    echo "keeping it as is"
  else
    echo "updating to latest version of branch: $branch"
    pushd $folder
    git checkout $branch
    git pull
    git submodule update
    popd
  fi
else
  echo "clone $folder and branch: $branch"
  git clone --recursive $repo $folder
  pushd $folder
  git checkout $branch
  popd
fi
}

# run and record results (mainly for the CI)
ms_run () {
  results_path=./RESULTS/STEPS$1/
  echo "*******************************************************************************"
  echo " *** STEPS$1 run *** "
  echo "*******************************************************************************"
  srun --overlap -n $bb5_ntasks dplace special -mpi -python main.py

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
}