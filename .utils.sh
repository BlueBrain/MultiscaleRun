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
  git status
  popd
fi
}


set_default () {
  var=$1
  if [[ -z "${!var}" ]]; then
    export $1=$2
  fi
}