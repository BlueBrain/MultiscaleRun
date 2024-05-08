#!/bin/bash -e

echo "pwd: ${PWD}"

module load unstable intel-oneapi-mkl llvm py-pytest

multiscale-run init --help
sim_name=ratV6_local_julia
rm -rf $sim_name
multiscale-run init --julia create $sim_name
sim_name=ratV6_no_julia

rm -rf ratV6_local_julia
