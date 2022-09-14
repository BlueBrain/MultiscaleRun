#!/bin/bash

set -e

echo "! setup !"

module purge
module load unstable git python-dev py-neurodamus py-mpi4py julia
module load intel gcc hpe-mpi

echo "JULIA"

rm -rf ~/.julia
julia -e 'using Pkg; Pkg.add("IJulia")'
julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
julia -e 'using Pkg; Pkg.add("DiffEqBase")'
julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
julia -e 'using Pkg; Pkg.add("StaticArrays")'
julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'

echo "JULIA : pushd/popd"
pushd ~/.julia
popd

rm -rf ./python-venv
python -m venv ./python-venv
source ./python-venv/bin/activate

pip install psutil
pip install julia
pip install diffeqpy
pip install pympler
pip install h5py
