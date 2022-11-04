#!/bin/bash

# Julia setup. It is supposed to be called before the simulations at least once.
# Constraints:
# - call it after python venv setup (julia needs the same location for the python interpreter).
# - call it after git cloning (julia module scrumbles certificates).

echo
echo "   ### julia"
echo

source $PYTHON_VENV_PATH/bin/activate

module load unstable julia

export JULIA_DEPOT_PATH=${PWD}/.julia
if [ -d ".julia" ]
then
  echo "julia packages already set"
else
  echo "setup julia"
  mkdir .julia
  julia -e 'using Pkg; Pkg.add("IJulia")'
  julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
  julia -e 'using Pkg; Pkg.add("DiffEqBase")'
  julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
  julia -e 'using Pkg; Pkg.add("StaticArrays")'
  julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
  julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'
fi

