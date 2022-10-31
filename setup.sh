echo
echo "   ### setup"
echo "   It is suggested to allocate a node if it is the first time you run this script!"
echo

module purge
module load unstable intel gcc hpe-mpi

if [[ -n "${CI}" ]]
then
  module load git
  echo "ndam from CI"
  spack load /${NEURODAMUS_NEOCORTEX_INSTALLED_HASH}
  echo "steps from CI"
  spack load /${STEPS_INSTALLED_HASH}
fi

echo
echo "   ### spack env"
echo
if [ -d "spackenv" ]
then
  echo "spackenv already set"
else
  spack env create -d spackenv
  sed -i '6 i\  concretization: together' spackenv/spack.yaml
fi
spack env activate -d spackenv

if [[ -z "${CI}" ]]
then
  echo "add ndam"
  spack add neurodamus-neocortex@develop+ngv+metabolism

  echo "add steps"
  if [ -d "HBP_STEPS" ]
  then
    echo "HBP_STEPS already set. Just pulling latest changes"
    pushd HBP_STEPS
    git pull
    git submodule update
    popd
  else
    echo "clone HBP_STEPS"
    if [[ -z "${steps_branch}" ]]
    then
      export steps_branch=master
    fi
    git clone -b $steps_branch --single-branch --recursive git@github.com:CNS-OIST/HBP_STEPS.git
  fi
  spack add steps@develop
  spack develop -p ${PWD}/HBP_STEPS --no-clone steps@develop
fi

echo
echo "   ### additional software"
echo
spack add py-psutil py-bluepysnap py-scipy py-neurodamus py-pytest

echo
echo "   ### spack install"
echo
spack install
spack env deactivate
spack env activate -d spackenv


echo
echo "   ### julia, python-venv"
echo
# apparently julia breaks the certificates. Loading after downloads and installations
module load julia

# we do not trust that julia has the correct packages if the folder is there since it is in the home folder
if [ -d "python-venv" ]
then
  echo "julia packages already set"
else
  echo "setup julia"
  rm -rf ~/.julia
  julia -e 'using Pkg; Pkg.add("IJulia")'
  julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
  julia -e 'using Pkg; Pkg.add("DiffEqBase")'
  julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
  julia -e 'using Pkg; Pkg.add("StaticArrays")'
  julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
  julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'
fi

# needed because there is no spack package for now. Needed before julia
echo
echo "   ### python-venv"
echo
if [ -d "python-venv" ]
then
  echo "python-venv already set"
  source ./python-venv/bin/activate
else
  python -m venv python-venv
  source python-venv/bin/activate
  pip install diffeqpy julia
fi

export OMP_NUM_THREADS=1
echo
echo "   ### setup completed"
echo
