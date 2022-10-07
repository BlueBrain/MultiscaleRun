echo "! setup !"

module purge
module load unstable git python-dev py-neurodamus py-mpi4py julia
module load intel gcc hpe-mpi

# If no args provided, then the script purges any previous installation
# of steps & neurodamus found in your system
if [ $# -eq 0 ]
then
    # Default behavior without args -> ./set_env.sh
    echo "Remove first any installed version of steps & neurodamus."
    spack uninstall --all -y steps@develop+distmesh+petsc
    spack uninstall --all -y neurodamus-neocortex@develop+ngv+metabolism

    spack install steps@develop+distmesh+petsc
    spack install neurodamus-neocortex@develop+ngv+metabolism
else
    # ./set_env.sh {random arg}
    echo "Just loading already installed versions of steps & neurodamus."
fi
spack load steps@develop+distmesh+petsc
# package that gives us the special
spack load neurodamus-neocortex@develop+ngv+metabolism

# same approach as above
if [ $# -eq 0 ]
then
    rm -rf ~/.julia
    julia -e 'using Pkg; Pkg.add("IJulia")'
    julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
    julia -e 'using Pkg; Pkg.add("DiffEqBase")'
    julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
    julia -e 'using Pkg; Pkg.add("StaticArrays")'
    julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
    julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'
else
    echo "Julia packages already set."
fi

if [ $# -eq 0 ]
then
    rm -rf ./python-venv
    python -m venv ./python-venv
    source ./python-venv/bin/activate

    pip install psutil
    pip install julia
    pip install diffeqpy
    pip install pympler
    pip install h5py
    pip install bluepysnap
else
    source ./python-venv/bin/activate
fi

if [ $# -eq 0 ]
then
    rm -rf metabolismndam
    git clone --quiet -b main --single-branch git@bbpgitlab.epfl.ch:molsys/metabolismndam.git
else
    echo "Metabolism repo already set, just updating it."
    pushd metabolismndam
    git pull
    popd
fi

# STEPS related
export OMP_NUM_THREADS=1

echo "! setup completed !"
