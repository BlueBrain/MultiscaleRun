echo "! setup !"

module purge
module load unstable python-dev py-neurodamus py-mpi4py julia
module load intel gcc hpe-mpi

# If no args provided, then the script purges any previous installation
# of steps & neurodamus found in your system
if [ $# -eq 0 ]
then
    # Default behavior without args -> ./set_env.sh
    echo "Remove first any installed version of steps & neurodamus."
    spack uninstall --all -y steps@develop+distmesh+petsc
    spack uninstall --all -y neurodamus-neocortex+ngv

    spack install steps@develop+distmesh+petsc
    spack install neurodamus-neocortex+ngv
else
    # ./set_env.sh {random arg}
    echo "Just loading already installed versions of steps & neurodamus."
fi
spack load steps@develop+distmesh+petsc
spack load neurodamus-neocortex+ngv

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

if [ $# -eq 0 ]
then
    echo "building custom special."
    rm -rf x86_64
    
    # legacy mod files for triplerun
    cp metabolismndam/custom_ndam_2021_02_22_archive202101/mod/* mod/

    # update the legacy ones with mod files from neurodamus-core
    rm -rf neurodamus-core/
    git clone --quiet -b main --single-branch git@bbpgitlab.epfl.ch:hpc/sim/neurodamus-core.git
    cp neurodamus-core/mod/* mod/
    rm -rf neurodamus-core/

    # additional mod files from common repo
    rm -rf common/
    git clone --quiet -b main --single-branch git@bbpgitlab.epfl.ch:hpc/sim/models/common.git
    cp -n common/mod/ngv/* mod/
    rm -rf common/

    build_neurodamus.sh mod
else
    echo "custom special already built."
fi

# STEPS related
export OMP_NUM_THREADS=1

echo "! setup completed !"
