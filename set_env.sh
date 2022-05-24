module purge
module load unstable python-dev py-neurodamus

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
    rm -rf ./python-venv
    python -m venv ./python-venv
    source ./python-venv/bin/activate

    pip install psutil
else
    source ./python-venv/bin/activate
fi

echo "setup completed"
