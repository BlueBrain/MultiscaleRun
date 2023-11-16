#!/bin/bash -l
#SBATCH --ntasks=1000
#SBATCH --time=1-00:00:00
#SBATCH --mem=0
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --account=proj137
#SBATCH --job-name=current

module load unstable py-mpi4py
module load unstable neurodamus-neocortex
module unload python

srun emodel-generalisation -v compute_currents \
    --input-path  /gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V10/build/sonata/networks/nodes/All/nodes.h5 \
    --output-path nodes.h5 \
    --morphology-path /gpfs/bbp.cscs.ch/project/proj83/entities/fixed-ais-L23PC-2020-12-10/ascii \
    --hoc-path  /gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V10/emodels/hoc \
    --protocol-config-path protocol_config.yaml \
    --parallel-lib dask_dataframe
