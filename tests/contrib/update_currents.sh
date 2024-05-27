#!/bin/bash -l
#SBATCH --ntasks=1000
#SBATCH --time=1-00:00:00
#SBATCH --mem=0
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --account=proj137
#SBATCH --job-name=msr-update-currents

# This script computes holding and threshold currents
# (columns with values per cell) and write them in HDF5 file "nodes.h5".
# These are needed to run a simulation (only neurons part) to know how much current
# to inject per cell, so they are below threshold, etc...
# It is usually part of circuit-build in regular BBP circuits.

# nodes.h5 file is required in MultiscaleRun simulation to provide input resistance
# information for the Ornstein-Uhlenbeck type of stimuli.

module load unstable
module load py-emodel-generalisation
module load neurodamus-neocortex-multiscale

srun \
emodel-generalisation -v compute_currents \
    --input-path  /gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V10/build/sonata/networks/nodes/All/nodes.h5 \
    --output-path nodes.h5 \
    --morphology-path /gpfs/bbp.cscs.ch/project/proj83/entities/fixed-ais-L23PC-2020-12-10/ascii \
    --hoc-path  /gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V10/emodels/hoc \
    --protocol-config-path protocol_config.yaml \
    --only-rin \
    --parallel-lib dask_dataframe
#    --parallel-lib multiprocessing
