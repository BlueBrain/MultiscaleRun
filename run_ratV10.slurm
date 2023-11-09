#!/bin/bash

#SBATCH --job-name="msr_ratV10"
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32

#SBATCH --account=proj40
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=08:00:00

#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output="%x-%j.log"

# check multiscale_run/config.py for more info on how the config object is constructed

# first time, today
export IS_SPACK_UPDATED=1
source setup.sh
unset IS_SPACK_UPDATED

# the other times
# source setup_quick.sh

export base_path="configs/rat_sscxS1HL_V10_all_valid_cells"

time srun --overlap dplace python main.py
