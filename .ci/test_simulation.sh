#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CI_JOB_NAME_SLUG=${CI_JOB_NAME_SLUG:-`date +%Y-%m-%d_%Hh%M`}
sim_name=${sim_name:-msr-sim-$CI_JOB_NAME_SLUG}

bb5_ntasks=${bb5_ntasks:-2}
postproc=${postproc:true}

# Variables used to edit the JSON configuration of the simulation
steps=${steps:-false}
metabolism=${metabolism:-false}
bloodflow=${bloodflow:-false}
sim_end=${sim_end:-1000}

if [ -z ${sim_name:x} ]; then
  fatal_error "expected environment variable 'SIM_NAME'."
fi

multiscale-run init --no-check -f "$sim_name"

pushd "$sim_name" >/dev/null
/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/bin/jq ".with_steps = $steps | .with_bloodflow = $bloodflow | .with_metabolism = $metabolism | .msr_sim_end = $sim_end " msr_config.json > msr_config.json.bak
mv msr_config.json.bak msr_config.json
cat msr_config.json
popd >/dev/null
unset steps bloodflow metabolism sim_end

module load unstable intel-oneapi-mkl
srun --overlap -n $bb5_ntasks multiscale-run compute "$sim_name"
