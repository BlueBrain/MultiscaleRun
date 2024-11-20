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
tstop=${tstop:-290}
circuit=${circuit:-tiny_CI}

if [ -z ${sim_name:x} ]; then
  fatal_error "expected environment variable 'SIM_NAME'."
fi

rm -rf "$sim_name"
multiscale-run init --no-check -f "$sim_name" --circuit $circuit

pushd "$sim_name"
/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/bin/jq ".multiscale_run.with_steps = $steps | .multiscale_run.with_bloodflow = $bloodflow | .multiscale_run.with_metabolism = $metabolism | .run.tstop = $tstop " simulation_config.json > simulation_config.json.bak
mv simulation_config.json.bak simulation_config.json
cat simulation_config.json
popd >/dev/null
unset steps bloodflow metabolism tstop

module load unstable intel-oneapi-mkl
srun --overlap -n $bb5_ntasks multiscale-run compute "$sim_name"
multiscale-run post-processing "$sim_name"
