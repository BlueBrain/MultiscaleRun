import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from multiscale_run import bloodflow_manager, utils


def test_init():
    # TODO reenable with smaller circuit
    # bloodflow_m = bloodflow_manager.MsrBloodflowManager(vasculature_path="/gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V6/build/sonata/networks/nodes/vasculature/nodes.h5", params=config.bloodflow_params)
    pass


if __name__ == "__main__":
    test_init()
