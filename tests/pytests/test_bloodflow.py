import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from multiscale_run import utils, bloodflow_manager

import config


def test_init():
    # TODO reenable with smaller circuit
    # bloodflow_m = bloodflow_manager.MsrBloodflowManager(vasculature_path="/gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V6/build/sonata/networks/nodes/vasculature/nodes.h5", params=config.bloodflow_params)
    pass


if __name__ == "__main__":
    test_init()
