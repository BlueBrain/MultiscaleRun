##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy

import logging

import config


def load_metabolism_data(main):

    logging.info("load metabolism data")
    ATDPtot_n = 1.4449961078157665
    main.eval(
        """
    modeldirname = "/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/metabolism_unit_models/"

    include(string(modeldirname,"FINAL_CLEAN/data_model_full/u0_db_refined_selected_oct2021.jl"))

    pardirname = string(modeldirname,"optimiz_unit/enzymes/enzymes_preBigg/COMBO/parameters_18nov22/")

    include(string(pardirname,"general_parameters.jl"))
    include(string(pardirname,"ephys_parameters.jl"))
    include(string(pardirname,"bf_input.jl"))
    include(string(pardirname,"generalisations.jl")) # Jolivet NADH shuttles, resp
    include(string(pardirname,"GLC_transport.jl"))
    include(string(pardirname,"GLYCOLYSIS.jl"))
    include(string(pardirname,"glycogen.jl"))

    include(string(pardirname,"creatine.jl"))

    include(string(pardirname,"ATDMP.jl"))

    include(string(pardirname,"pyrTrCytoMito.jl"))
    include(string(pardirname,"lactate.jl"))
    include(string(pardirname,"TCA.jl"))

    include(string(pardirname,"ETC.jl"))

    include(string(pardirname,"PPP_n.jl"))
    include(string(pardirname,"PPP_a.jl"))
    include(string(pardirname,"gshgssg.jl"))

    include(string(pardirname,"MAS.jl"))
    include(string(pardirname,"gltgln.jl"))
    include(string(pardirname,"pyrCarb.jl"))
    include(string(pardirname,"ketones.jl"))

    # for NEmodulation
    xNEmod = 0.025 # 0.1 #0.00011
    KdNEmod = 3.0e-4 # 3.6e-5  # 3.0e-4 #

    Iinj = 0.0
    synInput = 0.0

    """
    )

    logging.info("load metabolism data")
    return ATDPtot_n


def gen_metabolism_model(main):
    """import jl metabolism diff eq system code to py"""
    with open(config.julia_code_file, "r") as f:
        julia_code = f.read()
    metabolism = main.eval(julia_code)
    return metabolism
