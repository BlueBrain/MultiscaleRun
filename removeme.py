from julia import Main


from multiscale_run import (
    utils,
    steps_utils,
    metabolism_utils,
    neurodamus_utils,
    bloodflow_manager,
    printer,
)

ATDPtot_n = metabolism_utils.load_metabolism_data(Main)
print(ATDPtot_n)