from neuron import h

from multiscale_run import (
    MsrBloodflowManager,
    MsrConfig,
    MsrNeurodamusManager,
    MsrPreprocessor,
    utils,
)


@utils.pushtempd
def test_gen_msh():
    """
    Test the generation of a mesh with specific configurations and managers.

    This function is used to test the generation of a mesh with specific configurations and managers. It follows these steps:
    1. Initializes a 'MsrConfig' object to configure the test.
    2. Creates a temporary mesh path and renames the original mesh path.
    3. Initializes a preprocessor using 'MsrPreprocessor' with the provided configuration.
    4. Initializes a neurodamus manager using 'MsrNeurodamusManager' with the same configuration.
    5. Initializes a bloodflow manager using 'MsrBloodflowManager' with specific parameters and vasculature path.
    6. Generates a mesh using the 'autogen_mesh' method of the preprocessor with neurodamus and bloodflow managers.
    7. Removes the original mesh path.
    8. Renames the temporary mesh path back to its original name.

    This function is responsible for testing the mesh generation process with given configurations and managers.

    """
    conf = MsrConfig.default(circuit="rat_sscxS1HL_V10_CI", force=True, check=False)

    pp = MsrPreprocessor(config=conf)
    ndam_m = MsrNeurodamusManager(config=conf)
    bf_m = MsrBloodflowManager(
        vasculature_path=ndam_m.get_vasculature_path(),
        parameters=conf.multiscale_run.bloodflow,
    )
    pp.autogen_mesh(ndam_m=ndam_m, bf_m=bf_m)


if __name__ == "__main__":
    h.nrnmpi_init()
    test_gen_msh()
