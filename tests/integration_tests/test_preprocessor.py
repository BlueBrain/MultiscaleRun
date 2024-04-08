from neuron import h

h.nrnmpi_init()

from multiscale_run import (
    bloodflow_manager,
    config,
    neurodamus_manager,
    preprocessor,
    utils,
)


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
    conf = config.MsrConfig.rat_sscxS1HL_V6()
    tmp_mesh_path = conf.multiscale_run.mesh_path.parent.name + "_tmp"
    utils.rename_path(
        conf.multiscale_run.mesh_path.parent,
        conf.multiscale_run.mesh_path.parent.with_name(tmp_mesh_path),
    )

    pp = preprocessor.MsrPreprocessor(config=conf)
    ndam_m = neurodamus_manager.MsrNeurodamusManager(config=conf)
    bf_m = bloodflow_manager.MsrBloodflowManager(
        vasculature_path=ndam_m.get_vasculature_path(),
        parameters=conf.multiscale_run.bloodflow,
    )
    pp.autogen_mesh(ndam_m=ndam_m, bf_m=bf_m)

    utils.remove_path(conf.multiscale_run.mesh_path.parent)
    utils.rename_path(
        conf.multiscale_run.mesh_path.parent.with_name(tmp_mesh_path),
        conf.multiscale_run.mesh_path.parent,
    )


if __name__ == "__main__":
    test_gen_msh()
