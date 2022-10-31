import multiscale_run.steps_utils as su


def init_sim(steps_version):
    model = su.gen_model()
    mesh, _ = su.gen_mesh(
        steps_version=steps_version, mesh_path="steps_meshes/mc2c/mc2c.msh"
    )
    su.init_solver(model=model, mesh=mesh)


def test_S3():
    init_sim(steps_version=3)


def test_S4():
    init_sim(steps_version=4)
