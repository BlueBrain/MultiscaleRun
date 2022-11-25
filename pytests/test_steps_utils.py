import multiscale_run.steps_utils as su
import config


def init_sim():
    model = su.gen_model()
    mesh, _ = su.gen_mesh()
    su.init_solver(model=model, mesh=mesh)


def test_S3():
    config.steps_version = 3
    init_sim()


def test_S4():
    config.steps_version = 4
    init_sim()
