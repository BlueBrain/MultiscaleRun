from multiscale_run import MsrConfig, MsrNeurodamusManager, utils


def test_init(tmp_path):
    conf = MsrConfig.default(directory=tmp_path, check=False)
    MsrNeurodamusManager(config=conf)


if __name__ == "__main__":
    tmp_path = "./tmp"
    utils.remove_path(tmp_path)
    test_init(tmp_path)
    utils.remove_path(tmp_path)
