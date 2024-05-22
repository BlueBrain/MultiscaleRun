from multiscale_run import MsrConfig, MsrNeurodamusManager


def test_init(tmp_path):
    conf = MsrConfig.rat_sscxS1HL_V6(directory=tmp_path, check=False)
    MsrNeurodamusManager(config=conf)


if __name__ == "__main__":
    test_init()
