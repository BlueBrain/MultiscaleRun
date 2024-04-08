from multiscale_run import MsrConfig, MsrNeurodamusManager


def test_init():
    conf = MsrConfig.rat_sscxS1HL_V6()
    MsrNeurodamusManager(config=conf)


if __name__ == "__main__":
    test_init()
