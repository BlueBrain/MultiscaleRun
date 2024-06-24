from julia import Main

from multiscale_run import config, metabolism_manager


def test_metabolism():
    conf = config.MsrConfig.default()
    metabolism_manager.MsrMetabolismManager(
        config=conf, main=Main, neuron_pop_name="All", gids=[]
    )


if __name__ == "__main__":
    test_metabolism()
