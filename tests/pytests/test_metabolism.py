import sys, os, glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")



from multiscale_run import printer, metabolism_manager, utils
config = utils.load_config()

from julia import Main


def test_metabolism():
    prnt = printer.MsrPrinter()
    metab_m = metabolism_manager.MsrMetabolismManager(
        u0_file=config.u0_file, main=Main, prnt=prnt
    )


if __name__ == "__main__":
    test_metabolism()
