##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy
from julia import Main

import config


def gen_metabolism_model():
    """import jl metabolism diff eq system code to py"""
    with open(config.julia_code_file, "r") as f:
        julia_code = f.read()
    metabolism = Main.eval(julia_code)
    return metabolism
