##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy
from julia import Main

import params


def gen_metabolism_model():
    """import jl metabolism diff eq system code to py"""
    with open(params.julia_code_file, "r") as f:
        julia_code = f.read()
    metabolism = Main.eval(julia_code)
    return metabolism
