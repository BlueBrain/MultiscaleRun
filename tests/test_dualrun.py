import numpy as np

# import matplotlib.pyplot as plt


def test_Moles_Current():
    currents = np.loadtxt("./RESULTS/Moles_Current.csv", dtype=float, delimiter=",")

    # Sum of currents should be almost zero:
    # https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-591
    assert (currents[:, 1] * 1e6 <= 1e-3).all()  # from mA to nA


if __name__ == "__main__":
    test_Moles_Current()
