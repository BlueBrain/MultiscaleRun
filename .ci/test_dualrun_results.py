import numpy as np

# import matplotlib.pyplot as plt


"""
def test_CompCount():
    S3 = np.loadtxt('./RESULTS/S3_CompCount.dat', dtype=np.float, delimiter=',')
    S4 = np.loadtxt('./RESULTS/S4_CompCount.dat', dtype=np.float, delimiter=',')

    assert np.allclose(S3, S4)
    assert np.allclose(S4, S3)
"""


def test_Moles_Current():
    S3 = np.loadtxt("./RESULTS/STEPS3/Moles_Current.csv", dtype=float, delimiter=",")
    S4 = np.loadtxt("./RESULTS/STEPS4/Moles_Current.csv", dtype=float, delimiter=",")

    assert np.allclose(S3, S4, atol=1e-3)

    # Sum of currents should be almost zero:
    # https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-591
    assert (S3[:, 1] * 1e6 <= 1e-3).all()  # from mA to nA
    assert (S4[:, 1] * 1e6 <= 1e-3).all()  # from mA to nA


if __name__ == "__main__":
    test_Moles_Current()
