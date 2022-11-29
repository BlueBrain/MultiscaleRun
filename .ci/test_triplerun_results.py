import re
import numpy as np


def ck_and_get_ATPconc(path):
    notebook_html = open(path).read()
    notebook_html = notebook_html.split("\n")

    ATPconc = None
    for line in notebook_html:

        # Choose the line with the following format -> data_ATPConcAllCmps : mean=..., min=..., max=...
        if "data_ATPConcAllCmps : mean" in line:
            ATPconc = line
            break
    assert ATPconc is not None
    ATPconc = np.array(
        [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", ATPconc)]
    )

    print(ATPconc)

    # ATP should be between these values
    assert ATPconc[ATPconc < 0.1].size == 0
    assert ATPconc[ATPconc > 2.2].size == 0

    return ATPconc


def test_ATPconc():
    ATPconc_S3 = ck_and_get_ATPconc("./RESULTS/STEPS3/postproc.html")
    ATPconc_S4 = ck_and_get_ATPconc("./RESULTS/STEPS4/postproc.html")

    assert np.allclose(ATPconc_S3, ATPconc_S4)
    assert np.allclose(ATPconc_S4, ATPconc_S3)
