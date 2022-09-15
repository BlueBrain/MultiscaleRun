import re
import numpy as np

def test_ATPConc():
    notebook_html = open('./RESULTS/notebook.html').read()
    notebook_html = notebook_html.split('\n')

    ATPConc = None
    for line in notebook_html:
        if 'data_ATPConcAllCmps : ' in line:
            ATPConc = line

    assert ATPConc is not None
    
    ATPConc = np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)', ATPConc)])

    assert ATPConc[ATPConc < 1.0].size == 0
    assert ATPConc[ATPConc > 2.2].size == 0
