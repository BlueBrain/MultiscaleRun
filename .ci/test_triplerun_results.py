import re
import numpy as np

def test_ATPConc():
    notebook_html = open('./RESULTS/notebook.html').read()
    notebook_html = notebook_html.split('\n')

    ATPConc = None
    for line in notebook_html:
        # Choose the line with the following format -> data_ATPConcAllCmps : mean=..., min=..., max=...
        if 'data_ATPConcAllCmps : ' in line:
            ATPConc = line

    assert ATPConc is not None
    
    # Extract the numbers from the above-mentioned line
    ATPConc = np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)', ATPConc)])

    # ATP should be between these values
    assert ATPConc[ATPConc < 1.0].size == 0
    assert ATPConc[ATPConc > 2.2].size == 0
