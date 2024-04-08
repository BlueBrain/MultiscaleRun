import logging

import pytest
from neuron import h


@pytest.fixture(scope="session", autouse=True)
def neuron_init():
    logging.basicConfig(level=logging.INFO)
    h.nrnmpi_init()
