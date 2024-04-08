from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multiscale_run")
except PackageNotFoundError:
    # package is not installed
    __version__ = "develop"

from .bloodflow_manager import MsrBloodflowManager
from .config import MsrConfig
from .connection_manager import MsrConnectionManager
from .metabolism_manager import MsrMetabolismManager
from .neurodamus_manager import MsrNeurodamusManager
from .preprocessor import MsrPreprocessor
from .reporter import MsrReporter
from .simulation import MsrSimulation
from .steps_manager import MsrStepsManager
