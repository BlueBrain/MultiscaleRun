from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multiscale_run")
except PackageNotFoundError:
    # package is not installed
    __version__ = "develop"
