"""APGI — Adaptive Predictive Global Integration framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("apgi")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from apgi import apgi_core, hierarchical, liquid_network, parameter_recovery

__all__ = ["apgi_core", "liquid_network", "hierarchical", "parameter_recovery"]
