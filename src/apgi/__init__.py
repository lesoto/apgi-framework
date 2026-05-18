"""APGI — Adaptive Predictive Global Integration framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("apgi")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from apgi.core import (
    compute_pi_i_eff,
    compute_S_t,
    compute_theta_t,
    ignition_criterion,
    run_trial,
    update_theta,
)
from apgi.integration import APGICoreIntegration, TrialRecord
from apgi.normalizer import APGINormalizer
from apgi.clinical import (
    ConsciousnessLevel,
    ClinicalReport,
    EnhancedClinicalInterpreter,
)
from apgi import parameter_recovery

# Paper 2 / Paper 3 extensions are NOT imported here.
# Use: from apgi.extensions.liquid_network import LiquidNeuralNetwork
#      from apgi.extensions.hierarchical import APGIHierarchy

__all__ = [
    # core equations
    "compute_pi_i_eff",
    "compute_S_t",
    "compute_theta_t",
    "ignition_criterion",
    "run_trial",
    "update_theta",
    # session integration
    "APGICoreIntegration",
    "TrialRecord",
    # normalisation
    "APGINormalizer",
    # clinical interpretation
    "ConsciousnessLevel",
    "ClinicalReport",
    "EnhancedClinicalInterpreter",
    # parameter recovery sub-module
    "parameter_recovery",
]
