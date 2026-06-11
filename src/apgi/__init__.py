"""APGI — Adaptive Predictive Global Integration framework."""

from importlib.metadata import PackageNotFoundError, version

from apgi import datasets

try:
    __version__ = version("apgi")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from apgi import parameter_recovery
from apgi.clinical import (
    ClinicalReport,
    ConsciousnessLevel,
    EnhancedClinicalInterpreter,
)
from apgi.core import (
    accumulate_S_t,
    compute_pi_i_eff,
    compute_S_t,
    ignition_criterion,
    ignition_probability,
    run_trial,
    step_theta,
    theta_equilibrium,
)
from apgi.integration import APGICoreIntegration, TrialRecord
from apgi.normalizer import APGINormalizer

# Paper 2 / Paper 3 extensions are NOT imported here.
# Use: from apgi.extensions.liquid_network import LiquidNeuralNetwork
#      from apgi.extensions.hierarchical import APGIHierarchy

__all__ = [
    # core equations (paper §4.1, App. A.1)
    "accumulate_S_t",
    "compute_pi_i_eff",
    "compute_S_t",
    "ignition_criterion",
    "ignition_probability",
    "run_trial",
    "step_theta",
    "theta_equilibrium",
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
    # inline sample datasets (no download required)
    "datasets",
]
