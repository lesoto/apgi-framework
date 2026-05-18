"""Core APGI equations: Sₜ, θₜ, Πⁱ_eff, and the ignition criterion."""

import numpy as np


def compute_pi_i_eff(
    pi_i: float,
    C_metabolic: "float | np.ndarray",
    kappa: float = 100.0,
) -> "float | np.ndarray":
    """Effective inhibitory precision.

    Πⁱ_eff = Πⁱ · exp(−C_metabolic / κ)

    Accepts both scalar and array inputs for C_metabolic.

    Args:
        pi_i: Raw inhibitory precision Πⁱ (≥ 0).
        C_metabolic: Metabolic cost signal (ATP units/bit, ≥ 0); scalar or array.
        kappa: Energy-information coupling constant κ ≈ 100 ATP/bit.

    Returns:
        Effective inhibitory precision Πⁱ_eff (same shape as C_metabolic).
    """
    if pi_i < 0:
        raise ValueError("pi_i must be non-negative")
    C = np.asarray(C_metabolic)
    if np.any(C < 0):
        raise ValueError("C_metabolic must be non-negative")
    result = pi_i * np.exp(-C / kappa)
    return float(result) if result.ndim == 0 else result


def compute_S_t(
    pi_e: float,
    z_e: float,
    pi_i_eff: float,
    z_i: float,
) -> float:
    """Global integration signal.

    Sₜ = Πᵉ · |zᵉ| + Πⁱ_eff · |zⁱ|

    Args:
        pi_e: Excitatory precision Πᵉ (≥ 0).
        z_e: Excitatory state variable zᵉ.
        pi_i_eff: Effective inhibitory precision Πⁱ_eff (≥ 0).
        z_i: Inhibitory state variable zⁱ.

    Returns:
        Global integration signal Sₜ.
    """
    return float(pi_e * abs(z_e) + pi_i_eff * abs(z_i))


def compute_theta_t(
    C_metabolic: float,
    V_information: float,
    alpha: float,
    beta: float,
) -> float:
    """Adaptive ignition threshold.

    θₜ = α · C_metabolic + β · V_information

    Args:
        C_metabolic: Metabolic cost signal (≥ 0).
        V_information: Information value signal (≥ 0).
        alpha: Metabolic weighting coefficient α (≥ 0).
        beta: Information weighting coefficient β (≥ 0).

    Returns:
        Ignition threshold θₜ.
    """
    return float(alpha * C_metabolic + beta * V_information)


def update_theta(
    theta_t: float,
    C_metabolic: float,
    V_information: float,
    alpha: float,
    beta: float,
    gamma: float = 0.9,
) -> float:
    """Threshold update rule: exponential smoothing toward new target.

    θₜ₊₁ = γ · θₜ + (1 − γ) · (α · C_metabolic + β · V_information)

    Args:
        theta_t: Current threshold θₜ.
        C_metabolic: Metabolic cost at time t+1.
        V_information: Information value at time t+1.
        alpha: Metabolic weighting coefficient.
        beta: Information weighting coefficient.
        gamma: Smoothing factor γ ∈ [0, 1].

    Returns:
        Updated threshold θₜ₊₁.
    """
    target = compute_theta_t(C_metabolic, V_information, alpha, beta)
    return float(gamma * theta_t + (1.0 - gamma) * target)


def ignition_criterion(S_t: float, theta_t: float) -> bool:
    """Return True when global integration signal exceeds threshold.

    Ignition fires when Sₜ ≥ θₜ.
    """
    return S_t >= theta_t


def run_trial(
    pi_e: float,
    z_e: float,
    pi_i: float,
    z_i: float,
    C_metabolic: float,
    V_information: float,
    alpha: float,
    beta: float,
    kappa: float = 100.0,
) -> dict:
    """Run a single APGI trial and return all intermediate quantities.

    Returns:
        dict with keys: pi_i_eff, S_t, theta_t, ignition.
    """
    pi_i_eff = compute_pi_i_eff(pi_i, C_metabolic, kappa)
    S_t = compute_S_t(pi_e, z_e, pi_i_eff, z_i)
    theta_t = compute_theta_t(C_metabolic, V_information, alpha, beta)
    return {
        "pi_i_eff": pi_i_eff,
        "S_t": S_t,
        "theta_t": theta_t,
        "ignition": ignition_criterion(S_t, theta_t),
    }
