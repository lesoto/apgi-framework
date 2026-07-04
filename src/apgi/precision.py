"""Online precision / z-score estimation pipeline — Math Spec §1-2.

Implements the signal-preprocessing pipeline that the core equations
(:mod:`apgi.core`) assume as their input: raw prediction errors are
converted into precision-weighted z-scores via an online running-mean/
variance estimator, precision is the (clamped) inverse variance, and each
channel is scaled by its neuromodulatory gain (ACh for exteroceptive,
NE for interoceptive).

    Step 1  update_running_mean   μ(t+1) = (1-α_EMA)·μ(t) + α_EMA·ε(t)
    Step 2  update_running_var    σ²(t+1) = (1-α_EMA)·σ²(t) + α_EMA·(ε(t)-μ(t))²
    Eq.     precision_from_variance   Π(t) = 1/(σ²(t)+ε_stab), clamped to [Π_min, Π_max]
    Eq.     z_score                z(t) = (ε(t)-μ(t))/(σ(t)+ε_stab)
    Eq.     apply_neuromodulatory_gain   Π_e^eff = g_ACh·Π_e ; Π_i^eff = g_NE·Π_i

The two-step mean/variance update is mandatory (Math Spec §1): updating
variance from the raw signal instead of the mean-centred residual yields
E[ε²] rather than E[(ε-μ)²], which only coincide for a zero-mean predictor.
"""

from __future__ import annotations

import numpy as np

from apgi.core import PI_MAX, PI_MIN

ALPHA_EMA_DEFAULT: float = 0.1  # α_EMA ∈ (0, 1)
EPS_STAB_DEFAULT: float = 1e-6  # ε_stab, stability constant (denominators only)


def update_running_mean(
    mu: float,
    eps_t: float,
    alpha_ema: float = ALPHA_EMA_DEFAULT,
) -> float:
    """Step 1 — EMA update of the running mean μ(t+1).

    μ(t+1) = (1 - α_EMA)·μ(t) + α_EMA·ε(t)
    """
    return float((1.0 - alpha_ema) * mu + alpha_ema * eps_t)


def update_running_variance(
    sigma2: float,
    eps_t: float,
    mu_t: float,
    alpha_ema: float = ALPHA_EMA_DEFAULT,
) -> float:
    """Step 2 — EMA update of the running variance σ²(t+1), mean-centred.

    σ²(t+1) = (1 - α_EMA)·σ²(t) + α_EMA·(ε(t) - μ(t))²

    Must use the *pre-update* mean μ(t) (the two-step form), not μ(t+1).
    """
    return float((1.0 - alpha_ema) * sigma2 + alpha_ema * (eps_t - mu_t) ** 2)


def precision_from_variance(
    sigma2: float,
    eps_stab: float = EPS_STAB_DEFAULT,
) -> float:
    """Precision Π(t) = 1/(σ²(t) + ε_stab), clamped to [Π_min, Π_max]."""
    pi = 1.0 / (sigma2 + eps_stab)
    return float(np.clip(pi, PI_MIN, PI_MAX))


def z_score(
    eps_t: float,
    mu_t: float,
    sigma2_t: float,
    eps_stab: float = EPS_STAB_DEFAULT,
) -> float:
    """Standardised prediction error z(t) = (ε(t) - μ(t)) / (σ(t) + ε_stab)."""
    sigma_t = np.sqrt(max(sigma2_t, 0.0))
    return float((eps_t - mu_t) / (sigma_t + eps_stab))


def apply_neuromodulatory_gain(pi: float, gain: float) -> float:
    """Neuromodulatory precision gain: Π_e^eff = g_ACh·Π_e ; Π_i^eff = g_NE·Π_i.

    Each neuromodulator governs exactly one channel (Math Spec §2) — pass
    g_ACh for the exteroceptive channel and g_NE for the interoceptive
    channel. Does not itself re-clamp; callers may clamp the result to
    [Π_min, Π_max] if the gain can push it out of range.
    """
    return float(pi * gain)


class OnlinePrecisionChannel:
    """Stateful per-channel running estimator: ε(t) → (μ, σ², Π, z).

    One instance per channel (exteroceptive or interoceptive). Initialised
    per Math Spec §1: μ(0) = 0, σ²(0) = 1.
    """

    def __init__(
        self,
        alpha_ema: float = ALPHA_EMA_DEFAULT,
        eps_stab: float = EPS_STAB_DEFAULT,
    ) -> None:
        self.alpha_ema = alpha_ema
        self.eps_stab = eps_stab
        self.mu: float = 0.0
        self.sigma2: float = 1.0

    def update(self, eps_t: float) -> dict:
        """Ingest one raw prediction error and return the derived quantities.

        Returns:
            dict with keys: mu, sigma2, pi, z (all post-update except z,
            which uses the pre-update μ/σ² per the spec's z-score definition).
        """
        z = z_score(eps_t, self.mu, self.sigma2, eps_stab=self.eps_stab)
        pi = precision_from_variance(self.sigma2, eps_stab=self.eps_stab)
        mu_next = update_running_mean(self.mu, eps_t, alpha_ema=self.alpha_ema)
        sigma2_next = update_running_variance(
            self.sigma2, eps_t, self.mu, alpha_ema=self.alpha_ema
        )
        self.mu = mu_next
        self.sigma2 = sigma2_next
        return {"mu": self.mu, "sigma2": self.sigma2, "pi": pi, "z": z}
