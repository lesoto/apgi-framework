"""Core APGI equations — Paper §4.1, Table 2.

All implementations follow the paper exactly:

    Eq. 1   compute_S_t          Sₜ = Πᵉ·|zᵉ| + Πⁱ_eff·|zⁱ|
    Eq. 2   ignition_probability  P(ign) = σ(γ_sig·(Sₜ − θₜ))
    Eq. 3   step_theta           dθ/dt = −λθ(θ−θ₀) + κ_meta·C − Δ_info·I + η_NE·NE(t)
    §4.2    compute_pi_i_eff     Πⁱ_eff = Πⁱ_baseline · exp(β_SM · M̂(c,a))
    App A.1 accumulate_S_t       Sₜ = (1−λ_S)·S_{t−1} + λ_S·S_input,  λ_S = 1−exp(−dt/τ_S)

Default parameter values are drawn from the physiologically-constrained
ranges in Table 12 / Table 3, calibrated so that median simulated S_t sits
near the ignition boundary (P ≈ 0.50) for near-threshold tasks.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Publication-calibrated simulation defaults (Table 12 / Table 3)
# ---------------------------------------------------------------------------
THETA_0_DEFAULT: float = 0.60  # θ₀  ∈ [0.25, 0.85]
LAMBDA_THETA_DEFAULT: float = 0.30  # λθ  ∈ [0.1, 0.5]  s⁻¹
KAPPA_META_DEFAULT: float = 0.20  # κ_meta ∈ [0.001, 0.5]
DELTA_INFO_DEFAULT: float = 0.15  # Δ_info ∈ [0, 1]
ETA_NE_DEFAULT: float = 0.10  # η_NE ∈ [0.01, 0.5]
GAMMA_SIG_DEFAULT: float = 5.0  # γ_sig ∈ [2, 7.5]
TAU_S_DEFAULT: float = 5.0  # τ_S in trial steps  (≈ 300 ms at ~60 ms/step)
BETA_SM_DEFAULT: float = 0.60  # β_SM ∈ [0.1, 1.2]

# Math Spec §"Post-Ignition Reset and Refractory Period" / Notation Appendix
DELTA_RESET_DEFAULT: float = 0.20  # δ_reset ≥ 0, post-ignition threshold increment
RHO_RETAIN_DEFAULT: float = 0.30  # ρ_retain ∈ [0.1, 0.5], post-ignition signal retention

# Math Spec §"Norepinephrine Modulation of Threshold" — canonical multiplicative
# rescaling θ ← θ·(1 + γ_NE·g_NE), distinct from Paper 1's additive η_NE·NE(t)
# proxy already implemented in step_theta.
GAMMA_NE_DEFAULT: float = 0.30  # γ_NE ≥ 0
NE_SUBDOMINANCE_LIMIT: float = 0.50  # γ_NE·g_NE_max ≤ 0.5 (Math Spec / Notation Appendix)

# Math Spec §4.2 / Notation Appendix — precision clamp bounds (numerical stability)
PI_MIN: float = 0.01
PI_MAX: float = 10.0


# ---------------------------------------------------------------------------
# Eq. §4.2  —  Effective interoceptive precision
# ---------------------------------------------------------------------------


def compute_pi_i_eff(
    pi_i_baseline: float,
    beta_sm: float,
    M_hat: float,
) -> float:
    """Effective interoceptive precision.

    Πⁱ_eff = Πⁱ_baseline · exp(β_SM · M̂(c,a)), clamped to [Π_min, Π_max]

    Paper §4.2, Table 2.  High somatic-marker activity (positive M̂) amplifies
    the interoceptive precision channel, lowering the ignition threshold for
    bodily-salient stimuli.  The clamp is mandatory (Math Spec §4.2 / Notation
    Appendix): the exponential term is unbounded and must be clipped to
    [Π_min, Π_max] after every precision computation to prevent numerical
    blow-up for large |β_SM · M̂|.

    Args:
        pi_i_baseline: Baseline interoceptive precision Πⁱ_baseline (≥ 0).
        beta_sm: Somatic-marker gain β_SM ∈ [0.1, 1.2].
        M_hat: Somatic marker estimate M̂(c,a) ∈ [−2, +2].

    Returns:
        Effective interoceptive precision Πⁱ_eff, clamped to [Π_min, Π_max].
    """
    if pi_i_baseline < 0:
        raise ValueError("pi_i_baseline must be non-negative")
    pi_i_eff = pi_i_baseline * np.exp(beta_sm * M_hat)
    return float(np.clip(pi_i_eff, PI_MIN, PI_MAX))


# ---------------------------------------------------------------------------
# Eq. 1  —  Precision-weighted surprise (instantaneous)
# ---------------------------------------------------------------------------


def compute_S_t(
    pi_e: float,
    z_e: float,
    pi_i_eff: float,
    z_i: float,
) -> float:
    """Instantaneous precision-weighted surprise.

    Sₜ = Πᵉ · |zᵉ| + Πⁱ_eff · |zⁱ|

    Paper §4.1.1, Eq. 1.  Use :func:`accumulate_S_t` for the leaky temporal
    integration over the τ_S accumulation window (Appendix A.1).

    Args:
        pi_e: Exteroceptive precision Πᵉ (≥ 0).
        z_e: Exteroceptive prediction error zᵉ.
        pi_i_eff: Effective interoceptive precision Πⁱ_eff (≥ 0).
        z_i: Interoceptive prediction error zⁱ.

    Returns:
        Instantaneous integrated salience signal Sₜ.
    """
    return float(pi_e * abs(z_e) + pi_i_eff * abs(z_i))


# ---------------------------------------------------------------------------
# Appendix A.1  —  Leaky temporal accumulation of Sₜ
# ---------------------------------------------------------------------------


def accumulate_S_t(
    S_prev: float,
    S_input: float,
    tau_S: float = TAU_S_DEFAULT,
    dt: float = 1.0,
) -> float:
    """Leaky temporal accumulation of the integrated salience signal.

    Sₜ = (1 − λ_S) · S_{t−1} + λ_S · S_input
    λ_S = 1 − exp(−dt / τ_S)

    Paper Appendix A.1 discrete-time implementation.  Both *dt* and *tau_S*
    must be in the same units (e.g. trial steps or milliseconds).

    Args:
        S_prev: Accumulated signal from the previous time step.
        S_input: Current instantaneous input from :func:`compute_S_t`.
        tau_S: Surprise accumulation timescale τ_S (same units as dt).
        dt: Integration step size (default 1 trial step).

    Returns:
        Updated accumulated signal Sₜ.
    """
    lambda_s = 1.0 - np.exp(-dt / tau_S)
    return float((1.0 - lambda_s) * S_prev + lambda_s * S_input)


# ---------------------------------------------------------------------------
# Eq. 2  —  Ignition probability (sigmoid criterion)
# ---------------------------------------------------------------------------


def ignition_probability(
    S_t: float,
    theta_t: float,
    gamma_sig: float = GAMMA_SIG_DEFAULT,
) -> float:
    """Probability of conscious ignition.

    P(ignition | Sₜ, θₜ) = σ(γ_sig · (Sₜ − θₜ))

    Paper §4.1.2, Eq. 2, Table 15 Phase 4.  The sigmoid steepness γ_sig
    ∈ [2, 7.5] controls the sharpness of the threshold crossing; γ_sig = 5
    is the recommended population-level default (Table 12).

    Args:
        S_t: Accumulated precision-weighted surprise Sₜ.
        theta_t: Current allostatic threshold θₜ.
        gamma_sig: Sigmoid steepness γ_sig (default 5.0).

    Returns:
        Ignition probability in (0, 1).
    """
    return float(1.0 / (1.0 + np.exp(-gamma_sig * (S_t - theta_t))))


def ignition_criterion(
    S_t: float,
    theta_t: float,
    gamma_sig: float = GAMMA_SIG_DEFAULT,
    rng: np.random.Generator | None = None,
) -> bool:
    """Stochastic ignition decision sampled from the sigmoid criterion.

    Paper §4.1.2, Eq. 2.  When *rng* is provided, ignition is drawn
    stochastically from P(ignition).  When *rng* is None the function
    returns the deterministic outcome P ≥ 0.5, which is equivalent to the
    hard threshold Sₜ ≥ θₜ and is useful for unit tests and analysis.

    Args:
        S_t: Accumulated precision-weighted surprise Sₜ.
        theta_t: Current allostatic threshold θₜ.
        gamma_sig: Sigmoid steepness γ_sig.
        rng: NumPy Generator for stochastic sampling; None → deterministic.

    Returns:
        True if ignition fires, False otherwise.
    """
    p = ignition_probability(S_t, theta_t, gamma_sig)
    if rng is None:
        return p >= 0.5
    return bool(rng.random() < p)


# ---------------------------------------------------------------------------
# Eq. 3  —  Allostatic threshold dynamics (Euler ODE step)
# ---------------------------------------------------------------------------


def step_theta(
    theta_t: float,
    C_t: float,
    I_t: float,
    NE_t: float = 0.0,
    fired: bool = False,
    theta_0: float = THETA_0_DEFAULT,
    lambda_theta: float = LAMBDA_THETA_DEFAULT,
    kappa_meta: float = KAPPA_META_DEFAULT,
    delta_info: float = DELTA_INFO_DEFAULT,
    eta_NE: float = ETA_NE_DEFAULT,
    delta_reset: float = DELTA_RESET_DEFAULT,
    dt: float = 1.0,
) -> float:
    """One Euler step of the allostatic threshold ODE, plus refractory reset.

    dθ/dt = −λθ(θ − θ₀) + κ_meta · C − Δ_info · I + η_NE · NE(t)
    θₜ₊₁ ← θₜ₊₁ + δ_reset · B(t)

    Paper §4.1.3, Eq. 3, Table 2.  The δ_reset term implements Math Spec
    §"Post-Ignition Reset and Refractory Period" / Step 11: a one-time
    threshold increment applied exactly when ignition fires, modelling the
    ~500ms attentional-blink refractory window.

    * Mean-reversion `−λθ(θ−θ₀)` pulls θ toward its homeostatic baseline θ₀.
    * Metabolic cost term `+κ_meta·C` raises the threshold after costly ignitions.
    * Information value term `−Δ_info·I` *lowers* the threshold for high-value
      stimuli (negative sign, Table 2).
    * Noradrenaline term `+η_NE·NE(t)` allows phasic LC-NE events to sharpen
      the threshold transiently (Table 4 neuromodulator mappings). This is
      Paper 1's additive proxy for η_θ·[C(t)−V(t)]; see
      :func:`apply_ne_threshold_gain` for the canonical multiplicative
      θ ← θ·(1+γ_NE·g_NE) rescaling (Math Spec §"Norepinephrine Modulation
      of Threshold").
    * `δ_reset·B(t)` raises θ once, immediately after ignition fires.

    Args:
        theta_t: Current threshold θₜ.
        C_t: Metabolic cost signal at time t (≥ 0).
        I_t: Information value signal at time t (≥ 0).
        NE_t: Noradrenaline drive NE(t) (default 0).
        fired: Whether ignition fired (B(t)) at this timestep.
        theta_0: Homeostatic baseline θ₀.
        lambda_theta: Mean-reversion rate λθ.
        kappa_meta: Metabolic cost coefficient κ_meta.
        delta_info: Information value coefficient Δ_info.
        eta_NE: Noradrenaline modulation coefficient η_NE.
        delta_reset: Post-ignition threshold increment δ_reset (≥ 0).
        dt: Integration step size (trial steps; default 1).

    Returns:
        Updated threshold θₜ₊₁.
    """
    dtheta = (
        -lambda_theta * (theta_t - theta_0)
        + kappa_meta * C_t
        - delta_info * I_t
        + eta_NE * NE_t
    )
    theta_next = theta_t + dt * dtheta
    if fired:
        theta_next += delta_reset
    return float(theta_next)


def apply_ne_threshold_gain(
    theta_t: float,
    g_NE: float,
    gamma_NE: float = GAMMA_NE_DEFAULT,
) -> float:
    """Canonical multiplicative NE threshold rescaling.

    θ ← θ · (1 + γ_NE · g_NE(t))

    Math Spec §"Norepinephrine Modulation of Threshold" / Notation Appendix.
    High arousal (elevated NE) raises the ignition threshold multiplicatively,
    making the system less reactive.  Applied after the allostatic update
    (:func:`step_theta`) and before the ignition decision.

    Distinct from the additive η_NE·NE(t) proxy inside :func:`step_theta`
    (Paper 1's reduced form).  Per spec, when both NE pathways are active
    simultaneously, γ_NE·g_NE must satisfy γ_NE·g_NE ≤ 0.5 to keep the
    multiplicative modulation sub-dominant relative to the allostatic update;
    this function raises ValueError if that constraint is violated.

    Args:
        theta_t: Threshold to rescale (post allostatic update).
        g_NE: Norepinephrine gain g_NE(t) (≥ 0).
        gamma_NE: NE modulation strength γ_NE (≥ 0).

    Returns:
        Rescaled threshold θ·(1 + γ_NE·g_NE).
    """
    if g_NE < 0:
        raise ValueError("g_NE must be non-negative")
    if gamma_NE < 0:
        raise ValueError("gamma_NE must be non-negative")
    if gamma_NE * g_NE > NE_SUBDOMINANCE_LIMIT:
        raise ValueError(
            f"gamma_NE * g_NE = {gamma_NE * g_NE:.4f} exceeds the sub-dominance "
            f"limit of {NE_SUBDOMINANCE_LIMIT}; deactivate one NE pathway or "
            "reduce gamma_NE/g_NE (Math Spec / Notation Appendix constraint)."
        )
    return float(theta_t * (1.0 + gamma_NE * g_NE))


def post_ignition_reset(
    S_t: float,
    fired: bool,
    rho_retain: float = RHO_RETAIN_DEFAULT,
) -> float:
    """Post-ignition partial reset of the accumulated signal.

    If B(t) = 1: S(t) ← ρ_retain · S(t), ρ_retain ∈ [0.1, 0.5]

    Math Spec §"Post-Ignition Reset and Refractory Period" / Step 14.  The
    threshold's refractory increment is handled separately by
    :func:`step_theta`'s `delta_reset` — no further threshold change happens
    here.  ρ_retain close to 0 produces a near-full reset (strong refractory
    period); close to 1 produces a mild partial reset.

    Args:
        S_t: Accumulated signal Sₜ before reset.
        fired: Whether ignition fired (B(t)) at this timestep.
        rho_retain: Fraction of S(t) retained after ignition, ∈ [0.1, 0.5].

    Returns:
        S(t) unchanged if no ignition, else ρ_retain · S(t).
    """
    if not fired:
        return float(S_t)
    return float(rho_retain * S_t)


def theta_equilibrium(
    C: float,
    I_val: float,
    NE: float = 0.0,
    theta_0: float = THETA_0_DEFAULT,
    lambda_theta: float = LAMBDA_THETA_DEFAULT,
    kappa_meta: float = KAPPA_META_DEFAULT,
    delta_info: float = DELTA_INFO_DEFAULT,
    eta_NE: float = ETA_NE_DEFAULT,
) -> float:
    """Steady-state threshold where dθ/dt = 0.

    θ* = θ₀ + (κ_meta · C − Δ_info · I + η_NE · NE) / λθ

    Useful for initialising θₜ before a simulation sequence.

    Args:
        C: Metabolic cost signal.
        I_val: Information value signal.
        NE: Noradrenaline drive (default 0).
        theta_0: Homeostatic baseline θ₀.
        lambda_theta: Mean-reversion rate λθ.
        kappa_meta: Metabolic cost coefficient κ_meta.
        delta_info: Information value coefficient Δ_info.
        eta_NE: Noradrenaline modulation coefficient η_NE.

    Returns:
        Equilibrium threshold θ*.
    """
    return float(
        theta_0 + (kappa_meta * C - delta_info * I_val + eta_NE * NE) / lambda_theta
    )


# ---------------------------------------------------------------------------
# Convenience: single-trial runner
# ---------------------------------------------------------------------------


def run_trial(
    pi_e: float,
    z_e: float,
    pi_i_baseline: float,
    z_i: float,
    M_hat: float,
    C_t: float,
    I_t: float,
    theta_t: float,
    beta_sm: float = BETA_SM_DEFAULT,
    gamma_sig: float = GAMMA_SIG_DEFAULT,
    S_prev: float = 0.0,
    tau_S: float = TAU_S_DEFAULT,
    NE_t: float = 0.0,
    theta_0: float = THETA_0_DEFAULT,
    lambda_theta: float = LAMBDA_THETA_DEFAULT,
    kappa_meta: float = KAPPA_META_DEFAULT,
    delta_info: float = DELTA_INFO_DEFAULT,
    eta_NE: float = ETA_NE_DEFAULT,
    delta_reset: float = DELTA_RESET_DEFAULT,
    rho_retain: float = RHO_RETAIN_DEFAULT,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run a single APGI trial and return all intermediate quantities.

    Implements the full trial sequence:
        1. Compute Πⁱ_eff (§4.2, clamped to [Π_min, Π_max])
        2. Compute instantaneous S_input (Eq. 1)
        3. Accumulate S_t (App. A.1)
        4. Sample ignition (Eq. 2)
        5. Step θ to next value, applying δ_reset·B(t) on ignition (Eq. 3 +
           Post-Ignition Reset)
        6. Partially reset S(t) via ρ_retain on ignition (Post-Ignition Reset)

    Returns:
        dict with keys: pi_i_eff, S_input, S_t, S_next, theta_t, theta_next,
        ignition_prob, ignition.
    """
    pi_i_eff = compute_pi_i_eff(pi_i_baseline, beta_sm, M_hat)
    S_input = compute_S_t(pi_e, z_e, pi_i_eff, z_i)
    S_t = accumulate_S_t(S_prev, S_input, tau_S=tau_S)
    p = ignition_probability(S_t, theta_t, gamma_sig)
    fired = ignition_criterion(S_t, theta_t, gamma_sig, rng=rng)
    theta_next = step_theta(
        theta_t,
        C_t,
        I_t,
        NE_t=NE_t,
        fired=fired,
        theta_0=theta_0,
        lambda_theta=lambda_theta,
        kappa_meta=kappa_meta,
        delta_info=delta_info,
        eta_NE=eta_NE,
        delta_reset=delta_reset,
    )
    S_next = post_ignition_reset(S_t, fired, rho_retain=rho_retain)
    return {
        "pi_i_eff": pi_i_eff,
        "S_input": S_input,
        "S_t": S_t,
        "S_next": S_next,
        "theta_t": theta_t,
        "theta_next": theta_next,
        "ignition_prob": p,
        "ignition": fired,
    }
