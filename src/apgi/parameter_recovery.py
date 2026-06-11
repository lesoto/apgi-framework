"""Parameter recovery simulation — Appendix A.4, Table 13.

Verifies that the APGI model can recover its own generating parameters from
synthetic data via maximum-likelihood estimation.  Five parameters are
recovered (Table 13):

    θ₀      homeostatic threshold baseline         r ≥ 0.84  (target)
    τ_S     surprise accumulation timescale        r ≥ 0.82
    Πⁱ      baseline interoceptive precision       r ≥ 0.76
    β_SM    somatic-marker gain                    r ≥ 0.71
    γ_sig   sigmoid steepness                      r ≥ 0.58

The generative model follows §4.1 and Appendix A.1 exactly:
    Πⁱ_eff = Πⁱ_baseline · exp(β_SM · M̂)
    S_input = Πᵉ·|zᵉ| + Πⁱ_eff·|zⁱ|
    Sₜ      = (1−λ_S)·S_{t−1} + λ_S·S_input,  λ_S = 1 − exp(−1/τ_S)
    θₜ₊₁   = step_theta(θₜ, C_t, I_t)
    P(ign)  = σ(γ_sig · (Sₜ − θₜ))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import pearsonr

from apgi.core import (
    DELTA_INFO_DEFAULT,
    KAPPA_META_DEFAULT,
    LAMBDA_THETA_DEFAULT,
    accumulate_S_t,
    compute_pi_i_eff,
    compute_S_t,
    step_theta,
)

# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    n_trials: int,
    theta_0_true: float,
    tau_S_true: float,
    pi_i_true: float,
    beta_sm_true: float,
    gamma_sig_true: float,
    lambda_theta: float = LAMBDA_THETA_DEFAULT,
    kappa_meta: float = KAPPA_META_DEFAULT,
    delta_info: float = DELTA_INFO_DEFAULT,
    noise_sd: float = 0.05,
    seed: int = 0,
) -> dict:
    """Generate synthetic trial data from the APGI generative model.

    Fixed parameters (not recovered): λθ, κ_meta, Δ_info.
    Free parameters (to recover): θ₀, τ_S, Πⁱ_baseline, β_SM, γ_sig.

    Returns:
        dict with keys: pi_e, z_e, z_i, C_t, I_t, M_hat,
        S_t_observed (noisy), theta_t_series, ignition.
    """
    rng = np.random.default_rng(seed)

    pi_e = rng.uniform(0.8, 1.5, n_trials)
    z_e = rng.uniform(0.2, 0.8, n_trials)
    z_i = rng.uniform(0.1, 0.6, n_trials)
    C_t = rng.uniform(0.5, 2.0, n_trials)
    I_t = rng.uniform(0.1, 1.0, n_trials)
    M_hat = rng.uniform(0.0, 1.0, n_trials)

    S_acc = 0.0
    theta_t = (
        theta_0_true
        + (kappa_meta * float(C_t.mean()) - delta_info * float(I_t.mean()))
        / lambda_theta
    )

    S_t_series = np.empty(n_trials)
    theta_series = np.empty(n_trials)
    ignition_series = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        pi_i_eff = compute_pi_i_eff(pi_i_true, beta_sm_true, float(M_hat[t]))
        S_input = compute_S_t(float(pi_e[t]), float(z_e[t]), pi_i_eff, float(z_i[t]))
        S_acc = accumulate_S_t(S_acc, S_input, tau_S=tau_S_true)
        S_t_series[t] = S_acc
        theta_series[t] = theta_t
        p = 1.0 / (1.0 + np.exp(-gamma_sig_true * (S_acc - theta_t)))
        ignition_series[t] = int(rng.random() < p)
        theta_t = step_theta(
            theta_t,
            float(C_t[t]),
            float(I_t[t]),
            theta_0=theta_0_true,
            lambda_theta=lambda_theta,
            kappa_meta=kappa_meta,
            delta_info=delta_info,
        )

    S_t_observed = S_t_series + rng.normal(0, noise_sd, n_trials)

    return {
        "pi_e": pi_e,
        "z_e": z_e,
        "z_i": z_i,
        "C_t": C_t,
        "I_t": I_t,
        "M_hat": M_hat,
        "S_t_observed": S_t_observed,
        "theta_t_series": theta_series,
        "ignition": ignition_series,
    }


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def _negative_log_likelihood(
    params: NDArray,
    data: dict,
    lambda_theta: float,
    kappa_meta: float,
    delta_info: float,
) -> float:
    """Joint NLL: Gaussian over Sₜ + Bernoulli over ignition.

    params = [theta_0, tau_S, pi_i_baseline, beta_sm, gamma_sig]
    """
    theta_0, tau_S, pi_i_baseline, beta_sm, gamma_sig = params

    if (
        tau_S <= 0.2
        or pi_i_baseline <= 0.0
        or gamma_sig < 0.5
        or tau_S > 50.0
        or gamma_sig > 20.0
        or pi_i_baseline > 10.0
        or abs(beta_sm) > 3.0
    ):
        return 1e12

    pi_e = np.asarray(data["pi_e"])
    z_e = np.asarray(data["z_e"])
    z_i = np.asarray(data["z_i"])
    C_t = np.asarray(data["C_t"])
    I_t = np.asarray(data["I_t"])
    M_hat = np.asarray(data["M_hat"])
    S_obs = np.asarray(data["S_t_observed"])
    ign = np.asarray(data["ignition"])
    n = len(pi_e)

    S_acc = 0.0
    theta_t = (
        theta_0
        + (kappa_meta * float(C_t.mean()) - delta_info * float(I_t.mean()))
        / lambda_theta
    )

    gaussian_nll = 0.0
    bernoulli_nll = 0.0

    for t in range(n):
        pi_i_eff = pi_i_baseline * np.exp(beta_sm * M_hat[t])
        S_input = pi_e[t] * abs(z_e[t]) + pi_i_eff * abs(z_i[t])
        lam = 1.0 - np.exp(-1.0 / tau_S)
        S_acc = (1.0 - lam) * S_acc + lam * S_input

        gaussian_nll += 0.5 * (S_obs[t] - S_acc) ** 2

        p = np.clip(
            1.0 / (1.0 + np.exp(-gamma_sig * (S_acc - theta_t))),
            1e-9,
            1 - 1e-9,
        )
        bernoulli_nll -= ign[t] * np.log(p) + (1 - ign[t]) * np.log1p(-p)

        dtheta = (
            -lambda_theta * (theta_t - theta_0)
            + kappa_meta * C_t[t]
            - delta_info * I_t[t]
        )
        theta_t = theta_t + dtheta

    return float(gaussian_nll + bernoulli_nll)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def recover_parameters(
    data: dict,
    lambda_theta: float = LAMBDA_THETA_DEFAULT,
    kappa_meta: float = KAPPA_META_DEFAULT,
    delta_info: float = DELTA_INFO_DEFAULT,
    n_restarts: int = 8,
    seed: int = 99,
) -> dict:
    """Recover [θ₀, τ_S, Πⁱ_baseline, β_SM, γ_sig] via MLE with restarts.

    Returns:
        dict with keys: theta_0_hat, tau_S_hat, pi_i_hat, beta_sm_hat,
        gamma_sig_hat, nll, converged, converged_residual.
    """
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_restarts):
        x0 = np.array(
            [
                rng.uniform(0.3, 0.8),  # theta_0
                rng.uniform(1.0, 15.0),  # tau_S
                rng.uniform(0.5, 3.0),  # pi_i_baseline
                rng.uniform(0.1, 1.5),  # beta_sm
                rng.uniform(2.0, 8.0),  # gamma_sig
            ]
        )
        result = minimize(
            _negative_log_likelihood,
            x0=x0,
            args=(data, lambda_theta, kappa_meta, delta_info),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 20_000},
        )
        if best is None or result.fun < best.fun:
            best = result

    theta_0_hat, tau_S_hat, pi_i_hat, beta_sm_hat, gamma_sig_hat = best.x  # type: ignore[union-attr]
    nll_val = float(best.fun)  # type: ignore[union-attr]
    n_trials = len(data["pi_e"])
    return {
        "theta_0_hat": float(theta_0_hat),
        "tau_S_hat": float(tau_S_hat),
        "pi_i_hat": float(pi_i_hat),
        "beta_sm_hat": float(beta_sm_hat),
        "gamma_sig_hat": float(gamma_sig_hat),
        "nll": nll_val,
        "converged": bool(best.success),  # type: ignore[union-attr]
        "converged_residual": bool((nll_val / max(n_trials, 1)) < 3.0),
    }


# ---------------------------------------------------------------------------
# Full simulation (Figure 2 / Table 13)
# ---------------------------------------------------------------------------


def run_recovery_simulation(
    n_simulations: int = 50,
    n_trials_per_sim: int = 300,
    seed: int = 2024,
) -> dict:
    """Run the full parameter recovery simulation (Appendix A.4, Table 13).

    True parameter ranges follow Table 12 physiological constraints:
        θ₀    ∈ [0.35, 0.70]
        τ_S   ∈ [2.0, 12.0] steps
        Πⁱ    ∈ [0.8, 2.5]
        β_SM  ∈ [0.2, 1.2]
        γ_sig ∈ [2.5, 7.5]

    Returns:
        dict with *_true, *_hat lists and r_* Pearson correlations for all
        five parameters.
    """
    rng = np.random.default_rng(seed)

    theta_0_true_all = rng.uniform(0.35, 0.70, n_simulations)
    tau_S_true_all = rng.uniform(2.0, 12.0, n_simulations)
    pi_i_true_all = rng.uniform(0.8, 2.5, n_simulations)
    beta_sm_true_all = rng.uniform(0.2, 1.2, n_simulations)
    gamma_sig_true_all = rng.uniform(2.5, 7.5, n_simulations)

    theta_0_hat_all = np.empty(n_simulations)
    tau_S_hat_all = np.empty(n_simulations)
    pi_i_hat_all = np.empty(n_simulations)
    beta_sm_hat_all = np.empty(n_simulations)
    gamma_sig_hat_all = np.empty(n_simulations)
    converged_all = np.zeros(n_simulations, dtype=bool)
    converged_residual_all = np.zeros(n_simulations, dtype=bool)

    for i in range(n_simulations):
        data = generate_synthetic_data(
            n_trials=n_trials_per_sim,
            theta_0_true=float(theta_0_true_all[i]),
            tau_S_true=float(tau_S_true_all[i]),
            pi_i_true=float(pi_i_true_all[i]),
            beta_sm_true=float(beta_sm_true_all[i]),
            gamma_sig_true=float(gamma_sig_true_all[i]),
            seed=int(rng.integers(0, 2**31)),
        )
        recovered = recover_parameters(data)
        theta_0_hat_all[i] = recovered["theta_0_hat"]
        tau_S_hat_all[i] = recovered["tau_S_hat"]
        pi_i_hat_all[i] = recovered["pi_i_hat"]
        beta_sm_hat_all[i] = recovered["beta_sm_hat"]
        gamma_sig_hat_all[i] = recovered["gamma_sig_hat"]
        converged_all[i] = recovered["converged"]
        converged_residual_all[i] = recovered["converged_residual"]

    r_theta_0, _ = pearsonr(theta_0_true_all, theta_0_hat_all)
    r_tau_S, _ = pearsonr(tau_S_true_all, tau_S_hat_all)
    r_pi_i, _ = pearsonr(pi_i_true_all, pi_i_hat_all)
    r_beta_sm, _ = pearsonr(beta_sm_true_all, beta_sm_hat_all)
    r_gamma_sig, _ = pearsonr(gamma_sig_true_all, gamma_sig_hat_all)

    return {
        "r_theta_0": float(r_theta_0),
        "r_tau_S": float(r_tau_S),
        "r_pi_i": float(r_pi_i),
        "r_beta_sm": float(r_beta_sm),
        "r_gamma_sig": float(r_gamma_sig),
        "theta_0_true": theta_0_true_all.tolist(),
        "theta_0_hat": theta_0_hat_all.tolist(),
        "tau_S_true": tau_S_true_all.tolist(),
        "tau_S_hat": tau_S_hat_all.tolist(),
        "pi_i_true": pi_i_true_all.tolist(),
        "pi_i_hat": pi_i_hat_all.tolist(),
        "beta_sm_true": beta_sm_true_all.tolist(),
        "beta_sm_hat": beta_sm_hat_all.tolist(),
        "gamma_sig_true": gamma_sig_true_all.tolist(),
        "gamma_sig_hat": gamma_sig_hat_all.tolist(),
        "converged": converged_all.tolist(),
        "converged_residual": converged_residual_all.tolist(),
    }
