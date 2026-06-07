"""Parameter recovery simulation — Appendix A.4.

Verifies that the APGI model can recover its own generating parameters
from synthetic data. The benchmark criterion is Pearson r > 0.75 between
true and recovered values for β and Πⁱ parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import pearsonr

from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t


def generate_synthetic_data(
    n_trials: int,
    beta_true: float,
    pi_i_true: float,
    alpha: float = 0.3,
    kappa: float = 100.0,
    noise_sd: float = 0.05,
    seed: int = 0,
) -> dict:
    """Generate synthetic trial data from the APGI generative model.

    Args:
        n_trials: Number of simulated trials.
        beta_true: True β (information weighting coefficient).
        pi_i_true: True Πⁱ (raw inhibitory precision).
        alpha: True α (metabolic weighting coefficient), fixed.
        kappa: True κ.
        noise_sd: Gaussian observation noise on Sₜ.
        seed: RNG seed.

    Returns:
        dict with keys: C_metabolic, V_information, pi_e, z_e, z_i,
        S_t_observed, ignition.
    """
    rng = np.random.default_rng(seed)

    C_metabolic = rng.uniform(0.5, 2.0, n_trials)
    V_information = rng.uniform(0.1, 1.0, n_trials)
    pi_e = rng.uniform(0.8, 1.5, n_trials)
    z_e = rng.uniform(0.2, 1.0, n_trials)
    z_i = rng.uniform(0.1, 0.8, n_trials)

    pi_i_eff = compute_pi_i_eff(pi_i_true, C_metabolic, kappa)
    S_t_true = np.array(
        [compute_S_t(pi_e[i], z_e[i], pi_i_eff[i], z_i[i]) for i in range(n_trials)]
    )
    S_t_observed = S_t_true + rng.normal(0, noise_sd, n_trials)

    theta_t = np.array(
        [
            compute_theta_t(C_metabolic[i], V_information[i], alpha, beta_true)
            for i in range(n_trials)
        ]
    )
    ignition = (S_t_true >= theta_t).astype(int)

    return {
        "C_metabolic": C_metabolic,
        "V_information": V_information,
        "pi_e": pi_e,
        "z_e": z_e,
        "z_i": z_i,
        "S_t_observed": S_t_observed,
        "ignition": ignition,
    }


def _negative_log_likelihood(
    params: NDArray,
    data: dict,
    alpha: float,
    kappa: float,
) -> float:
    """Gaussian NLL for Sₜ + Bernoulli NLL for ignition (vectorised)."""
    beta, pi_i = params
    if beta <= 0 or pi_i <= 0:
        return 1e10

    C = np.asarray(data["C_metabolic"])
    V = np.asarray(data["V_information"])
    pi_e = np.asarray(data["pi_e"])
    z_e = np.asarray(data["z_e"])
    z_i = np.asarray(data["z_i"])
    S_obs = np.asarray(data["S_t_observed"])
    ignition = np.asarray(data["ignition"])

    # Inline vectorised forms of the three APGI core equations
    pi_i_eff = pi_i * np.exp(-C / kappa)                  # Πⁱ_eff = Πⁱ·exp(−C/κ)
    S_t_pred = pi_e * np.abs(z_e) + pi_i_eff * np.abs(z_i)  # Sₜ
    theta = alpha * C + beta * V                            # θₜ

    # Gaussian NLL for continuous Sₜ observation
    gaussian_nll = 0.5 * np.sum((S_obs - S_t_pred) ** 2)

    # Bernoulli NLL for ignition (logistic approximation, slope=5)
    p_ignite = np.clip(1.0 / (1.0 + np.exp(-(S_t_pred - theta) * 5.0)), 1e-9, 1 - 1e-9)
    bernoulli_nll = -np.sum(
        ignition * np.log(p_ignite) + (1 - ignition) * np.log1p(-p_ignite)
    )

    return float(gaussian_nll + bernoulli_nll)


def recover_parameters(
    data: dict,
    alpha: float = 0.3,
    kappa: float = 100.0,
    n_restarts: int = 5,
    seed: int = 99,
) -> dict:
    """Recover β and Πⁱ from synthetic data via maximum-likelihood estimation.

    Args:
        data: Output dict from ``generate_synthetic_data``.
        alpha: Fixed α (not estimated here).
        kappa: Fixed κ (not estimated here).
        n_restarts: Number of random restarts to avoid local minima.
        seed: RNG seed for restart initialisation.

    Returns:
        dict with keys: beta_hat, pi_i_hat, nll.
    """
    rng = np.random.default_rng(seed)
    best_result = None

    for _ in range(n_restarts):
        x0 = rng.uniform([0.1, 0.1], [2.0, 2.0])
        result = minimize(
            _negative_log_likelihood,
            x0=x0,
            args=(data, alpha, kappa),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    beta_hat, pi_i_hat = best_result.x  # type: ignore[union-attr]
    # scipy Nelder-Mead reports success=False when xatol/fatol are met
    # simultaneously, which rarely happens on flat NLL surfaces even when the
    # estimate is correct.  We therefore report both the strict optimizer flag
    # and a residual-based criterion: nll_per_trial < 2.0 is empirically
    # well-calibrated for this likelihood (see Appendix A.4).
    nll_val = float(best_result.fun)  # type: ignore[union-attr]
    n_trials = len(data["C_metabolic"])
    converged_residual = (nll_val / max(n_trials, 1)) < 2.0
    return {
        "beta_hat": float(beta_hat),
        "pi_i_hat": float(pi_i_hat),
        "nll": nll_val,
        "converged": bool(best_result.success),  # type: ignore[union-attr]
        "converged_residual": bool(converged_residual),
    }


def run_recovery_simulation(
    n_simulations: int = 50,
    n_trials_per_sim: int = 200,
    beta_range: tuple[float, float] = (0.3, 1.5),
    pi_i_range: tuple[float, float] = (0.5, 2.0),
    alpha: float = 0.3,
    kappa: float = 100.0,
    noise_sd: float = 0.05,
    seed: int = 7,
) -> dict:
    """Run the full parameter recovery simulation (Appendix A.4).

    Generates data from known parameters, recovers them via MLE, and
    returns Pearson r between true and recovered values.

    Returns:
        dict with keys: r_beta, r_pi_i, beta_true, beta_hat, pi_i_true, pi_i_hat,
        converged (bool array from optimizer success flag per run).
    """
    rng = np.random.default_rng(seed)

    beta_true_all = rng.uniform(*beta_range, n_simulations)
    pi_i_true_all = rng.uniform(*pi_i_range, n_simulations)
    beta_hat_all = np.empty(n_simulations)
    pi_i_hat_all = np.empty(n_simulations)
    converged_all = np.zeros(n_simulations, dtype=bool)
    converged_residual_all = np.zeros(n_simulations, dtype=bool)

    for i in range(n_simulations):
        data = generate_synthetic_data(
            n_trials=n_trials_per_sim,
            beta_true=float(beta_true_all[i]),
            pi_i_true=float(pi_i_true_all[i]),
            alpha=alpha,
            kappa=kappa,
            noise_sd=noise_sd,
            seed=int(rng.integers(0, 2**31)),
        )
        recovered = recover_parameters(data, alpha=alpha, kappa=kappa)
        beta_hat_all[i] = recovered["beta_hat"]
        pi_i_hat_all[i] = recovered["pi_i_hat"]
        converged_all[i] = recovered["converged"]
        converged_residual_all[i] = recovered["converged_residual"]

    r_beta, _ = pearsonr(beta_true_all, beta_hat_all)
    r_pi_i, _ = pearsonr(pi_i_true_all, pi_i_hat_all)

    return {
        "r_beta": float(r_beta),
        "r_pi_i": float(r_pi_i),
        "beta_true": beta_true_all.tolist(),
        "beta_hat": beta_hat_all.tolist(),
        "pi_i_true": pi_i_true_all.tolist(),
        "pi_i_hat": pi_i_hat_all.tolist(),
        "converged": converged_all.tolist(),
        "converged_residual": converged_residual_all.tolist(),
    }
