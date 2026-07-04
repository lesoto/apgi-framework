"""APGICoreIntegration — stateful session-level integration of APGI computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apgi.core import (
    BETA_SM_DEFAULT,
    DELTA_INFO_DEFAULT,
    DELTA_RESET_DEFAULT,
    ETA_NE_DEFAULT,
    GAMMA_SIG_DEFAULT,
    KAPPA_META_DEFAULT,
    LAMBDA_THETA_DEFAULT,
    RHO_RETAIN_DEFAULT,
    TAU_S_DEFAULT,
    THETA_0_DEFAULT,
    accumulate_S_t,
    compute_pi_i_eff,
    compute_S_t,
    ignition_criterion,
    post_ignition_reset,
    step_theta,
    theta_equilibrium,
)


@dataclass
class TrialRecord:
    """Immutable record of a single integrated trial."""

    t: int
    pi_i_eff: float
    S_t: float
    theta_t: float
    ignition: bool
    C_t: float
    I_t: float


class APGICoreIntegration:
    """Stateful integrator that runs APGI trials and tracks session history.

    Implements the full §4.1 equations across a trial sequence, maintaining
    a running S_t leaky accumulator and evolving θₜ via the ODE stepper.

    Parameters
    ----------
    beta_sm : float
        Somatic-marker gain β_SM (§4.2).
    M_hat : float
        Fixed somatic marker estimate for the session (use 0 for neutral).
    theta_0 : float
        Homeostatic threshold baseline θ₀.
    lambda_theta : float
        Mean-reversion rate λθ.
    kappa_meta : float
        Metabolic cost coefficient κ_meta.
    delta_info : float
        Information value coefficient Δ_info.
    eta_NE : float
        Noradrenaline modulation coefficient η_NE.
    gamma_sig : float
        Sigmoid steepness γ_sig.
    tau_S : float
        Surprise accumulation timescale τ_S (trial steps).
    theta_init : float or None
        Initial threshold.  If None, initialised at equilibrium.
    """

    def __init__(
        self,
        beta_sm: float = BETA_SM_DEFAULT,
        M_hat: float = 0.0,
        theta_0: float = THETA_0_DEFAULT,
        lambda_theta: float = LAMBDA_THETA_DEFAULT,
        kappa_meta: float = KAPPA_META_DEFAULT,
        delta_info: float = DELTA_INFO_DEFAULT,
        eta_NE: float = ETA_NE_DEFAULT,
        gamma_sig: float = GAMMA_SIG_DEFAULT,
        tau_S: float = TAU_S_DEFAULT,
        delta_reset: float = DELTA_RESET_DEFAULT,
        rho_retain: float = RHO_RETAIN_DEFAULT,
        theta_init: float | None = None,
    ) -> None:
        self.beta_sm = beta_sm
        self.M_hat = M_hat
        self.theta_0 = theta_0
        self.lambda_theta = lambda_theta
        self.kappa_meta = kappa_meta
        self.delta_info = delta_info
        self.eta_NE = eta_NE
        self.gamma_sig = gamma_sig
        self.tau_S = tau_S
        self.delta_reset = delta_reset
        self.rho_retain = rho_retain
        self._S_acc: float = 0.0
        self._theta: float | None = theta_init
        self._records: list[TrialRecord] = []

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def step(
        self,
        pi_e: float,
        z_e: float,
        pi_i_baseline: float,
        z_i: float,
        C_t: float,
        I_t: float,
        M_hat: float | None = None,
        NE_t: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> TrialRecord:
        """Integrate one trial and update internal state.

        Args:
            pi_e: Exteroceptive precision Πᵉ.
            z_e: Exteroceptive prediction error.
            pi_i_baseline: Baseline interoceptive precision Πⁱ_baseline.
            z_i: Interoceptive prediction error.
            C_t: Metabolic cost at this trial.
            I_t: Information value at this trial.
            M_hat: Per-trial somatic marker (overrides session default if given).
            NE_t: Noradrenaline drive.
            rng: RNG for stochastic ignition; None → deterministic (P≥0.5).

        Returns:
            TrialRecord for this trial.
        """
        m = M_hat if M_hat is not None else self.M_hat
        pi_i_eff = compute_pi_i_eff(pi_i_baseline, self.beta_sm, m)
        S_input = compute_S_t(pi_e, z_e, pi_i_eff, z_i)
        self._S_acc = accumulate_S_t(self._S_acc, S_input, tau_S=self.tau_S)

        if self._theta is None:
            self._theta = theta_equilibrium(
                C_t,
                I_t,
                theta_0=self.theta_0,
                lambda_theta=self.lambda_theta,
                kappa_meta=self.kappa_meta,
                delta_info=self.delta_info,
            )

        theta_t = self._theta
        fired = ignition_criterion(self._S_acc, theta_t, self.gamma_sig, rng=rng)
        t = len(self._records)
        record = TrialRecord(
            t=t,
            pi_i_eff=pi_i_eff,
            S_t=self._S_acc,
            theta_t=theta_t,
            ignition=fired,
            C_t=C_t,
            I_t=I_t,
        )
        self._records.append(record)
        self._theta = step_theta(
            theta_t,
            C_t,
            I_t,
            NE_t=NE_t,
            fired=fired,
            theta_0=self.theta_0,
            lambda_theta=self.lambda_theta,
            kappa_meta=self.kappa_meta,
            delta_info=self.delta_info,
            eta_NE=self.eta_NE,
            delta_reset=self.delta_reset,
        )
        self._S_acc = post_ignition_reset(self._S_acc, fired, rho_retain=self.rho_retain)
        return record

    def run_sequence(
        self,
        pi_e: NDArray,
        z_e: NDArray,
        pi_i: NDArray,
        z_i: NDArray,
        C_t: NDArray,
        I_t: NDArray,
        M_hat: NDArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[TrialRecord]:
        """Run a sequence of trials from equal-length arrays.

        Returns the list of TrialRecord objects appended in this call.
        """
        arrays = [pi_e, z_e, pi_i, z_i, C_t, I_t]
        length = len(arrays[0])
        if any(len(a) != length for a in arrays):
            raise ValueError("All input arrays must have the same length")

        return [
            self.step(
                float(pi_e[i]),
                float(z_e[i]),
                float(pi_i[i]),
                float(z_i[i]),
                float(C_t[i]),
                float(I_t[i]),
                M_hat=float(M_hat[i]) if M_hat is not None else None,
                rng=rng,
            )
            for i in range(length)
        ]

    # ------------------------------------------------------------------
    # Session summaries
    # ------------------------------------------------------------------

    def ignition_rate(self) -> float:
        """Proportion of trials in which ignition fired."""
        if not self._records:
            return float("nan")
        return sum(r.ignition for r in self._records) / len(self._records)

    def mean_S_t(self) -> float:
        """Mean global integration signal across all recorded trials."""
        if not self._records:
            return float("nan")
        return float(np.mean([r.S_t for r in self._records]))

    def mean_theta(self) -> float:
        """Mean ignition threshold across all recorded trials."""
        if not self._records:
            return float("nan")
        return float(np.mean([r.theta_t for r in self._records]))

    @property
    def records(self) -> list[TrialRecord]:
        """Read-only view of accumulated trial records."""
        return list(self._records)

    def reset(self) -> None:
        """Clear session history and reset θₜ to its initial value."""
        self._records.clear()
        self._theta = None
