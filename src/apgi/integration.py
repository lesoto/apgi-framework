"""APGICoreIntegration — stateful session-level integration of APGI computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apgi.core import (
    compute_pi_i_eff,
    compute_S_t,
    compute_theta_t,
    ignition_criterion,
    update_theta,
)


@dataclass
class TrialRecord:
    """Immutable record of a single integrated trial."""

    t: int
    pi_i_eff: float
    S_t: float
    theta_t: float
    ignition: bool
    C_metabolic: float
    V_information: float


class APGICoreIntegration:
    """Stateful integrator that runs APGI trials and tracks session history.

    Maintains a rolling estimate of θₜ across trials using exponential
    smoothing, accumulates trial records, and exposes aggregate summaries
    (ignition rate, mean Sₜ, etc.) without requiring the caller to manage
    state manually.

    Parameters
    ----------
    alpha : float
        Metabolic weighting coefficient α.
    beta : float
        Information weighting coefficient β.
    kappa : float
        Energy-information coupling constant κ.
    gamma : float
        Smoothing factor for the θₜ update rule.
    theta_init : float or None
        Initial threshold.  If None, the threshold for the first trial is
        computed directly from α, β, C, V without smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        kappa: float = 100.0,
        gamma: float = 0.9,
        theta_init: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.gamma = gamma
        self._theta: float | None = theta_init
        self._records: list[TrialRecord] = []

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def step(
        self,
        pi_e: float,
        z_e: float,
        pi_i: float,
        z_i: float,
        C_metabolic: float,
        V_information: float,
    ) -> TrialRecord:
        """Integrate one trial and update internal state.

        Args:
            pi_e: Excitatory precision Πᵉ.
            z_e: Excitatory state variable.
            pi_i: Raw inhibitory precision Πⁱ.
            z_i: Inhibitory state variable.
            C_metabolic: Metabolic cost at this trial.
            V_information: Information value at this trial.

        Returns:
            TrialRecord for this trial.
        """
        pi_i_eff = compute_pi_i_eff(pi_i, C_metabolic, self.kappa)
        S_t = compute_S_t(pi_e, z_e, pi_i_eff, z_i)

        if self._theta is None:
            theta_t = compute_theta_t(C_metabolic, V_information, self.alpha, self.beta)
        else:
            theta_t = self._theta

        fired = ignition_criterion(S_t, theta_t)
        t = len(self._records)
        record = TrialRecord(
            t=t,
            pi_i_eff=pi_i_eff,
            S_t=S_t,
            theta_t=theta_t,
            ignition=fired,
            C_metabolic=C_metabolic,
            V_information=V_information,
        )
        self._records.append(record)
        self._theta = update_theta(
            theta_t, C_metabolic, V_information, self.alpha, self.beta, self.gamma
        )
        return record

    def run_sequence(
        self,
        pi_e: NDArray,
        z_e: NDArray,
        pi_i: NDArray,
        z_i: NDArray,
        C_metabolic: NDArray,
        V_information: NDArray,
    ) -> list[TrialRecord]:
        """Run a sequence of trials from equal-length arrays.

        Returns the list of TrialRecord objects appended in this call.
        """
        arrays = [pi_e, z_e, pi_i, z_i, C_metabolic, V_information]
        length = len(arrays[0])
        if any(len(a) != length for a in arrays):
            raise ValueError("All input arrays must have the same length")

        return [
            self.step(
                float(pi_e[i]),
                float(z_e[i]),
                float(pi_i[i]),
                float(z_i[i]),
                float(C_metabolic[i]),
                float(V_information[i]),
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
