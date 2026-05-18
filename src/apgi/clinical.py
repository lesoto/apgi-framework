"""EnhancedClinicalInterpreter — maps APGI signals to clinical consciousness indices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ConsciousnessLevel(str, Enum):
    """Ordinal consciousness level labels derived from APGI signal ratios."""

    UNRESPONSIVE = "unresponsive"
    MINIMALLY_CONSCIOUS = "minimally_conscious"
    CONSCIOUS = "conscious"
    HYPER_ALERT = "hyper_alert"


@dataclass(frozen=True)
class ClinicalReport:
    """Summary clinical report for a single observation window."""

    level: ConsciousnessLevel
    ignition_index: float  # mean S_t / mean theta_t
    ignition_rate: float  # fraction of windows with ignition
    mean_S_t: float
    mean_theta_t: float
    somatic_load: float  # mean C_metabolic


class EnhancedClinicalInterpreter:
    """Interprets APGI outputs in clinical and anaesthesia-monitoring contexts.

    The interpreter accepts windowed arrays of Sₜ, θₜ, and C_metabolic,
    computes a composite ignition index (Sₜ / θₜ), and maps it onto a
    four-level consciousness scale calibrated against the literature
    (Northoff & Huang 2017; Mashour et al. 2020).

    Parameters
    ----------
    window_size : int
        Number of trials per analysis window.
    thresholds : dict or None
        Custom ignition-index thresholds for level boundaries.
        Keys: "minimally_conscious", "conscious", "hyper_alert".
        Defaults to empirically motivated values.
    """

    _DEFAULT_THRESHOLDS: dict[str, float] = {
        "minimally_conscious": 0.4,
        "conscious": 0.75,
        "hyper_alert": 1.3,
    }

    def __init__(
        self,
        window_size: int = 20,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.thresholds = thresholds or dict(self._DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    # Core interpretation
    # ------------------------------------------------------------------

    def interpret(
        self,
        S_t: NDArray,
        theta_t: NDArray,
        C_metabolic: NDArray,
    ) -> ClinicalReport:
        """Generate a clinical report for one observation window.

        Args:
            S_t: Array of global integration signal values.
            theta_t: Array of ignition threshold values (same length).
            C_metabolic: Array of metabolic cost values (same length).

        Returns:
            ClinicalReport dataclass.
        """
        S_t = np.asarray(S_t, dtype=float)
        theta_t = np.asarray(theta_t, dtype=float)
        C_metabolic = np.asarray(C_metabolic, dtype=float)

        if not (S_t.shape == theta_t.shape == C_metabolic.shape):
            raise ValueError("S_t, theta_t, and C_metabolic must have the same shape")

        mean_S = float(np.mean(S_t))
        mean_theta = float(np.mean(theta_t))
        mean_C = float(np.mean(C_metabolic))
        ignition_rate = float(np.mean(S_t >= theta_t))
        ignition_index = mean_S / (mean_theta + 1e-8)

        level = self._classify(ignition_index)
        return ClinicalReport(
            level=level,
            ignition_index=ignition_index,
            ignition_rate=ignition_rate,
            mean_S_t=mean_S,
            mean_theta_t=mean_theta,
            somatic_load=mean_C,
        )

    def interpret_session(
        self,
        S_t: NDArray,
        theta_t: NDArray,
        C_metabolic: NDArray,
    ) -> list[ClinicalReport]:
        """Slide a window over a full session and return per-window reports.

        Trailing samples that do not fill a complete window are discarded.
        """
        S_t = np.asarray(S_t, dtype=float)
        theta_t = np.asarray(theta_t, dtype=float)
        C_metabolic = np.asarray(C_metabolic, dtype=float)

        n = len(S_t)
        reports = []
        for start in range(0, n - self.window_size + 1, self.window_size):
            end = start + self.window_size
            reports.append(
                self.interpret(
                    S_t[start:end], theta_t[start:end], C_metabolic[start:end]
                )
            )
        return reports

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, ignition_index: float) -> ConsciousnessLevel:
        t = self.thresholds
        if ignition_index < t["minimally_conscious"]:
            return ConsciousnessLevel.UNRESPONSIVE
        if ignition_index < t["conscious"]:
            return ConsciousnessLevel.MINIMALLY_CONSCIOUS
        if ignition_index < t["hyper_alert"]:
            return ConsciousnessLevel.CONSCIOUS
        return ConsciousnessLevel.HYPER_ALERT
