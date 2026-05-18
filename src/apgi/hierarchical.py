"""Five-level hierarchical architecture — Paper 3.

Implements the APGI hierarchy:

    Level 1 — Sensory encoding
    Level 2 — Local integration
    Level 3 — Regional coordination
    Level 4 — Global workspace
    Level 5 — Metacognitive monitoring

Each level computes a precision-weighted prediction error and passes
both bottom-up signals (prediction errors) and top-down signals
(predictions) to adjacent levels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from apgi.apgi_core import compute_pi_i_eff, compute_S_t


class HierarchicalLevel:
    """Single level of the APGI hierarchy.

    Parameters
    ----------
    level_id : int
        Level index (1–5).
    n_units : int
        Number of representational units.
    pi_e : float
        Excitatory precision for this level.
    pi_i : float
        Raw inhibitory precision for this level.
    kappa : float
        Energy-information coupling constant.
    """

    def __init__(
        self,
        level_id: int,
        n_units: int,
        pi_e: float = 1.0,
        pi_i: float = 1.0,
        kappa: float = 100.0,
    ) -> None:
        self.level_id = level_id
        self.n_units = n_units
        self.pi_e = pi_e
        self.pi_i = pi_i
        self.kappa = kappa

        self.state = np.zeros(n_units)
        self.prediction = np.zeros(n_units)

    def update(
        self,
        bottom_up: NDArray,
        top_down: NDArray,
        C_metabolic: float,
    ) -> tuple[NDArray, float]:
        """Update level state and return prediction error and Sₜ contribution.

        Args:
            bottom_up: Signal from the level below (or sensory input at L1).
            top_down: Prediction from the level above (zeros at L5).
            C_metabolic: Current metabolic cost.

        Returns:
            Tuple of (prediction_error, S_t_contribution).
        """
        prediction_error = bottom_up - top_down
        self.state = self.state + 0.1 * (prediction_error - self.state)
        self.prediction = np.tanh(self.state)

        pi_i_eff = compute_pi_i_eff(self.pi_i, C_metabolic, self.kappa)
        z_e = float(np.mean(np.abs(prediction_error)))
        z_i = float(np.mean(np.abs(self.state)))
        S_t_contribution = compute_S_t(self.pi_e, z_e, pi_i_eff, z_i)

        return prediction_error, S_t_contribution


class APGIHierarchy:
    """Five-level APGI hierarchical architecture.

    Instantiates levels 1–5 with progressively fewer units (coarser
    representations) and higher precision weights, reflecting the
    empirical organisation of cortical hierarchy.

    Parameters
    ----------
    n_sensory : int
        Number of sensory input units (Level 1 width).
    kappa : float
        Energy-information coupling constant shared across all levels.
    """

    LEVEL_CONFIGS: list[dict] = [
        {"pi_e": 0.5, "pi_i": 0.8},  # L1 — Sensory encoding
        {"pi_e": 0.8, "pi_i": 0.9},  # L2 — Local integration
        {"pi_e": 1.0, "pi_i": 1.0},  # L3 — Regional coordination
        {"pi_e": 1.2, "pi_i": 1.1},  # L4 — Global workspace
        {"pi_e": 1.5, "pi_i": 1.2},  # L5 — Metacognitive monitoring
    ]

    def __init__(self, n_sensory: int = 64, kappa: float = 100.0) -> None:
        widths = [
            n_sensory,
            n_sensory // 2,
            n_sensory // 4,
            n_sensory // 8,
            n_sensory // 16,
        ]
        widths = [max(w, 1) for w in widths]

        self.levels: list[HierarchicalLevel] = [
            HierarchicalLevel(
                level_id=i + 1,
                n_units=widths[i],
                pi_e=cfg["pi_e"],
                pi_i=cfg["pi_i"],
                kappa=kappa,
            )
            for i, cfg in enumerate(self.LEVEL_CONFIGS)
        ]

    def forward(self, sensory_input: NDArray, C_metabolic: float) -> dict:
        """Run one forward pass through all five levels.

        Args:
            sensory_input: Array of shape (n_sensory,).
            C_metabolic: Current metabolic cost signal.

        Returns:
            dict with keys: S_t_total, level_S_t, level_errors, predictions.
        """
        n = len(self.levels)
        bottom_up_signals: list[NDArray] = [None] * n  # type: ignore[list-item]
        top_down_signals: list[NDArray] = [None] * n  # type: ignore[list-item]

        # Resize sensory input to L1 width via averaging if needed
        l1_width = self.levels[0].n_units
        if len(sensory_input) != l1_width:
            sensory_input = np.interp(
                np.linspace(0, 1, l1_width),
                np.linspace(0, 1, len(sensory_input)),
                sensory_input,
            )
        bottom_up_signals[0] = sensory_input

        # Bottom-up pass: propagate signals upward, each level's prediction
        # becomes the bottom-up signal for the next
        for i in range(1, n):
            src = self.levels[i - 1]
            tgt_width = self.levels[i].n_units
            src_pred = src.prediction
            bottom_up_signals[i] = np.interp(
                np.linspace(0, 1, tgt_width),
                np.linspace(0, 1, len(src_pred)),
                src_pred,
            )

        # Top-down pass: highest level receives no top-down signal
        top_down_signals[n - 1] = np.zeros(self.levels[n - 1].n_units)
        for i in range(n - 2, -1, -1):
            src = self.levels[i + 1]
            tgt_width = self.levels[i].n_units
            src_pred = src.prediction
            top_down_signals[i] = np.interp(
                np.linspace(0, 1, tgt_width),
                np.linspace(0, 1, len(src_pred)),
                src_pred,
            )

        # Update all levels
        level_errors: list[NDArray] = []
        level_S_t: list[float] = []
        for i, level in enumerate(self.levels):
            err, s = level.update(
                bottom_up_signals[i], top_down_signals[i], C_metabolic
            )
            level_errors.append(err)
            level_S_t.append(s)

        return {
            "S_t_total": float(np.sum(level_S_t)),
            "level_S_t": level_S_t,
            "level_errors": level_errors,
            "predictions": [lv.prediction.copy() for lv in self.levels],
        }

    def reset(self) -> None:
        """Reset all level states to zero."""
        for level in self.levels:
            level.state[:] = 0.0
            level.prediction[:] = 0.0
