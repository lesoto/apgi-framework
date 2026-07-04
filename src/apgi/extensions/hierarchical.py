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

Per-level threshold dynamics follow Paper 3 §3.4's cross-level coupled
threshold ODE:

    dθ_k/dt = −(θ_k − θ_k*)/τ_θ,k + κ_{k,k+1}·(θ_{k+1} − θ_{k+1}*) + β_k·S_k + η_k·NE_k(t)

Each level also carries the three dissociable timescales the paper insists
must not be conflated (§2.2): τ_int,ℓ (evidence-accumulation window),
τ_θ,ℓ (threshold-recovery/refractory timescale), and τ_ign,ℓ (broadcast
persistence). Note: the paper itself flags κ_{k,k+1} as "not yet
experimentally constrained" and the full stochastic (Ornstein-Uhlenbeck)
threshold process with 1/f-spectrum validation as an open item requiring
dedicated protocol-level estimation (§6.4) — this module implements the
deterministic coupled-ODE skeleton the paper specifies, not the full
empirical OU/DFA validation pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from apgi.core import compute_pi_i_eff, compute_S_t


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
    theta_star : float
        Homeostatic threshold set-point θ_k* for this level.
    tau_int : float
        Evidence-accumulation integration window τ_int,ℓ (Paper 3 §2.2).
    tau_theta : float
        Threshold-recovery/refractory timescale τ_θ,ℓ.
    tau_ign : float
        Broadcast-persistence timescale τ_ign,ℓ (reported, not simulated
        directly by this class).
    kappa_coupling : float
        Cross-level coupling strength κ_{k,k+1} onto the level above
        (Paper 3 §3.4). Canonical range [0.1, 0.5]; the paper notes this
        constant is not yet experimentally constrained.
    beta_k : float
        Bottom-up ignition-drive coefficient β_k on θ_k.
    eta_k : float
        Phasic-NE drive coefficient η_k on θ_k.
    """

    def __init__(
        self,
        level_id: int,
        n_units: int,
        pi_e: float = 1.0,
        pi_i: float = 1.0,
        theta_star: float = 0.6,
        tau_int: float = 10.0,
        tau_theta: float = 5.0,
        tau_ign: float = 2.0,
        kappa_coupling: float = 0.3,
        beta_k: float = 0.1,
        eta_k: float = 0.05,
    ) -> None:
        self.level_id = level_id
        self.n_units = n_units
        self.pi_e = pi_e
        self.pi_i = pi_i
        self.theta_star = theta_star
        self.tau_int = tau_int
        self.tau_theta = tau_theta
        self.tau_ign = tau_ign
        self.kappa_coupling = kappa_coupling
        self.beta_k = beta_k
        self.eta_k = eta_k

        self.theta = theta_star
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
        self.state = self.state + (1.0 / self.tau_int) * (prediction_error - self.state)
        self.prediction = np.tanh(self.state)

        # Hierarchical levels have no somatic-marker drive (M̂=0, β_SM=0):
        # precision is unmodulated at the sensory hierarchy level.
        pi_i_eff = compute_pi_i_eff(self.pi_i, beta_sm=0.0, M_hat=0.0)
        z_e = float(np.mean(np.abs(prediction_error)))
        z_i = float(np.mean(np.abs(self.state)))
        S_t_contribution = compute_S_t(self.pi_e, z_e, pi_i_eff, z_i)

        return prediction_error, S_t_contribution

    def step_theta(
        self,
        theta_next: float,
        theta_star_next: float,
        S_k: float,
        NE_k: float = 0.0,
        dt: float = 1.0,
    ) -> float:
        """One Euler step of the cross-level coupled threshold ODE.

        dθ_k/dt = −(θ_k − θ_k*)/τ_θ,k + κ_{k,k+1}·(θ_{k+1} − θ_{k+1}*)
                  + β_k·S_k + η_k·NE_k(t)

        Paper 3 §3.4. Pass theta_next=theta_star_next=0.0 for the top level
        (no level above to couple to).

        Args:
            theta_next: Current threshold θ_{k+1} of the level above.
            theta_star_next: Set-point θ_{k+1}* of the level above.
            S_k: This level's ignition/integration signal driving adaptation.
            NE_k: Phasic noradrenaline drive at this level.
            dt: Integration step size.

        Returns:
            Updated threshold θ_k.
        """
        dtheta = (
            -(self.theta - self.theta_star) / self.tau_theta
            + self.kappa_coupling * (theta_next - theta_star_next)
            + self.beta_k * S_k
            + self.eta_k * NE_k
        )
        self.theta = self.theta + dt * dtheta
        return self.theta


class APGIHierarchy:
    """Five-level APGI hierarchical architecture.

    Instantiates levels 1–5 with progressively fewer units (coarser
    representations) and higher precision weights, reflecting the
    empirical organisation of cortical hierarchy.

    Parameters
    ----------
    n_sensory : int
        Number of sensory input units (Level 1 width). Must be ≥ 16 to
        avoid degenerate single-unit levels at the top of the hierarchy.

    Examples
    --------
    Single forward pass through the five-level hierarchy:

    >>> import numpy as np
    >>> hier = APGIHierarchy(n_sensory=64)
    >>> sensory = np.random.default_rng(0).uniform(0.0, 1.0, 64)
    >>> result = hier.forward(sensory, C_metabolic=1.0)
    >>> sorted(result.keys())
    ['S_t_total', 'level_S_t', 'level_errors', 'level_ignition', 'level_theta', 'predictions']
    >>> len(result["level_S_t"])
    5

    Simulating 50 trials and collecting total integration signal:

    >>> rng = np.random.default_rng(42)
    >>> hier.reset()
    >>> S_t_series = [
    ...     hier.forward(rng.uniform(0, 1, 64), C_metabolic=rng.uniform(0.5, 2.0))["S_t_total"]
    ...     for _ in range(50)
    ... ]
    >>> len(S_t_series)
    50
    """

    # Timescales follow Paper 3 §2.2's geometric spacing across levels; at
    # L1 the three timescales collapse toward the same order of magnitude
    # (spec), while at L2+ they separate with tau_int > tau_theta > tau_ign.
    LEVEL_CONFIGS: list[dict] = [
        {  # L1 — Sensory encoding
            "pi_e": 0.5, "pi_i": 0.8,
            "tau_int": 5.0, "tau_theta": 5.0, "tau_ign": 5.0,
            "theta_star": 0.5, "kappa_coupling": 0.3,
        },
        {  # L2 — Local integration
            "pi_e": 0.8, "pi_i": 0.9,
            "tau_int": 22.0, "tau_theta": 15.0, "tau_ign": 8.0,
            "theta_star": 0.55, "kappa_coupling": 0.3,
        },
        {  # L3 — Regional coordination
            "pi_e": 1.0, "pi_i": 1.0,
            "tau_int": 75.0, "tau_theta": 50.0, "tau_ign": 25.0,
            "theta_star": 0.6, "kappa_coupling": 0.3,
        },
        {  # L4 — Global workspace
            "pi_e": 1.2, "pi_i": 1.1,
            "tau_int": 225.0, "tau_theta": 150.0, "tau_ign": 75.0,
            "theta_star": 0.65, "kappa_coupling": 0.3,
        },
        {  # L5 — Metacognitive monitoring (top level: no level above to couple to)
            "pi_e": 1.5, "pi_i": 1.2,
            "tau_int": 675.0, "tau_theta": 450.0, "tau_ign": 225.0,
            "theta_star": 0.7, "kappa_coupling": 0.0,
        },
    ]

    def __init__(self, n_sensory: int = 64) -> None:
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
                theta_star=cfg["theta_star"],
                tau_int=cfg["tau_int"],
                tau_theta=cfg["tau_theta"],
                tau_ign=cfg["tau_ign"],
                kappa_coupling=cfg["kappa_coupling"],
            )
            for i, cfg in enumerate(self.LEVEL_CONFIGS)
        ]

    def forward(
        self,
        sensory_input: NDArray,
        C_metabolic: float,
        NE: float = 0.0,
        dt: float = 1.0,
    ) -> dict:
        """Run one forward pass through all five levels.

        Executes a bottom-up then top-down sweep: prediction errors
        propagate upward, predictions propagate downward, and each level
        contributes an Sₜ component to the total integration signal. Each
        level's threshold θ_ℓ is then updated via the cross-level coupled
        ODE (Paper 3 §3.4), using a simultaneous (Jacobi-style) update so
        all levels couple to each other's *pre-update* threshold.

        Args:
            sensory_input: Array of shape (n_sensory,). Values outside
                [−3, 3] will propagate but may saturate tanh activations.
            C_metabolic: Current metabolic cost signal (≥ 0).
            NE: Phasic noradrenaline drive shared across levels.
            dt: Integration step size for the threshold update.

        Returns:
            dict with keys:

            - ``S_t_total`` (float): Sum of Sₜ contributions across all levels.
            - ``level_S_t`` (list[float]): Per-level Sₜ contributions (L1–L5).
            - ``level_errors`` (list[NDArray]): Per-level prediction errors.
            - ``predictions`` (list[NDArray]): Per-level top-down predictions.
            - ``level_theta`` (list[float]): Per-level threshold θ_ℓ after
              this step's coupled update.
            - ``level_ignition`` (list[bool]): Per-level ignition
              (Sₜ,ℓ > θ_ℓ, evaluated against the pre-update threshold).

        Examples
        --------
        >>> import numpy as np
        >>> hier = APGIHierarchy(n_sensory=32)
        >>> result = hier.forward(np.ones(32) * 0.5, C_metabolic=1.0)
        >>> isinstance(result["S_t_total"], float)
        True
        >>> len(result["level_errors"])
        5
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

        # Ignition is evaluated against each level's pre-update threshold.
        level_ignition = [
            level_S_t[i] > self.levels[i].theta for i in range(n)
        ]

        # Cross-level coupled threshold update (Paper 3 §3.4), Jacobi-style:
        # every level couples to the level above's *pre-update* theta/theta*.
        theta_snapshot = [lv.theta for lv in self.levels]
        theta_star_snapshot = [lv.theta_star for lv in self.levels]
        level_theta: list[float] = []
        for i, level in enumerate(self.levels):
            if i == n - 1:
                theta_next, theta_star_next = 0.0, 0.0
            else:
                theta_next, theta_star_next = (
                    theta_snapshot[i + 1],
                    theta_star_snapshot[i + 1],
                )
            level_theta.append(
                level.step_theta(
                    theta_next, theta_star_next, level_S_t[i], NE_k=NE, dt=dt
                )
            )

        return {
            "S_t_total": float(np.sum(level_S_t)),
            "level_S_t": level_S_t,
            "level_errors": level_errors,
            "predictions": [lv.prediction.copy() for lv in self.levels],
            "level_theta": level_theta,
            "level_ignition": level_ignition,
        }

    def reset(self) -> None:
        """Reset all level states to zero and thresholds to θ_ℓ*."""
        for level in self.levels:
            level.state[:] = 0.0
            level.prediction[:] = 0.0
            level.theta = level.theta_star
