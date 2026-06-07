"""Tiny in-memory sample datasets for quick exploration without Zenodo downloads.

These functions require only numpy and the core apgi package — no external data.

Examples
--------
>>> import apgi
>>> session = apgi.datasets.make_sample_session()
>>> integ = apgi.APGICoreIntegration()
>>> integ.run_sequence(**{k: session[k] for k in
...     ("pi_e", "z_e", "pi_i", "z_i", "C_metabolic", "V_information")})
>>> print(f"ignition rate: {integ.ignition_rate():.3f}")
"""

from __future__ import annotations

import numpy as np


def make_sample_session(n_trials: int = 50, seed: int = 0) -> dict[str, np.ndarray]:
    """Return biologically-plausible APGI trial arrays for quick exploration.

    All parameter ranges are drawn from the same distributions used in the
    canonical seed datasets (MASTER_SEED=2025).  No file download required.

    Args:
        n_trials: Number of trials to generate.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys: pi_e, z_e, pi_i, z_i, C_metabolic, V_information.
        Each value is a 1-D float64 array of length *n_trials*.
    """
    rng = np.random.default_rng(seed)
    return {
        "pi_e": rng.uniform(0.8, 1.5, n_trials),
        "z_e": rng.uniform(0.2, 1.0, n_trials),
        "pi_i": rng.uniform(0.5, 1.5, n_trials),
        "z_i": rng.uniform(0.1, 0.8, n_trials),
        "C_metabolic": rng.uniform(0.5, 2.0, n_trials),
        "V_information": rng.uniform(0.1, 1.0, n_trials),
    }


def make_sample_doc_groups(
    n_per_group: int = 20, n_trials: int = 60, seed: int = 0
) -> dict[str, np.ndarray]:
    """Return a small simulated DoC dataset (VS/UWS, MCS, Controls).

    Precision parameters are calibrated to Protocol 6 group means.

    Args:
        n_per_group: Subjects per diagnostic group.
        n_trials: Trials per subject.
        seed: RNG seed.

    Returns:
        dict with keys: group_labels (str), group_codes (int),
        subject_id (int), S_t (float), ignition (bool).
    """
    from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t

    rng = np.random.default_rng(seed)
    KAPPA, ALPHA, BETA = 100.0, 0.3, 0.7

    group_cfg = {
        "VS_UWS": {"pi_i": 0.30, "pi_e": 0.40, "code": 0},
        "MCS": {"pi_i": 0.80, "pi_e": 0.85, "code": 1},
        "Controls": {"pi_i": 1.20, "pi_e": 1.30, "code": 2},
    }

    groups, codes, subj_ids = [], [], []
    S_t_all, ign_all = [], []
    sid = 0
    for group, cfg in group_cfg.items():
        for _ in range(n_per_group):
            pi_i_s = max(0.05, rng.normal(cfg["pi_i"], cfg["pi_i"] * 0.15))
            pi_e_s = max(0.05, rng.normal(cfg["pi_e"], cfg["pi_e"] * 0.10))
            C = rng.uniform(0.5, 2.0, n_trials)
            V = rng.uniform(0.1, 1.0, n_trials)
            z_e = rng.uniform(0.2, 1.0, n_trials)
            z_i = rng.uniform(0.1, 0.8, n_trials)
            for t in range(n_trials):
                pi_i_eff = compute_pi_i_eff(pi_i_s, C[t], KAPPA)
                s = compute_S_t(pi_e_s, z_e[t], pi_i_eff, z_i[t])
                theta = compute_theta_t(C[t], V[t], ALPHA, BETA)
                groups.append(group)
                codes.append(cfg["code"])
                subj_ids.append(sid)
                S_t_all.append(s)
                ign_all.append(s >= theta)
            sid += 1

    return {
        "group_labels": np.array(groups),
        "group_codes": np.array(codes, dtype=np.int32),
        "subject_id": np.array(subj_ids, dtype=np.int32),
        "S_t": np.array(S_t_all, dtype=np.float64),
        "ignition": np.array(ign_all, dtype=bool),
    }
