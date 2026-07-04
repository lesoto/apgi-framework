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

    Parameter ranges match the distributions used in the canonical archived
    seed datasets (generated separately by ``data/generate_seeds.py`` under
    ``MASTER_SEED=2025``), but this function draws its own independent
    samples from *seed* — it does not reproduce the archived datasets
    bit-for-bit. No file download required.

    Args:
        n_trials: Number of trials to generate.
        seed: RNG seed for reproducibility (independent of MASTER_SEED).

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
    from apgi.core import (
        BETA_SM_DEFAULT,
        GAMMA_SIG_DEFAULT,
        TAU_S_DEFAULT,
        accumulate_S_t,
        compute_pi_i_eff,
        compute_S_t,
        ignition_criterion,
        step_theta,
        theta_equilibrium,
    )

    rng = np.random.default_rng(seed)

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
            I_t = rng.uniform(0.1, 1.0, n_trials)
            z_e = rng.uniform(0.2, 1.0, n_trials)
            z_i = rng.uniform(0.1, 0.8, n_trials)
            M_hat = rng.uniform(0.0, 0.5, n_trials)
            theta_t = theta_equilibrium(float(C.mean()), float(I_t.mean()))
            S_acc = 0.0
            for t in range(n_trials):
                pi_i_eff = compute_pi_i_eff(
                    pi_i_s, beta_sm=BETA_SM_DEFAULT, M_hat=float(M_hat[t])
                )
                s_inp = compute_S_t(pi_e_s, float(z_e[t]), pi_i_eff, float(z_i[t]))
                S_acc = accumulate_S_t(S_acc, s_inp, tau_S=TAU_S_DEFAULT)
                ign = ignition_criterion(S_acc, theta_t, GAMMA_SIG_DEFAULT)
                theta_t = step_theta(theta_t, float(C[t]), float(I_t[t]))
                groups.append(group)
                codes.append(cfg["code"])
                subj_ids.append(sid)
                S_t_all.append(S_acc)
                ign_all.append(ign)
            sid += 1

    return {
        "group_labels": np.array(groups),
        "group_codes": np.array(codes, dtype=np.int32),
        "subject_id": np.array(subj_ids, dtype=np.int32),
        "S_t": np.array(S_t_all, dtype=np.float64),
        "ignition": np.array(ign_all, dtype=bool),
    }
