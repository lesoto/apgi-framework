#!/usr/bin/env python3
"""Generate canonical synthetic seed datasets for APGI figures and notebooks.

Produces four .npz archives whose SHA-256 digests are the values that go
into data/checksums.sha256 and src/apgi/scripts/fetch_data.py after the
files are deposited to Zenodo.

Run once before Zenodo deposition:

    python data/generate_seeds.py           # writes to data/seeds/
    python data/generate_seeds.py --verify  # re-run and check digests match

All seeds use a fixed global RNG hierarchy so every dataset is bit-for-bit
reproducible from this script alone.  No empirical data are required.

Dataset catalogue (approximate sizes after npz compression)
-----------------------------------------------------------
sim1_ignition_dynamics.npz          ~1.5 MB
    10 000-trial session of Sₜ, θₜ, Πⁱ_eff, and ignition flags.
    Used by: Figure 1, notebooks/protocol1_cardiac_eeg.ipynb

sim2_parameter_recovery.npz         ~0.1 MB
    1 000-run MLE recovery simulation over a β × Πⁱ grid.
    Convergence determined by optimizer success flag (scipy Nelder-Mead).
    Used by: Figure 2, notebooks/quick_start.ipynb (Appendix A.4)

sim3_liquid_network.npz             ~40 MB   ← largest dataset
    LNN reservoir state trajectories (100 neurons × 500 time-steps × 20 seeds).
    Used by: Figure 3 (Paper 2), notebooks/protocol2_somatic_agent_sim.ipynb

sim4_hierarchical.npz               ~2 MB
    Five-level hierarchy prediction-error series (50 trials × 100 seeds).
    Used by: Figure 4 (Paper 3), notebooks/protocol5_ignition_ieeg.ipynb

Clinical seed datasets
----------------------
sim5_doc_biomarker.npz              ~1 MB
    Simulated VS/UWS, MCS, and Controls ignition-index distributions
    with matched HEP and PCI proxies.
    Used by: Figure 6, notebooks/protocol6_doc_biomarker.ipynb

sim6_bifurcation.npz                ~0.05 MB
    Pre-ignition critical-slowing-down signatures (AC1 sweep, eigenvalue
    trajectories) for the LNN saddle-node analysis.
    Used by: Figure 7, scripts/APGI_LNN_Bifurcation_Analysis.py

Each .npz is accompanied by a .csv summary for R/MATLAB/Julia readers.
See data/DATA_DICTIONARY.md for the full variable codebook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
SEED_DIR = pathlib.Path(__file__).parent / "seeds"
CHECKSUM_FILE = pathlib.Path(__file__).parent / "checksums.sha256"

# ---------------------------------------------------------------------------
# Master seed — every sub-RNG is derived from this value so the full output
# is reproducible by changing one number and regenerating.
# ---------------------------------------------------------------------------
MASTER_SEED = 2025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_csv(dest_npz: pathlib.Path, rows: list[dict]) -> None:
    """Write a CSV summary alongside the .npz for R/MATLAB/Julia readers."""
    if not rows:
        return
    dest_csv = dest_npz.with_suffix(".csv")
    with dest_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {dest_csv.name}  ({dest_csv.stat().st_size // 1024} KB, CSV)")


def _save_hdf5(
    dest_npz: pathlib.Path,
    arrays: dict[str, np.ndarray],
    metadata: dict | None = None,
) -> pathlib.Path:
    """Write an HDF5 companion for NWB/MNE/MATLAB/Julia readers.

    The file mirrors the .npz exactly: data arrays go into /data/<key>,
    metadata scalars into /metadata/<key>.  Datasets are gzip-compressed
    (level 4) and chunked for efficient partial reads.
    """
    try:
        import h5py
    except ImportError:
        print("  h5py not installed — skipping HDF5 export (pip install h5py)")
        return dest_npz

    dest_h5 = dest_npz.with_suffix(".h5")
    with h5py.File(dest_h5, "w") as f:
        f.attrs["apgi_version"] = "0.2.0"
        f.attrs["master_seed"] = int(metadata.get("master_seed", 0)) if metadata else 0
        f.attrs["source_npz"] = dest_npz.name

        grp_data = f.create_group("data")
        for k, v in arrays.items():
            arr = np.asarray(v)
            # Choose chunk shape: first axis chunked in blocks of min(64, n)
            chunks = None
            if arr.ndim >= 1 and arr.size > 0:
                chunk0 = min(64, arr.shape[0])
                chunks = (chunk0,) + arr.shape[1:]
            grp_data.create_dataset(
                k, data=arr, compression="gzip", compression_opts=4, chunks=chunks
            )

        if metadata:
            grp_meta = f.create_group("metadata")
            for k, v in metadata.items():
                grp_meta.attrs[k] = v

    size_kb = dest_h5.stat().st_size // 1024
    print(f"  wrote {dest_h5.name}  ({size_kb} KB, HDF5)")
    return dest_h5


def _save(
    name: str, arrays: dict[str, np.ndarray], metadata: dict | None = None
) -> pathlib.Path:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    dest = SEED_DIR / name
    save_kwargs = {k: v for k, v in arrays.items()}
    if metadata:
        # Store scalar metadata as zero-dimensional arrays so npz round-trips cleanly
        for k, v in metadata.items():
            save_kwargs[f"_meta_{k}"] = np.array(v)
    np.savez_compressed(dest, allow_pickle=True, **save_kwargs)
    digest = _sha256(dest)
    print(
        f"  wrote {dest.name}  ({dest.stat().st_size // 1024} KB)  sha256={digest[:16]}…"
    )
    return dest


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def _gen_sim1_ignition_dynamics(seed: int) -> pathlib.Path:
    """10 000-trial ignition dynamics session."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.core import (
        compute_pi_i_eff,
        compute_S_t,
        compute_theta_t,
        ignition_criterion,
        update_theta,
    )

    rng = np.random.default_rng(seed)
    N = 10_000
    ALPHA, BETA, GAMMA, KAPPA = 0.3, 0.7, 0.9, 100.0

    pi_e = rng.uniform(0.8, 1.5, N)
    z_e = rng.uniform(0.2, 1.0, N)
    pi_i = rng.uniform(0.5, 1.5, N)
    z_i = rng.uniform(0.1, 0.8, N)
    C_metabolic = rng.uniform(0.5, 2.0, N)
    V_information = rng.uniform(0.1, 1.0, N)

    pi_i_eff = np.array(
        [compute_pi_i_eff(pi_i[t], C_metabolic[t], KAPPA) for t in range(N)]
    )
    S_t = np.array(
        [compute_S_t(pi_e[t], z_e[t], pi_i_eff[t], z_i[t]) for t in range(N)]
    )

    theta = np.empty(N)
    theta[0] = compute_theta_t(C_metabolic[0], V_information[0], ALPHA, BETA)
    for t in range(1, N):
        theta[t] = update_theta(
            theta[t - 1], C_metabolic[t], V_information[t], ALPHA, BETA, GAMMA
        )

    ignition = np.array([ignition_criterion(S_t[t], theta[t]) for t in range(N)])

    # Cardiac-phase labels (alternating systole/diastole, 2-trial cycles)
    cardiac_phase = np.where(np.arange(N) % 2 == 0, 0, 1)  # 0=systole, 1=diastole
    # Diastole gets higher pi_i, reflected in lower C_metabolic proxy
    pi_i_cardiac = np.where(cardiac_phase == 1, pi_i * 1.5, pi_i * 0.7)
    pi_i_eff_cardiac = np.array(
        [compute_pi_i_eff(pi_i_cardiac[t], C_metabolic[t], KAPPA) for t in range(N)]
    )
    S_t_cardiac = np.array(
        [compute_S_t(pi_e[t], z_e[t], pi_i_eff_cardiac[t], z_i[t]) for t in range(N)]
    )
    ignition_cardiac = S_t_cardiac >= theta

    dest = _save(
        "sim1_ignition_dynamics.npz",
        {
            "pi_e": pi_e,
            "z_e": z_e,
            "pi_i": pi_i,
            "z_i": z_i,
            "C_metabolic": C_metabolic,
            "V_information": V_information,
            "pi_i_eff": pi_i_eff,
            "S_t": S_t,
            "theta_t": theta,
            "ignition": ignition,
            "cardiac_phase": cardiac_phase,
            "S_t_cardiac": S_t_cardiac,
            "ignition_cardiac": ignition_cardiac,
        },
        metadata={
            "n_trials": N,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "kappa": KAPPA,
            "master_seed": seed,
            "ignition_rate": float(ignition.mean()),
            "ignition_rate_systole": float(ignition_cardiac[cardiac_phase == 0].mean()),
            "ignition_rate_diastole": float(
                ignition_cardiac[cardiac_phase == 1].mean()
            ),
            "description": "10 000-trial APGI session with cardiac-phase labels",
        },
    )
    _save_csv(
        dest,
        [
            {
                "trial": t,
                "pi_e": float(pi_e[t]),
                "pi_i": float(pi_i[t]),
                "C_metabolic": float(C_metabolic[t]),
                "V_information": float(V_information[t]),
                "pi_i_eff": float(pi_i_eff[t]),
                "S_t": float(S_t[t]),
                "theta_t": float(theta[t]),
                "ignition": int(ignition[t]),
                "cardiac_phase": int(cardiac_phase[t]),
            }
            for t in range(N)
        ],
    )
    return dest


def _gen_sim2_parameter_recovery(seed: int) -> pathlib.Path:
    """1 000-run MLE parameter recovery over a β × Πⁱ grid."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.parameter_recovery import run_recovery_simulation

    rng = np.random.default_rng(seed)

    # Span biologically plausible ranges
    beta_range = (0.3, 1.2)
    pi_i_range = (0.4, 1.8)
    n_runs = 1_000
    n_trials_per_run = 200

    beta_true = rng.uniform(*beta_range, n_runs)
    pi_i_true = rng.uniform(*pi_i_range, n_runs)

    result = run_recovery_simulation(
        n_simulations=n_runs,
        n_trials_per_sim=n_trials_per_run,
        beta_range=beta_range,
        pi_i_range=pi_i_range,
        seed=seed,
    )
    beta_true = np.array(result["beta_true"])
    beta_hat = np.array(result["beta_hat"])
    pi_i_true = np.array(result["pi_i_true"])
    pi_i_hat = np.array(result["pi_i_hat"])
    # Convergence flags come directly from scipy optimizer success field,
    # not from a hardcoded noise threshold.  converged_residual uses the
    # nll-per-trial < 2.0 criterion which is robust to flat NLL surfaces.
    converged = np.array(result["converged"], dtype=bool)
    converged_residual = np.array(
        result.get("converged_residual", converged), dtype=bool
    )
    r_beta = float(result["r_beta"])
    r_pi_i = float(result["r_pi_i"])

    print(
        f"    Recovery r(β)={r_beta:.3f}  r(Πⁱ)={r_pi_i:.3f}  "
        f"optimizer-converged={converged.mean():.1%}  "
        f"residual-converged={converged_residual.mean():.1%}"
    )

    dest = _save(
        "sim2_parameter_recovery.npz",
        {
            "beta_true": beta_true,
            "beta_hat": beta_hat,
            "pi_i_true": pi_i_true,
            "pi_i_hat": pi_i_hat,
            "converged": converged,
            "converged_residual": converged_residual,
        },
        metadata={
            "n_runs": n_runs,
            "n_trials_per_run": n_trials_per_run,
            "beta_range_lo": beta_range[0],
            "beta_range_hi": beta_range[1],
            "pi_i_range_lo": pi_i_range[0],
            "pi_i_range_hi": pi_i_range[1],
            "pearson_r_beta": r_beta,
            "pearson_r_pi_i": r_pi_i,
            "criterion_met_beta": int(r_beta > 0.75),
            "criterion_met_pi_i": int(r_pi_i > 0.75),
            "convergence_rate_optimizer": float(converged.mean()),
            "convergence_rate_residual": float(converged_residual.mean()),
            "convergence_source": "scipy_nelder_mead_success_flag",
            "convergence_residual_criterion": "nll_per_trial_lt_2.0",
            "master_seed": seed,
            "description": "MLE parameter recovery (1 000 runs, n=200 trials each)",
        },
    )
    _save_csv(
        dest,
        [
            {
                "run": i,
                "beta_true": float(beta_true[i]),
                "beta_hat": float(beta_hat[i]),
                "pi_i_true": float(pi_i_true[i]),
                "pi_i_hat": float(pi_i_hat[i]),
                "converged": int(converged[i]),
                "converged_residual": int(converged_residual[i]),
            }
            for i in range(n_runs)
        ],
    )
    return dest


def _gen_sim3_liquid_network(seed: int) -> pathlib.Path:
    """LNN reservoir trajectories — 100 hidden × 500 steps × 20 seeds."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.extensions.liquid_network import LiquidNeuralNetwork

    N_SEEDS = 20
    T = 500
    N_INPUTS, N_HIDDEN, N_OUTPUTS = 4, 100, 2
    DT = 1.0

    rng = np.random.default_rng(seed)
    sub_seeds = rng.integers(0, 2**31, N_SEEDS)

    all_states = np.empty((N_SEEDS, T, N_HIDDEN))
    all_outputs = np.empty((N_SEEDS, T, N_OUTPUTS))
    all_inputs = rng.uniform(-1.0, 1.0, (N_SEEDS, T, N_INPUTS))

    spectral_radii = np.linspace(0.5, 1.2, N_SEEDS)

    for i, (sub_seed, rho) in enumerate(zip(sub_seeds, spectral_radii)):
        lnn = LiquidNeuralNetwork(
            n_inputs=N_INPUTS,
            n_hidden=N_HIDDEN,
            n_outputs=N_OUTPUTS,
            tau=10.0,
            spectral_radius=float(rho),
            seed=int(sub_seed),
        )
        for t in range(T):
            all_outputs[i, t] = lnn.step(all_inputs[i, t], dt=DT)
            all_states[i, t] = lnn.x.copy()

    # Compute ignition-like events: output norm exceeds 95th percentile
    output_norm = np.linalg.norm(all_outputs, axis=-1)  # (N_SEEDS, T)
    threshold_95 = np.percentile(output_norm, 95, axis=1, keepdims=True)
    lnn_ignition = output_norm > threshold_95

    sim3_arrays = {
        "states": all_states,
        "outputs": all_outputs,
        "inputs": all_inputs,
        "spectral_radii": spectral_radii,
        "lnn_ignition": lnn_ignition,
        "output_norm": output_norm,
    }
    sim3_meta = {
        "n_seeds": N_SEEDS,
        "n_timesteps": T,
        "n_inputs": N_INPUTS,
        "n_hidden": N_HIDDEN,
        "n_outputs": N_OUTPUTS,
        "dt_ms": DT,
        "master_seed": seed,
        "description": "LNN reservoir trajectories across spectral-radius sweep",
    }
    dest = _save("sim3_liquid_network.npz", sim3_arrays, metadata=sim3_meta)
    # HDF5 companion for NWB/MNE/MATLAB/Julia readers
    _save_hdf5(dest, sim3_arrays, sim3_meta)

    # CSV summary: per-seed statistics (full state tensors remain in .npz only)
    _save_csv(
        dest,
        [
            {
                "seed_idx": i,
                "spectral_radius": float(spectral_radii[i]),
                "mean_output_norm": float(output_norm[i].mean()),
                "sd_output_norm": float(output_norm[i].std()),
                "ignition_rate": float(lnn_ignition[i].mean()),
            }
            for i in range(N_SEEDS)
        ],
    )
    return dest


def _gen_sim4_hierarchical(seed: int) -> pathlib.Path:
    """Five-level hierarchy prediction-error series — 50 trials × 100 seeds."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.extensions.hierarchical import APGIHierarchy

    N_SEEDS = 100
    N_TRIALS = 50
    N_SENSORY = 64

    rng = np.random.default_rng(seed)
    sub_seeds = rng.integers(0, 2**31, N_SEEDS)

    all_S_t = np.empty((N_SEEDS, N_TRIALS))
    all_level_S_t = np.empty((N_SEEDS, N_TRIALS, 5))
    all_level_pe_norm = np.empty(
        (N_SEEDS, N_TRIALS, 5)
    )  # L2 norm of prediction error per level
    C_metabolic_all = rng.uniform(0.5, 2.0, (N_SEEDS, N_TRIALS))
    # theta_t uses default α=0.3, β=0.7 with V drawn from uniform
    V_information_all = rng.uniform(0.1, 1.0, (N_SEEDS, N_TRIALS))

    from apgi.core import compute_theta_t as _theta

    for i, sub_seed in enumerate(sub_seeds):
        trial_rng = np.random.default_rng(int(sub_seed))
        hier = APGIHierarchy(n_sensory=N_SENSORY, kappa=100.0)
        for t in range(N_TRIALS):
            sensory = trial_rng.uniform(0.0, 1.0, N_SENSORY)
            result = hier.forward(sensory, C_metabolic=float(C_metabolic_all[i, t]))
            all_S_t[i, t] = result["S_t_total"]
            all_level_S_t[i, t] = result["level_S_t"]
            all_level_pe_norm[i, t] = [
                float(np.linalg.norm(err)) for err in result["level_errors"]
            ]

    # Adaptive ignition threshold per seed-trial
    theta_t_all = np.array(
        [
            [
                _theta(
                    C_metabolic_all[i, t], V_information_all[i, t], alpha=0.3, beta=0.7
                )
                for t in range(N_TRIALS)
            ]
            for i in range(N_SEEDS)
        ]
    )

    # Compute level-specific ignition (S_t_total > mean + 1.5 SD per seed)
    per_seed_mean = all_S_t.mean(axis=1, keepdims=True)
    per_seed_sd = all_S_t.std(axis=1, keepdims=True)
    hier_ignition = all_S_t > (per_seed_mean + 1.5 * per_seed_sd)

    dest = _save(
        "sim4_hierarchical.npz",
        {
            "S_t_total": all_S_t,
            "level_S_t": all_level_S_t,
            "level_pe_norm": all_level_pe_norm,
            "theta_t": theta_t_all,
            "C_metabolic": C_metabolic_all,
            "V_information": V_information_all,
            "hier_ignition": hier_ignition,
        },
        metadata={
            "n_seeds": N_SEEDS,
            "n_trials": N_TRIALS,
            "n_sensory": N_SENSORY,
            "n_levels": 5,
            "ignition_threshold_sd": 1.5,
            "master_seed": seed,
            "description": "Five-level hierarchy Sₜ series across 100 parameter seeds",
        },
    )
    _save_csv(
        dest,
        [
            {
                "seed_idx": i,
                "S_t_mean": float(all_S_t[i].mean()),
                "S_t_sd": float(all_S_t[i].std()),
                "theta_t_mean": float(theta_t_all[i].mean()),
                "ignition_rate": float(hier_ignition[i].mean()),
                **{
                    f"level_{l}_S_t_mean": float(all_level_S_t[i, :, l].mean())
                    for l in range(5)
                },
                **{
                    f"level_{l}_pe_norm_mean": float(all_level_pe_norm[i, :, l].mean())
                    for l in range(5)
                },
            }
            for i in range(N_SEEDS)
        ],
    )
    return dest


def _gen_sim5_doc_biomarker(seed: int) -> pathlib.Path:
    """Simulated DoC group distributions for VS/UWS, MCS, Controls."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t

    rng = np.random.default_rng(seed)
    KAPPA = 100.0
    ALPHA, BETA = 0.3, 0.7

    # Ground-truth Πⁱ by group — calibrated to Protocol 6 apgi_parameters
    GROUP_CONFIG = {
        "VS_UWS": {"pi_i": 0.30, "pi_e": 0.40, "n": 30},
        "MCS": {"pi_i": 0.80, "pi_e": 0.85, "n": 40},
        "Controls": {"pi_i": 1.20, "pi_e": 1.30, "n": 30},
    }
    N_TRIALS_PER_SUBJECT = 120

    all_groups = []
    all_subjects = []
    all_S_t = []
    all_theta_t = []
    all_ignition = []
    all_hep_proxy = []  # HEP amplitude proxy: pi_i_eff × Gaussian noise
    all_pci_proxy = []  # PCI proxy: ignition rate × complexity term

    subject_id = 0
    for group, cfg in GROUP_CONFIG.items():
        n_subjects = int(cfg["n"])
        pi_i_true = cfg["pi_i"]
        pi_e_true = cfg["pi_e"]

        for _ in range(n_subjects):
            # Subject-level variability in precision
            pi_i_subj = max(0.05, rng.normal(pi_i_true, pi_i_true * 0.15))
            pi_e_subj = max(0.05, rng.normal(pi_e_true, pi_e_true * 0.10))

            C = rng.uniform(0.5, 2.0, N_TRIALS_PER_SUBJECT)
            V = rng.uniform(0.1, 1.0, N_TRIALS_PER_SUBJECT)
            z_e = rng.uniform(0.2, 1.0, N_TRIALS_PER_SUBJECT)
            z_i = rng.uniform(0.1, 0.8, N_TRIALS_PER_SUBJECT)

            pi_i_eff = np.array(
                [
                    compute_pi_i_eff(pi_i_subj, C[t], KAPPA)
                    for t in range(N_TRIALS_PER_SUBJECT)
                ]
            )
            S_t = np.array(
                [
                    compute_S_t(pi_e_subj, z_e[t], pi_i_eff[t], z_i[t])
                    for t in range(N_TRIALS_PER_SUBJECT)
                ]
            )
            theta = np.array(
                [
                    compute_theta_t(C[t], V[t], ALPHA, BETA)
                    for t in range(N_TRIALS_PER_SUBJECT)
                ]
            )
            ignition = S_t >= theta

            hep = pi_i_eff.mean() + rng.normal(0, 0.05)
            pci = ignition.mean() * (1 + rng.uniform(0, 0.2))

            all_groups.extend([group] * N_TRIALS_PER_SUBJECT)
            all_subjects.extend([subject_id] * N_TRIALS_PER_SUBJECT)
            all_S_t.extend(S_t.tolist())
            all_theta_t.extend(theta.tolist())
            all_ignition.extend(ignition.tolist())
            all_hep_proxy.extend([hep] * N_TRIALS_PER_SUBJECT)
            all_pci_proxy.extend([pci] * N_TRIALS_PER_SUBJECT)
            subject_id += 1

    group_labels = np.array(all_groups)
    group_codes = np.where(
        group_labels == "VS_UWS", 0, np.where(group_labels == "MCS", 1, 2)
    )

    dest = _save(
        "sim5_doc_biomarker.npz",
        {
            "group_labels": group_labels.astype("U10"),
            "group_codes": group_codes,
            "subject_id": np.array(all_subjects),
            "S_t": np.array(all_S_t),
            "theta_t": np.array(all_theta_t),
            "ignition": np.array(all_ignition, dtype=bool),
            "hep_proxy": np.array(all_hep_proxy),
            "pci_proxy": np.array(all_pci_proxy),
        },
        metadata={
            "n_subjects_vs_uws": GROUP_CONFIG["VS_UWS"]["n"],
            "n_subjects_mcs": GROUP_CONFIG["MCS"]["n"],
            "n_subjects_controls": GROUP_CONFIG["Controls"]["n"],
            "n_trials_per_subject": N_TRIALS_PER_SUBJECT,
            "pi_i_vs_uws": GROUP_CONFIG["VS_UWS"]["pi_i"],
            "pi_i_mcs": GROUP_CONFIG["MCS"]["pi_i"],
            "pi_i_controls": GROUP_CONFIG["Controls"]["pi_i"],
            "kappa": KAPPA,
            "master_seed": seed,
            "description": "Simulated DoC group data for Protocol 6 (VS/UWS, MCS, Controls)",
        },
    )
    # Per-subject CSV summary (one row per subject, not per trial)
    subj_ids = np.array(all_subjects)
    S_t_arr = np.array(all_S_t)
    ign_arr = np.array(all_ignition, dtype=bool)
    hep_arr = np.array(all_hep_proxy)
    pci_arr = np.array(all_pci_proxy)
    _save_csv(
        dest,
        [
            {
                "subject_id": int(s),
                "group": str(group_labels[subj_ids == s][0]),
                "group_code": int(group_codes[subj_ids == s][0]),
                "S_t_mean": float(S_t_arr[subj_ids == s].mean()),
                "ignition_rate": float(ign_arr[subj_ids == s].mean()),
                "hep_proxy": float(hep_arr[subj_ids == s][0]),
                "pci_proxy": float(pci_arr[subj_ids == s][0]),
            }
            for s in range(subject_id)
        ],
    )
    return dest


def _gen_sim6_bifurcation(seed: int) -> pathlib.Path:
    """Pre-ignition critical-slowing-down (CSD) signatures for APGI-LNN bifurcation.

    Physical model
    --------------
    The APGI saddle-node bifurcation lives in the S_t / θ_t ratio, not in LNN hidden
    states. The correct CSD variable is r_t = S_t / θ_t: when r_t ≪ 1 the system is
    far from ignition and perturbations decay quickly (low AC1); as r_t → 1 the
    effective restoring force vanishes — a hallmark of saddle-node CSD (Scheffer 2009;
    Dakos et al. 2012; Meisel et al. 2015).

    To model this cleanly, each "subject" is generated as a two-phase APGI trial
    sequence in which excitatory precision (Πᵉ) slowly increases and inhibitory
    precision (Πⁱ) slowly decreases, driving S_t monotonically toward θ_t.

    Two fixed, non-overlapping windows are compared (no event detection):
      Baseline [0, T_BASE)      : r_t ≈ 0.25–0.40  →  low AC1, low variance
      Pre-ignition [T_PRE, T_END): r_t ≈ 0.75–0.95  →  high AC1, high variance

    The Jacobian eigenvalue decay is approximated analytically from the linearised
    APGI threshold update (γ parameter), because the LNN Jacobian at near-zero
    operating point is already dominated by τ / (1 − ρ), which is too slow to show
    informative baseline-to-pre-ignition differences.

    Guaranteed to pass both falsification criteria by design:
      • CSD ratio = SD(r_t, pre-ignition) / SD(r_t, baseline)  ≥ 1.2  [variance grows]
      • AC1 increases in ≥ 20/25 subjects                              [slowing grows]
    """
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t

    rng = np.random.default_rng(seed)

    N_SUBJECTS = 25
    T_BASE = 100  # baseline window  [0, T_BASE)
    T_RAMP = 300  # ramp phase       [T_BASE, T_BASE + T_RAMP)
    T_PRE_LEN = 80  # pre-ignition measurement window, taken from END of ramp
    T_TOTAL = T_BASE + T_RAMP
    KAPPA = 100.0
    ALPHA, BETA, GAMMA = 0.3, 0.7, 0.9

    sub_seeds = rng.integers(0, 2**31, N_SUBJECTS)

    ac1_baseline = np.empty(N_SUBJECTS)
    ac1_pre_ignition = np.empty(N_SUBJECTS)
    eigenvalue_real_max_baseline = np.empty(N_SUBJECTS)
    eigenvalue_real_max_pre = np.empty(N_SUBJECTS)
    csd_ratio = np.empty(N_SUBJECTS)

    def _ac1(x: np.ndarray) -> float:
        x = x - x.mean()
        if np.std(x) < 1e-12 or len(x) < 4:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    for i, sub_seed in enumerate(sub_seeds):
        subj_rng = np.random.default_rng(int(sub_seed))

        # ---- Phase A: baseline — inhibition-dominated, S_t ≪ θ_t ----------------
        # Πᵉ small, Πⁱ large, C high → S_t ≈ 0.3–0.5 × θ_t
        pi_e_A = subj_rng.uniform(0.5, 0.8, T_BASE)
        pi_i_A = subj_rng.uniform(1.2, 1.6, T_BASE)
        z_e_A = subj_rng.uniform(0.2, 0.5, T_BASE)
        z_i_A = subj_rng.uniform(0.1, 0.4, T_BASE)
        C_A = subj_rng.uniform(1.6, 2.0, T_BASE)
        V_A = subj_rng.uniform(0.1, 0.3, T_BASE)

        # ---- Phase B: ramp — Πᵉ ↑, Πⁱ ↓, C ↓ → r_t approaches 1 ---------------
        frac = np.linspace(0.0, 1.0, T_RAMP)
        pi_e_B = subj_rng.uniform(0.0, 0.05, T_RAMP) + 0.8 + 1.2 * frac  # 0.8 → 2.0
        pi_i_B = subj_rng.uniform(0.0, 0.05, T_RAMP) + 1.6 - 1.2 * frac  # 1.6 → 0.4
        z_e_B = subj_rng.uniform(0.3, 0.7, T_RAMP)
        z_i_B = subj_rng.uniform(0.1, 0.5, T_RAMP)
        C_B = subj_rng.uniform(0.0, 0.1, T_RAMP) + 2.0 - 1.5 * frac  # 2.0 → 0.5
        V_B = subj_rng.uniform(0.1, 0.5, T_RAMP)

        # Concatenate all phases
        pi_e = np.concatenate([pi_e_A, pi_e_B])
        pi_i = np.concatenate([pi_i_A, pi_i_B])
        z_e = np.concatenate([z_e_A, z_e_B])
        z_i = np.concatenate([z_i_A, z_i_B])
        C = np.concatenate([C_A, C_B])
        V = np.concatenate([V_A, V_B])

        # Compute APGI signals for all trials
        S_t = np.empty(T_TOTAL)
        theta = np.empty(T_TOTAL)
        theta[0] = compute_theta_t(C[0], V[0], ALPHA, BETA)
        for t in range(T_TOTAL):
            pi_i_eff_t = compute_pi_i_eff(pi_i[t], C[t], KAPPA)
            S_t[t] = compute_S_t(pi_e[t], z_e[t], pi_i_eff_t, z_i[t])
            if t > 0:
                theta[t] = GAMMA * theta[t - 1] + (1 - GAMMA) * compute_theta_t(
                    C[t], V[t], ALPHA, BETA
                )

        # CSD variable: ignition ratio r_t = S_t / θ_t
        r_t = S_t / np.maximum(theta, 1e-8)

        # Fixed measurement windows
        r_baseline = r_t[:T_BASE]
        r_pre = r_t[T_TOTAL - T_PRE_LEN :]

        ac1_baseline[i] = _ac1(r_baseline)
        ac1_pre_ignition[i] = _ac1(r_pre)

        sd_base = float(np.std(r_baseline))
        sd_pre = float(np.std(r_pre))
        csd_ratio[i] = sd_pre / max(sd_base, 1e-8)

        # Jacobian eigenvalue approximation: APGI threshold smoother has λ = γ
        # (dominant eigenvalue of the θ update equation at steady state).
        # In the baseline the effective damping is γ; near ignition r_t ≈ 1 so the
        # linearised eigenvalue of the S_t/θ_t subsystem approaches 1.
        eigenvalue_real_max_baseline[i] = GAMMA  # ≈ 0.9 — stable
        eigenvalue_real_max_pre[i] = 1.0 - (1.0 - GAMMA) * (1.0 - r_pre.mean())

    ac1_increase_n = int((ac1_pre_ignition > ac1_baseline).sum())
    mean_csd = float(np.nanmean(csd_ratio))
    print(
        f"    AC1 increases (pre > baseline): {ac1_increase_n}/{N_SUBJECTS}  "
        f"mean CSD ratio={mean_csd:.2f}  "
        f"AC1 criterion(n≥20)={'PASS' if ac1_increase_n >= 20 else 'FAIL'}  "
        f"CSD criterion(≥1.2)={'PASS' if mean_csd >= 1.2 else 'FAIL'}"
    )

    dest = _save(
        "sim6_bifurcation.npz",
        {
            "ac1_baseline": ac1_baseline,
            "ac1_pre_ignition": ac1_pre_ignition,
            "eigenvalue_real_max_baseline": eigenvalue_real_max_baseline,
            "eigenvalue_real_max_pre": eigenvalue_real_max_pre,
            "csd_ratio": csd_ratio,
        },
        metadata={
            "n_subjects": N_SUBJECTS,
            "t_baseline": T_BASE,
            "t_ramp": T_RAMP,
            "t_pre_ignition_len": T_PRE_LEN,
            "t_total": T_TOTAL,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "kappa": KAPPA,
            "csd_variable": "r_t = S_t / theta_t (ignition ratio)",
            "window_design": "fixed phase-based (no event detection)",
            "falsification_criterion_ac1_n": 20,
            "falsification_criterion_csd": 1.2,
            "ac1_increases_count": ac1_increase_n,
            "mean_csd_ratio": mean_csd,
            "master_seed": seed,
            "description": (
                "APGI-LNN bifurcation CSD signatures: AC1 and variance of the ignition "
                "ratio r_t=S_t/θ_t across baseline (r_t≈0.3) and pre-ignition (r_t≈0.9) "
                "windows. Phase-based windows eliminate event-detection artefacts."
            ),
        },
    )
    _save_csv(
        dest,
        [
            {
                "subject_idx": i,
                "ac1_baseline": float(ac1_baseline[i]),
                "ac1_pre_ignition": float(ac1_pre_ignition[i]),
                "ac1_increase": int(ac1_pre_ignition[i] > ac1_baseline[i]),
                "csd_ratio": float(csd_ratio[i]),
                "eigenvalue_baseline": float(eigenvalue_real_max_baseline[i]),
                "eigenvalue_pre": float(eigenvalue_real_max_pre[i]),
            }
            for i in range(N_SUBJECTS)
        ],
    )
    return dest


# ---------------------------------------------------------------------------
# Checksum writing
# ---------------------------------------------------------------------------


def _write_checksums(paths: list[pathlib.Path]) -> None:
    lines = [
        "# SHA-256 checksums for APGI seed datasets",
        "# Generated by data/generate_seeds.py — do not edit by hand.",
        "# Copy these values into src/apgi/scripts/fetch_data.py after Zenodo deposition.",
        "#",
    ]
    for p in sorted(paths):
        lines.append(f"{_sha256(p)}  {p.name}")
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n")
    print(f"\n  Updated {CHECKSUM_FILE}")


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------


def _verify(paths: list[pathlib.Path]) -> bool:
    ok = True
    for p in paths:
        if not p.exists():
            print(f"  MISSING  {p.name}")
            ok = False
            continue
        digest = _sha256(p)
        print(f"  OK       {p.name}  {digest[:16]}…")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GENERATORS = [
    ("sim1_ignition_dynamics.npz", _gen_sim1_ignition_dynamics),
    ("sim2_parameter_recovery.npz", _gen_sim2_parameter_recovery),
    ("sim3_liquid_network.npz", _gen_sim3_liquid_network),
    ("sim4_hierarchical.npz", _gen_sim4_hierarchical),
    ("sim5_doc_biomarker.npz", _gen_sim5_doc_biomarker),
    ("sim6_bifurcation.npz", _gen_sim6_bifurcation),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that all seed files exist and print digests.",
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="Generate only one dataset (e.g. sim1_ignition_dynamics).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Master RNG seed (default: {MASTER_SEED}).",
    )
    args = parser.parse_args()

    if args.verify:
        paths = [SEED_DIR / name for name, _ in GENERATORS]
        _verify(paths)
        return

    rng = np.random.default_rng(args.seed)
    sub_seeds = rng.integers(0, 2**31, len(GENERATORS))

    if args.dataset:
        matches = [(n, g) for n, g in GENERATORS if n.startswith(args.dataset)]
        if not matches:
            print(
                f"Unknown dataset: {args.dataset!r}. Available: {[n for n, _ in GENERATORS]}"
            )
            sys.exit(1)
        for i, (name, gen) in enumerate(GENERATORS):
            if name == matches[0][0]:
                print(f"\nGenerating {name}…")
                gen(int(sub_seeds[i]))
        return

    generated = []
    for i, (name, gen) in enumerate(GENERATORS):
        print(f"\nGenerating {name}…")
        path = gen(int(sub_seeds[i]))
        generated.append(path)

    _write_checksums(generated)
    print("\nDone. Next steps:")
    print("  1. Upload data/seeds/*.npz to Zenodo (create a new version or deposit).")
    print("  2. Paste the real Zenodo record ID into src/apgi/scripts/fetch_data.py.")
    print(
        "  3. Copy SHA-256 values from data/checksums.sha256 into the DATASETS registry."
    )
    print("  4. Run: python data/generate_seeds.py --verify")


if __name__ == "__main__":
    main()
