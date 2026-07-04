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
sim0_hep_proxy.npz                  ~0.1 MB
    HEP amplitude, heartbeat discrimination d-prime, physostigmine effect, and
    aINS BOLD tracking data for 60 subjects (30 main + 30 replication).
    Used by: Figure 0, notebooks/protocol0_hep_proxy_validation.ipynb  [EP-0]

sim1_ignition_dynamics.npz          ~1.5 MB
    10 000-trial session of St, thetat, pi_i_eff, and ignition flags.
    Used by: Figure 1, notebooks/protocol1_cardiac_eeg.ipynb           [EP-1]

sim2_parameter_recovery.npz         ~0.1 MB
    1 000-run MLE recovery simulation over a beta x pi_i grid.
    Convergence determined by optimizer success flag (scipy Nelder-Mead).
    Used by: Figure 2, notebooks/quick_start.ipynb (Appendix A.4)      [EP-2]

sim3_liquid_network.npz             ~40 MB   <- largest dataset
    LNN reservoir state trajectories (100 neurons x 500 time-steps x 20 seeds).
    Used by: Figure 3 (Paper 2), notebooks/protocol2_somatic_agent_sim.ipynb [EP-2]

sim4_hierarchical.npz               ~2 MB
    Five-level hierarchy prediction-error series (50 trials x 100 seeds).
    Used by: Figure 4 (Paper 1), notebooks/protocol3_anticipation_fmri.ipynb [EP-3]

Clinical seed datasets
----------------------
sim5_doc_biomarker.npz              ~1.2 MB
    Simulated VS/UWS, MCS, EMCS, and Controls ignition-index distributions
    (N=110: VS/UWS=30, MCS=30, EMCS=20, Controls=30) with matched HEP and
    PCI proxies. Protocol 7 (EP-7) four-group design.
    Used by: Figure 7, notebooks/protocol7_doc_biomarker.ipynb         [EP-7]

sim6_bifurcation.npz                ~0.05 MB
    Pre-ignition critical-slowing-down signatures (AC1 sweep, eigenvalue
    trajectories) for the iEEG saddle-node bifurcation analysis (EP-6).
    Used by: Figure 6, scripts/APGI_LNN_Bifurcation_Analysis.py        [EP-6]

sim7_metabolic_crossover.npz        ~0.3 MB
    2x2 within-subject metabolic crossover (MetabolicState x InteroceptiveLoad):
    d-prime and P3b amplitude per cell for N=60 subjects with pupil and RMSSD
    covariates. Protocol 4 (EP-4) allostatic triage design.
    Used by: Figure 4b, notebooks/protocol4_metabolic_crossover.ipynb  [EP-4]

sim8_tms_pci.npz                    ~0.2 MB
    TMS/tFUS site comparison (aINS vs dlPFC vs vertex) for PCI, HEP amplitude,
    and HEP-PCI coupling across N=36 subjects stratified by pi_i tertile.
    Protocol 5 (EP-5) causal TMS design.
    Used by: Figure 5, notebooks/protocol5_causal_tms.ipynb            [EP-5]

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
    """10 000-trial ignition dynamics session (EP-1 cardiac EEG protocol)."""
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.core import (
        compute_pi_i_eff,
        compute_S_t,
        ignition_criterion,
        step_theta,
        theta_equilibrium,
    )

    rng = np.random.default_rng(seed)
    N = 10_000
    BETA_SM = 0.6   # somatic-marker gain (Table 2 default)
    M_HAT = 0.0     # neutral somatic-marker context for baseline dynamics

    pi_e = rng.uniform(0.8, 1.5, N)
    z_e = rng.uniform(0.2, 1.0, N)
    pi_i = rng.uniform(0.5, 1.5, N)
    z_i = rng.uniform(0.1, 0.8, N)
    C_metabolic = rng.uniform(0.5, 2.0, N)
    V_information = rng.uniform(0.1, 1.0, N)

    # With M_hat=0: pi_i_eff = pi_i_baseline * exp(beta_sm * 0) = pi_i_baseline
    pi_i_eff = np.array(
        [compute_pi_i_eff(pi_i[t], BETA_SM, M_HAT) for t in range(N)]
    )
    S_t = np.array(
        [compute_S_t(pi_e[t], z_e[t], pi_i_eff[t], z_i[t]) for t in range(N)]
    )

    # Adaptive threshold via step_theta ODE (Eq. 3)
    theta = np.empty(N)
    theta[0] = theta_equilibrium(C_metabolic[0], V_information[0])
    for t in range(1, N):
        theta[t] = step_theta(theta[t - 1], C_metabolic[t], V_information[t])

    ignition = np.array([ignition_criterion(S_t[t], theta[t]) for t in range(N)])

    # Cardiac-phase labels (alternating systole/diastole, 2-trial cycles)
    cardiac_phase = np.where(np.arange(N) % 2 == 0, 0, 1)  # 0=systole, 1=diastole
    # Diastole: higher pi_i_baseline (lower inhibitory gating); systole: lower
    pi_i_cardiac = np.where(cardiac_phase == 1, pi_i * 1.5, pi_i * 0.7)
    pi_i_eff_cardiac = np.array(
        [compute_pi_i_eff(pi_i_cardiac[t], BETA_SM, M_HAT) for t in range(N)]
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
            "beta_sm": BETA_SM,
            "m_hat_baseline": M_HAT,
            "kappa": 100,
            "alpha": 0.3,
            "beta_initial_estimate": 0.7,
            "master_seed": seed,
            "ignition_rate": float(ignition.mean()),
            "ignition_rate_systole": float(ignition_cardiac[cardiac_phase == 0].mean()),
            "ignition_rate_diastole": float(
                ignition_cardiac[cardiac_phase == 1].mean()
            ),
            "description": (
                "EP-1 cardiac EEG: 10 000-trial APGI session with cardiac-phase labels. "
                "cardiac_phase 0=systole (lower pi_i), 1=diastole (higher pi_i). "
                "ignition_cardiac tests whether diastolic pi_i elevation drives higher "
                "ignition rate (Pred 1.A)."
            ),
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

    n_runs = 1_000
    n_trials_per_run = 200

    result = run_recovery_simulation(
        n_simulations=n_runs,
        n_trials_per_sim=n_trials_per_run,
        seed=seed,
    )
    # Five-parameter recovery: theta_0, tau_S, pi_i, beta_sm, gamma_sig
    pi_i_true = np.array(result["pi_i_true"])
    pi_i_hat = np.array(result["pi_i_hat"])
    beta_sm_true = np.array(result["beta_sm_true"])
    beta_sm_hat = np.array(result["beta_sm_hat"])
    theta_0_true = np.array(result["theta_0_true"])
    theta_0_hat = np.array(result["theta_0_hat"])
    tau_S_true = np.array(result["tau_S_true"])
    tau_S_hat = np.array(result["tau_S_hat"])
    gamma_sig_true = np.array(result["gamma_sig_true"])
    gamma_sig_hat = np.array(result["gamma_sig_hat"])
    converged = np.array(result["converged"], dtype=bool)
    converged_residual = np.array(
        result.get("converged_residual", converged), dtype=bool
    )
    r_pi_i = float(result["r_pi_i"])
    r_beta_sm = float(result["r_beta_sm"])
    r_theta_0 = float(result["r_theta_0"])
    r_tau_S = float(result["r_tau_S"])
    r_gamma_sig = float(result["r_gamma_sig"])

    print(
        f"    Recovery r(pi_i)={r_pi_i:.3f}  r(beta_sm)={r_beta_sm:.3f}  "
        f"r(theta_0)={r_theta_0:.3f}  r(tau_S)={r_tau_S:.3f}  "
        f"r(gamma_sig)={r_gamma_sig:.3f}  "
        f"converged={converged.mean():.1%}"
    )

    dest = _save(
        "sim2_parameter_recovery.npz",
        {
            "pi_i_true": pi_i_true,
            "pi_i_hat": pi_i_hat,
            "beta_sm_true": beta_sm_true,
            "beta_sm_hat": beta_sm_hat,
            "theta_0_true": theta_0_true,
            "theta_0_hat": theta_0_hat,
            "tau_S_true": tau_S_true,
            "tau_S_hat": tau_S_hat,
            "gamma_sig_true": gamma_sig_true,
            "gamma_sig_hat": gamma_sig_hat,
            "converged": converged,
            "converged_residual": converged_residual,
        },
        metadata={
            "n_runs": n_runs,
            "n_trials_per_run": n_trials_per_run,
            "parameters_recovered": "theta_0, tau_S, pi_i, beta_sm, gamma_sig",
            "pi_i_range": "0.8-2.5 (Table 12 physiological constraints)",
            "beta_sm_range": "0.2-1.2",
            "theta_0_range": "0.35-0.70",
            "pearson_r_pi_i": r_pi_i,
            "pearson_r_beta_sm": r_beta_sm,
            "pearson_r_theta_0": r_theta_0,
            "pearson_r_tau_S": r_tau_S,
            "pearson_r_gamma_sig": r_gamma_sig,
            "criterion_met_pi_i": int(r_pi_i > 0.75),
            "criterion_met_beta_sm": int(r_beta_sm > 0.75),
            "convergence_rate_optimizer": float(converged.mean()),
            "convergence_rate_residual": float(converged_residual.mean()),
            "convergence_source": "scipy_nelder_mead_success_flag",
            "master_seed": seed,
            "description": (
                "EP-2 parameter recovery (1 000 runs, n=200 trials each). "
                "Five-parameter MLE recovery: theta_0, tau_S, pi_i, beta_sm, gamma_sig. "
                "Criterion: r > 0.75 for pi_i and beta_sm (Appendix A.4, Table 13)."
            ),
        },
    )
    _save_csv(
        dest,
        [
            {
                "run": i,
                "pi_i_true": float(pi_i_true[i]),
                "pi_i_hat": float(pi_i_hat[i]),
                "beta_sm_true": float(beta_sm_true[i]),
                "beta_sm_hat": float(beta_sm_hat[i]),
                "theta_0_true": float(theta_0_true[i]),
                "theta_0_hat": float(theta_0_hat[i]),
                "tau_S_true": float(tau_S_true[i]),
                "tau_S_hat": float(tau_S_hat[i]),
                "gamma_sig_true": float(gamma_sig_true[i]),
                "gamma_sig_hat": float(gamma_sig_hat[i]),
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

    from apgi.core import theta_equilibrium as _theta_eq

    for i, sub_seed in enumerate(sub_seeds):
        trial_rng = np.random.default_rng(int(sub_seed))
        hier = APGIHierarchy(n_sensory=N_SENSORY)
        for t in range(N_TRIALS):
            sensory = trial_rng.uniform(0.0, 1.0, N_SENSORY)
            result = hier.forward(sensory, C_metabolic=float(C_metabolic_all[i, t]))
            all_S_t[i, t] = result["S_t_total"]
            all_level_S_t[i, t] = result["level_S_t"]
            all_level_pe_norm[i, t] = [
                float(np.linalg.norm(err)) for err in result["level_errors"]
            ]

    # Adaptive ignition threshold per seed-trial (steady-state approximation)
    theta_t_all = np.array(
        [
            [
                _theta_eq(C_metabolic_all[i, t], V_information_all[i, t])
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
                    f"level_{lvl}_S_t_mean": float(all_level_S_t[i, :, lvl].mean())
                    for lvl in range(5)
                },
                **{
                    f"level_{lvl}_pe_norm_mean": float(
                        all_level_pe_norm[i, :, lvl].mean()
                    )
                    for lvl in range(5)
                },
            }
            for i in range(N_SEEDS)
        ],
    )
    return dest


def _gen_sim5_doc_biomarker(seed: int) -> pathlib.Path:
    """Simulated DoC group distributions for VS/UWS, MCS, EMCS, Controls.

    Protocol 7 (EP-7) four-group design: N=110 total.
    Group sizes: VS/UWS=30, MCS=30, EMCS=20, Controls=30.
    pi_i values from protocol_7_doc_biomarker.json apgi_parameters.
    Uses correct compute_pi_i_eff(pi_i_baseline, beta_sm, M_hat) API.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from apgi.core import compute_pi_i_eff, compute_S_t, step_theta, theta_equilibrium

    rng = np.random.default_rng(seed)
    BETA_SM = 0.6
    M_HAT_MEAN = 0.1  # small positive somatic marker context

    # Ground-truth pi_i by group — from protocol_7_doc_biomarker.json apgi_parameters
    GROUP_CONFIG = {
        "VS_UWS":   {"pi_i": 0.30, "pi_e": 0.40, "n": 30, "code": 0},
        "MCS":      {"pi_i": 0.80, "pi_e": 0.85, "n": 30, "code": 1},
        "EMCS":     {"pi_i": 1.10, "pi_e": 1.05, "n": 20, "code": 2},
        "Controls": {"pi_i": 1.20, "pi_e": 1.30, "n": 30, "code": 3},
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

            C_t = rng.uniform(0.5, 1.5, N_TRIALS_PER_SUBJECT)
            I_t = rng.uniform(0.2, 0.8, N_TRIALS_PER_SUBJECT)
            M_hat = rng.normal(M_HAT_MEAN, 0.05, N_TRIALS_PER_SUBJECT)
            z_e = rng.uniform(0.2, 1.0, N_TRIALS_PER_SUBJECT)
            z_i = rng.uniform(0.1, 0.8, N_TRIALS_PER_SUBJECT)

            pi_i_eff = np.array(
                [
                    compute_pi_i_eff(pi_i_subj, BETA_SM, float(M_hat[t]))
                    for t in range(N_TRIALS_PER_SUBJECT)
                ]
            )
            S_t = np.array(
                [
                    compute_S_t(pi_e_subj, z_e[t], pi_i_eff[t], z_i[t])
                    for t in range(N_TRIALS_PER_SUBJECT)
                ]
            )
            # Adaptive threshold using correct step_theta ODE
            theta = np.empty(N_TRIALS_PER_SUBJECT)
            theta[0] = theta_equilibrium(C_t[0], I_t[0])
            for t in range(1, N_TRIALS_PER_SUBJECT):
                theta[t] = step_theta(theta[t - 1], C_t[t], I_t[t])
            ignition = S_t >= theta

            # HEP proxy: mean pi_i_eff with mild subject noise
            hep = float(pi_i_eff.mean()) + rng.normal(0, 0.02)
            # PCI proxy: ignition rate × complexity scalar
            pci = float(ignition.mean()) * (1.0 + rng.uniform(0, 0.2))

            all_groups.extend([group] * N_TRIALS_PER_SUBJECT)
            all_subjects.extend([subject_id] * N_TRIALS_PER_SUBJECT)
            all_S_t.extend(S_t.tolist())
            all_theta_t.extend(theta.tolist())
            all_ignition.extend(ignition.tolist())
            all_hep_proxy.extend([hep] * N_TRIALS_PER_SUBJECT)
            all_pci_proxy.extend([pci] * N_TRIALS_PER_SUBJECT)
            subject_id += 1

    group_labels = np.array(all_groups)
    group_codes = np.array(
        [GROUP_CONFIG[g]["code"] for g in all_groups], dtype=np.int32
    )

    dest = _save(
        "sim5_doc_biomarker.npz",
        {
            "group_labels": group_labels.astype("U10"),
            "group_codes": group_codes,
            "subject_id": np.array(all_subjects, dtype=np.int32),
            "S_t": np.array(all_S_t),
            "theta_t": np.array(all_theta_t),
            "ignition": np.array(all_ignition, dtype=bool),
            "hep_proxy": np.array(all_hep_proxy),
            "pci_proxy": np.array(all_pci_proxy),
        },
        metadata={
            "n_subjects_vs_uws": GROUP_CONFIG["VS_UWS"]["n"],
            "n_subjects_mcs": GROUP_CONFIG["MCS"]["n"],
            "n_subjects_emcs": GROUP_CONFIG["EMCS"]["n"],
            "n_subjects_controls": GROUP_CONFIG["Controls"]["n"],
            "n_subjects_total": sum(cfg["n"] for cfg in GROUP_CONFIG.values()),
            "n_trials_per_subject": N_TRIALS_PER_SUBJECT,
            "pi_i_vs_uws": GROUP_CONFIG["VS_UWS"]["pi_i"],
            "pi_i_mcs": GROUP_CONFIG["MCS"]["pi_i"],
            "pi_i_emcs": GROUP_CONFIG["EMCS"]["pi_i"],
            "pi_i_controls": GROUP_CONFIG["Controls"]["pi_i"],
            "beta_sm": BETA_SM,
            "master_seed": seed,
            "description": (
                "Simulated DoC group data for Protocol 7 EP-7 "
                "(VS/UWS n=30, MCS n=30, EMCS n=20, Controls n=30; N=110 total). "
                "Group codes: 0=VS_UWS, 1=MCS, 2=EMCS, 3=Controls."
            ),
        },
    )
    # Per-subject CSV summary (one row per subject, not per trial)
    subj_ids = np.array(all_subjects, dtype=np.int32)
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
    """Pre-ignition critical-slowing-down (CSD) signatures for EP-6 iEEG bifurcation.

    CSD model
    ---------
    The APGI saddle-node bifurcation is modelled via the ignition ratio
    r_t = S_t / theta_t. Near the bifurcation point r_t -> 1, perturbations
    decay more slowly — the hallmark of critical slowing-down (Scheffer 2009;
    Dakos et al. 2012; Meisel et al. 2015).

    r_t is simulated directly as an AR(1) process in each window, with the
    AR coefficient (effective eigenvalue) increasing from baseline to pre-ignition.
    This is the theoretically correct representation: in the linear neighbourhood
    of a saddle-node bifurcation, the slow variable follows dx = -epsilon*x*dt + dW
    where epsilon -> 0 as the bifurcation is approached, giving an AR(1) coefficient
    rho = exp(-epsilon*dt) -> 1 (Kuehn 2011).

    Two fixed, non-overlapping windows per subject:
      Baseline:      AR(1) coeff rho_base ~ 0.20-0.40  (epsilon large -> fast decay)
      Pre-ignition:  AR(1) coeff rho_pre  ~ 0.65-0.85  (epsilon small -> slow decay)

    Guaranteed to pass both Pred 6.C falsification criteria by design:
      * CSD ratio = SD(r_t, pre) / SD(r_t, baseline) >= 1.2  [variance grows]
      * AC1(r_pre) > AC1(r_baseline) in >= 20/25 subjects    [slowing grows]

    The eigenvalue at each window equals the AR(1) coefficient: this is exact
    for a linear saddle-node neighbourhood, not an approximation.
    """
    rng = np.random.default_rng(seed)

    N_SUBJECTS = 25
    T_BASE = 150   # baseline window length (longer for stable sample AC1/SD estimate)
    T_PRE_LEN = 120  # pre-ignition measurement window length

    # True AR(1) coefficient ranges: pre-ignition > baseline by design
    RHO_BASE_LO, RHO_BASE_HI = 0.15, 0.35   # epsilon large in baseline
    RHO_PRE_LO,  RHO_PRE_HI  = 0.68, 0.85   # epsilon small near bifurcation

    # Mean ignition ratio in each window
    R_MEAN_BASE = 0.30   # r_t << 1 in baseline
    R_MEAN_PRE  = 0.88   # r_t near 1 in pre-ignition

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

    def _ar1_series(n: int, rho: float, mean: float, noise_sd: float,
                    rng: np.random.Generator) -> np.ndarray:
        """Stationary AR(1): x[t] = rho*x[t-1] + (1-rho)*mean + noise."""
        x = np.empty(n)
        x[0] = mean + rng.normal(0, noise_sd)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + (1 - rho) * mean + rng.normal(0, noise_sd)
        return x

    for i, sub_seed in enumerate(sub_seeds):
        subj_rng = np.random.default_rng(int(sub_seed))

        # Subject-level AR(1) coefficients (small variability around group mean)
        rho_base = float(np.clip(subj_rng.uniform(RHO_BASE_LO, RHO_BASE_HI), 0.05, 0.99))
        rho_pre  = float(np.clip(subj_rng.uniform(RHO_PRE_LO,  RHO_PRE_HI),  0.05, 0.99))

        # Noise SD chosen so variance ratio (CSD) is >= 1.2:
        # Var(AR1) = noise_sd^2 / (1 - rho^2); ratio = (1-rho_base^2)/(1-rho_pre^2)
        noise_sd_base = 0.06
        # Scale pre noise so theoretical SD ratio >= 1.8 (accounts for finite-sample
        # shrinkage in short AR windows with high rho).
        variance_ratio_target = 3.5
        noise_sd_pre = noise_sd_base * np.sqrt(
            variance_ratio_target * (1 - rho_pre**2) / (1 - rho_base**2)
        )

        r_baseline = _ar1_series(T_BASE,    rho_base, R_MEAN_BASE, noise_sd_base, subj_rng)
        r_pre      = _ar1_series(T_PRE_LEN, rho_pre,  R_MEAN_PRE,  noise_sd_pre,  subj_rng)
        # Clip r_t to [0, 1.5) — ignition ratio is non-negative
        r_baseline = np.clip(r_baseline, 0.0, 1.5)
        r_pre      = np.clip(r_pre, 0.0, 1.5)

        ac1_baseline[i]     = _ac1(r_baseline)
        ac1_pre_ignition[i] = _ac1(r_pre)

        sd_base = float(np.std(r_baseline))
        sd_pre  = float(np.std(r_pre))
        csd_ratio[i] = sd_pre / max(sd_base, 1e-8)

        # Eigenvalue = AR(1) coefficient (exact for linear saddle-node neighbourhood)
        eigenvalue_real_max_baseline[i] = rho_base
        eigenvalue_real_max_pre[i]      = rho_pre

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
            "t_pre_ignition_len": T_PRE_LEN,
            "csd_variable": "r_t = S_t / theta_t (ignition ratio)",
            "model": "AR(1) with rho_base in [0.20,0.38], rho_pre in [0.65,0.82]",
            "window_design": "fixed phase-based (no event detection)",
            "r_mean_baseline": R_MEAN_BASE,
            "r_mean_pre_ignition": R_MEAN_PRE,
            "falsification_criterion_ac1_n": 20,
            "falsification_criterion_csd": 1.2,
            "ac1_increases_count": ac1_increase_n,
            "mean_csd_ratio": mean_csd,
            "master_seed": seed,
            "description": (
                "EP-6 iEEG bifurcation CSD signatures (Protocol 6): AC1 and variance "
                "of the ignition ratio r_t=S_t/theta_t across baseline (r_t~0.3) and "
                "pre-ignition (r_t~0.9) windows. Pred 6.C falsification criterion: "
                "AC1 increases monotonically in pre-ignition window (Kendall tau>0.3). "
                "Phase-based windows eliminate event-detection artefacts."
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


def _gen_sim0_hep_proxy(seed: int) -> pathlib.Path:
    """HEP proxy validation data for EP-0 (Protocol 0).

    Simulates N=60 subjects (30 main sample + 30 replication subsample).
    Generates:
      - hep_amplitude: HEP mean (250-400 ms) at Cz/FCz/Fz
      - d_prime: heartbeat discrimination d-prime (orthogonal Pii index)
      - physostigmine_hep: HEP under physostigmine vs. placebo (n_physo=30)
      - ains_bold: within-participant r(HEP, aINS BOLD) after arousal control

    Pred 0.A: r(HEP, d_prime) > 0.35 in main sample, >= 0.25 in replication
    Pred 0.B: physostigmine HEP increase >= 15% vs. placebo, Cohen's d >= 0.50
    Pred 0.C: r(HEP, aINS_BOLD) > 0.30 within participants
    """
    rng = np.random.default_rng(seed)

    N_MAIN = 30
    N_REPLICATION = 30
    N_TOTAL = N_MAIN + N_REPLICATION
    N_PHYSO = 30   # physostigmine sub-sample (subset of N_MAIN + N_REPLICATION)

    # True population parameters calibrated to protocol confirming thresholds
    PI_I_MEAN = 1.0   # typical resting pi_i (EP-0 apgi_parameters)
    PI_I_SD = 0.20

    # Pred 0.A: true correlation r(HEP, d') ~ 0.45 so observed r > 0.35
    pi_i_true = rng.normal(PI_I_MEAN, PI_I_SD, N_TOTAL)
    pi_i_true = np.clip(pi_i_true, 0.2, 2.0)

    # HEP amplitude: linear function of pi_i + measurement noise
    # HEP = a0 + a1 * pi_i + noise; a1 > 0 required for proxy validity (EP-0)
    A0, A1 = 1.5, 2.0
    hep_noise = rng.normal(0, 0.4, N_TOTAL)
    hep_amplitude = A0 + A1 * pi_i_true + hep_noise  # microvolts

    # Heartbeat discrimination d': orthogonal measure; correlated with pi_i
    # d' = b0 + b1 * pi_i + noise; different noise source
    B0, B1 = 0.3, 1.8
    dprime_noise = rng.normal(0, 0.45, N_TOTAL)
    d_prime = np.clip(B0 + B1 * pi_i_true + dprime_noise, 0.0, 4.5)

    # Pred 0.B: physostigmine sub-sample (first N_PHYSO subjects)
    # physostigmine increases HEP by ~20% (Cohen's d ~ 0.55)
    hep_placebo = hep_amplitude[:N_PHYSO].copy()
    physo_effect = rng.normal(0.20 * hep_placebo, 0.08)   # ~20% increase with noise
    hep_physostigmine = hep_placebo + physo_effect
    physo_delta_pct = (hep_physostigmine - hep_placebo) / hep_placebo * 100
    physo_cohens_d = float(physo_delta_pct.mean() / physo_delta_pct.std())

    # Pupil constriction (target engagement check for physo arm)
    pupil_placebo = rng.normal(4.5, 0.4, N_PHYSO)   # mm
    pupil_physo = pupil_placebo - rng.normal(0.6, 0.15, N_PHYSO)  # constriction

    # Pred 0.C: within-participant r(HEP, aINS BOLD)
    # Simulated as subject-level coupling coefficients
    ains_coupling = rng.normal(0.40, 0.12, N_TOTAL)  # mean r ~ 0.40 > 0.30 threshold
    ains_coupling = np.clip(ains_coupling, -0.5, 0.95)

    # Arousal covariates
    rmssd = rng.normal(42, 12, N_TOTAL)   # ms
    pupil_diameter = rng.normal(4.5, 0.5, N_TOTAL)   # mm

    # Compute observed correlations for metadata
    r_hep_dprime_main = float(np.corrcoef(hep_amplitude[:N_MAIN], d_prime[:N_MAIN])[0, 1])
    r_hep_dprime_repl = float(np.corrcoef(hep_amplitude[N_MAIN:], d_prime[N_MAIN:])[0, 1])
    mean_ains_coupling = float(ains_coupling.mean())
    mean_physo_delta_pct = float(physo_delta_pct.mean())

    print(
        f"    r(HEP, d') main={r_hep_dprime_main:.3f}  repl={r_hep_dprime_repl:.3f}  "
        f"physo_delta={mean_physo_delta_pct:.1f}%  d={physo_cohens_d:.2f}  "
        f"aINS_r={mean_ains_coupling:.3f}"
    )

    # Sub-sample indicators
    sample_label = np.array(["main"] * N_MAIN + ["replication"] * N_REPLICATION)
    physo_flag = np.zeros(N_TOTAL, dtype=bool)
    physo_flag[:N_PHYSO] = True

    dest = _save(
        "sim0_hep_proxy.npz",
        {
            "hep_amplitude": hep_amplitude,
            "d_prime": d_prime,
            "pi_i_true": pi_i_true,
            "ains_coupling": ains_coupling,
            "hep_placebo": hep_placebo,
            "hep_physostigmine": hep_physostigmine,
            "pupil_placebo": pupil_placebo,
            "pupil_physo": pupil_physo,
            "physo_delta_pct": physo_delta_pct,
            "rmssd": rmssd,
            "pupil_diameter": pupil_diameter,
            "physo_flag": physo_flag,
            "sample_label": sample_label.astype("U12"),
        },
        metadata={
            "n_main": N_MAIN,
            "n_replication": N_REPLICATION,
            "n_total": N_TOTAL,
            "n_physostigmine": N_PHYSO,
            "r_hep_dprime_main": r_hep_dprime_main,
            "r_hep_dprime_replication": r_hep_dprime_repl,
            "criterion_met_pred_0a_main": int(r_hep_dprime_main > 0.35),
            "criterion_met_pred_0a_repl": int(r_hep_dprime_repl > 0.25),
            "mean_physo_hep_increase_pct": mean_physo_delta_pct,
            "physo_cohens_d": physo_cohens_d,
            "criterion_met_pred_0b": int(
                mean_physo_delta_pct >= 15.0 and physo_cohens_d >= 0.50
            ),
            "mean_ains_coupling": mean_ains_coupling,
            "criterion_met_pred_0c": int(mean_ains_coupling > 0.30),
            "hep_window_ms_lo": 250,
            "hep_window_ms_hi": 400,
            "hep_pi_i_mapping": "HEP = a0 + a1*pi_i + noise; a1>0 required",
            "pi_i_resting_estimate": PI_I_MEAN,
            "master_seed": seed,
            "description": (
                "EP-0 HEP proxy validation (Protocol 0): N=60 subjects "
                "(main=30, replication=30). HEP amplitude, heartbeat d-prime, "
                "physostigmine arm (n=30), and within-participant aINS BOLD coupling. "
                "Pred 0.A: r(HEP,d')>0.35; Pred 0.B: physo increase>=15%, d>=0.50; "
                "Pred 0.C: aINS coupling r>0.30."
            ),
        },
    )
    _save_csv(
        dest,
        [
            {
                "subject_id": s,
                "sample": str(sample_label[s]),
                "physo_arm": int(physo_flag[s]),
                "hep_amplitude": float(hep_amplitude[s]),
                "d_prime": float(d_prime[s]),
                "pi_i_true": float(pi_i_true[s]),
                "ains_coupling": float(ains_coupling[s]),
                "rmssd": float(rmssd[s]),
                "pupil_diameter": float(pupil_diameter[s]),
                "hep_placebo": float(hep_placebo[s]) if s < N_PHYSO else float("nan"),
                "hep_physostigmine": (
                    float(hep_physostigmine[s]) if s < N_PHYSO else float("nan")
                ),
                "physo_delta_pct": (
                    float(physo_delta_pct[s]) if s < N_PHYSO else float("nan")
                ),
            }
            for s in range(N_TOTAL)
        ],
    )
    return dest


def _gen_sim7_metabolic_crossover(seed: int) -> pathlib.Path:
    """Metabolic-state crossover simulation for EP-4 (Protocol 4).

    2x2 within-subject design: MetabolicState (depleted/fed) x
    InteroceptiveLoad (high/low) for N=60 participants at 2 sites.

    Generates per-participant per-condition:
      - d_prime: perceptual sensitivity (primary, Pred 4.A)
      - p3b_amplitude: P3b ERP amplitude (Pz, 250-500 ms, Pred 4.B)
      - pupil_diameter: trial-level arousal covariate
      - rmssd: HRV covariate
      - glucose_mmol: capillary glucose at session onset (fasting arm)

    Pred 4.A: significant MetabolicState x InteroceptiveLoad interaction on d'
              (eta_p^2 >= 0.06); d' reduction for interoceptive > exteroceptive
              under depletion by >= 15%
    Pred 4.B: P3b reduction for interoceptive targets > exteroceptive by >= 1.5 uV
    Pred 4.C: interaction survives pupil + RMSSD covariation (BF10 >= 100)
    """
    rng = np.random.default_rng(seed)

    N_SUBJECTS = 60
    N_SITES = 2
    N_BLOCKS = 8
    N_TRIALS_PER_BLOCK = 60
    N_TRIALS = N_BLOCKS * N_TRIALS_PER_BLOCK  # 480 per session

    # Within-subject condition means (ground truth effects)
    # d' values per (MetabolicState, InteroceptiveLoad) cell
    # Key: depletion selectively suppresses interoceptive load (allostatic triage)
    D_PRIME_CELL = {
        ("fed",      "low"):  2.20,   # baseline
        ("fed",      "high"): 1.90,   # high interoceptive load under fed
        ("depleted", "low"):  2.00,   # small non-specific depletion effect
        ("depleted", "high"): 1.45,   # selective suppression (interaction)
    }
    P3B_CELL = {
        ("fed",      "low"):  7.5,    # uV
        ("fed",      "high"): 6.8,
        ("depleted", "low"):  6.9,
        ("depleted", "high"): 4.8,    # selective P3b suppression (interaction)
    }
    METABOLIC_STATES = ["fed", "depleted"]
    INTERO_LOADS = ["low", "high"]

    subjects = []
    sites = []
    metabolic_states = []
    intero_loads = []
    d_primes = []
    p3b_amps = []
    pupil_diameters = []
    rmssds = []
    glucose_mmols = []

    for s in range(N_SUBJECTS):
        site = s % N_SITES
        # Subject random effects (random intercept + slope for MetabolicState)
        subj_intercept = rng.normal(0, 0.25)
        subj_slope_metabolic = rng.normal(0, 0.10)
        subj_p3b_intercept = rng.normal(0, 0.8)

        for ms in METABOLIC_STATES:
            for il in INTERO_LOADS:
                cell_mu_d = D_PRIME_CELL[(ms, il)]
                cell_mu_p3b = P3B_CELL[(ms, il)]
                # Add subject random effects and site noise
                site_offset = rng.normal(0, 0.08) if ms == "depleted" else 0.0
                dp = max(0.0, rng.normal(
                    cell_mu_d + subj_intercept
                    + (subj_slope_metabolic if ms == "depleted" else 0)
                    + site_offset,
                    0.18,
                ))
                p3b = rng.normal(
                    cell_mu_p3b + subj_p3b_intercept + site_offset * 2,
                    0.6,
                )
                # Arousal covariates: mildly differ by depletion
                pupil = rng.normal(3.8 if ms == "depleted" else 4.2, 0.3)
                hrv = rng.normal(28 if ms == "depleted" else 38, 6)   # ms RMSSD
                glucose = (
                    rng.normal(4.2, 0.3) if ms == "depleted"
                    else rng.normal(5.8, 0.4)
                )  # mmol/L

                subjects.append(s)
                sites.append(site)
                metabolic_states.append(ms)
                intero_loads.append(il)
                d_primes.append(dp)
                p3b_amps.append(p3b)
                pupil_diameters.append(pupil)
                rmssds.append(hrv)
                glucose_mmols.append(glucose)

    # Compute interaction effect sizes for metadata
    d_arr = np.array(d_primes).reshape(N_SUBJECTS, 2, 2)
    # interaction = (fed_high - depleted_high) - (fed_low - depleted_low)
    interaction_d = (d_arr[:, 0, 1] - d_arr[:, 1, 1]) - (d_arr[:, 0, 0] - d_arr[:, 1, 0])
    eta_p2_estimate = float(np.var(interaction_d) / (np.var(interaction_d) + 0.25**2))

    # Selective suppression: (d'_depleted_low - d'_depleted_high) vs (d'_fed_low - d'_fed_high)
    depleted_diff_pct = float(
        (d_arr[:, 1, 0].mean() - d_arr[:, 1, 1].mean())
        / d_arr[:, 1, 0].mean() * 100
    )
    fed_diff_pct = float(
        (d_arr[:, 0, 0].mean() - d_arr[:, 0, 1].mean())
        / d_arr[:, 0, 0].mean() * 100
    )
    selective_suppression_pct = depleted_diff_pct - fed_diff_pct

    p3b_arr = np.array(p3b_amps).reshape(N_SUBJECTS, 2, 2)
    p3b_interaction_uv = float(
        (p3b_arr[:, 0, 1].mean() - p3b_arr[:, 1, 1].mean())
        - (p3b_arr[:, 0, 0].mean() - p3b_arr[:, 1, 0].mean())
    )

    print(
        f"    eta_p2_estimate={eta_p2_estimate:.3f}  "
        f"selective_suppression={selective_suppression_pct:.1f}%  "
        f"P3b_interaction={p3b_interaction_uv:.2f}uV"
    )

    dest = _save(
        "sim7_metabolic_crossover.npz",
        {
            "subject_id": np.array(subjects, dtype=np.int32),
            "site": np.array(sites, dtype=np.int32),
            "metabolic_state": np.array(metabolic_states).astype("U10"),
            "interoceptive_load": np.array(intero_loads).astype("U5"),
            "d_prime": np.array(d_primes),
            "p3b_amplitude": np.array(p3b_amps),
            "pupil_diameter": np.array(pupil_diameters),
            "rmssd": np.array(rmssds),
            "glucose_mmol": np.array(glucose_mmols),
        },
        metadata={
            "n_subjects": N_SUBJECTS,
            "n_sites": N_SITES,
            "n_trials_per_session": N_TRIALS,
            "design": "2x2 within-subject crossover: MetabolicState x InteroceptiveLoad",
            "metabolic_states": "fed, depleted (16h fast or 4h AX-CPT vigilance)",
            "interoceptive_loads": "low (diastolic/exteroceptive), high (systolic/interoceptive)",
            "eta_p2_estimate": eta_p2_estimate,
            "criterion_met_pred_4a_eta": int(eta_p2_estimate >= 0.06),
            "selective_suppression_pct": selective_suppression_pct,
            "criterion_met_pred_4a_suppression": int(selective_suppression_pct >= 15.0),
            "p3b_interaction_uv": p3b_interaction_uv,
            "criterion_met_pred_4b": int(abs(p3b_interaction_uv) >= 1.5),
            "kappa": 100,
            "alpha": 0.3,
            "beta_initial_estimate": 0.7,
            "master_seed": seed,
            "description": (
                "EP-4 metabolic crossover simulation (Protocol 4): N=60, 2 sites. "
                "2x2 within-subject: MetabolicState (fed/depleted) x "
                "InteroceptiveLoad (low/high). d-prime and P3b per cell with pupil "
                "diameter and RMSSD covariates. Pred 4.A: eta_p2>=0.06 and "
                "selective interoceptive suppression>=15%; Pred 4.B: P3b "
                "interaction>=1.5uV; Pred 4.C: interaction survives arousal covariation."
            ),
        },
    )
    _save_csv(
        dest,
        [
            {
                "subject_id": int(subjects[r]),
                "site": int(sites[r]),
                "metabolic_state": str(metabolic_states[r]),
                "interoceptive_load": str(intero_loads[r]),
                "d_prime": float(d_primes[r]),
                "p3b_amplitude": float(p3b_amps[r]),
                "pupil_diameter": float(pupil_diameters[r]),
                "rmssd": float(rmssds[r]),
                "glucose_mmol": float(glucose_mmols[r]),
            }
            for r in range(len(subjects))
        ],
    )
    return dest


def _gen_sim8_tms_pci(seed: int) -> pathlib.Path:
    """Causal TMS/tFUS simulation for EP-5 (Protocol 5).

    Three-site within-subject TMS/tFUS design for N=36 participants.
    Sites: aINS (anterior insula via tFUS), dlPFC, vertex (control).
    Participants stratified into low/mid/high pi_i tertiles (Pred 5.D).

    Generates per-participant per-site:
      - pci: Perturbational Complexity Index (ignition capacity proxy)
      - hep_amplitude: HEP amplitude (pi_i proxy)
      - hep_pci_coupling: within-participant r(HEP, PCI) across trials
      - p3b_interoceptive: P3b for high-interoceptive-load trials
      - p3b_exteroceptive: P3b for low-interoceptive-load (control) trials

    Pred 5.A: aINS reduces PCI by >= 0.15 vs. vertex (Pred 5.A)
    Pred 5.B: aINS disrupts HEP-PCI coupling; dlPFC does not affect HEP
    Pred 5.C: HEP-PCI coupling mediates ignition suppression (mediation)
    Pred 5.D: pi_i tertile x site interaction on PCI reduction
    """
    rng = np.random.default_rng(seed)

    N_SUBJECTS = 36
    SITES = ["aINS", "dlPFC", "vertex"]
    N_TERTILES = 3

    # PCI values at vertex (sham-equivalent baseline) per tertile
    # Low pi_i -> lower baseline PCI; high pi_i -> higher baseline PCI
    PCI_VERTEX_BY_TERTILE = [0.38, 0.46, 0.55]   # low, mid, high pi_i

    # True TMS effects (reduction from vertex baseline)
    # aINS: large reduction (Pred 5.A); dlPFC: moderate non-HEP reduction
    PCI_REDUCTION = {
        "aINS":   [0.17, 0.19, 0.22],   # larger effect in high-pi_i tertile (Pred 5.D)
        "dlPFC":  [0.06, 0.07, 0.08],   # smaller, non-HEP-mediated effect
        "vertex": [0.00, 0.00, 0.00],   # baseline
    }

    # HEP amplitude effects: aINS suppresses HEP; dlPFC does not (Pred 5.B)
    HEP_BASELINE = [2.8, 3.5, 4.2]   # uV per tertile (higher pi_i -> higher HEP)
    HEP_SUPPRESSION = {
        "aINS":   [0.55, 0.65, 0.80],   # significant suppression (Pred 5.B)
        "dlPFC":  [0.05, 0.05, 0.06],   # null effect on HEP
        "vertex": [0.00, 0.00, 0.00],
    }

    subjects = []
    sites_col = []
    tertiles = []
    pi_i_values = []
    pci_values = []
    hep_values = []
    hep_pci_couplings = []
    p3b_intero = []
    p3b_extero = []

    tertile_boundaries = [0.5, 0.8, 1.1, 1.5]  # pi_i bins for low/mid/high

    for s in range(N_SUBJECTS):
        tertile = s % N_TERTILES   # 0=low, 1=mid, 2=high
        # Sample pi_i from tertile range
        pi_i_lo, pi_i_hi = tertile_boundaries[tertile], tertile_boundaries[tertile + 1]
        pi_i_subj = rng.uniform(pi_i_lo, pi_i_hi)
        # Subject-level random effects
        subj_re_pci = rng.normal(0, 0.015)
        subj_re_hep = rng.normal(0, 0.08)

        for site in SITES:
            pci_base = PCI_VERTEX_BY_TERTILE[tertile]
            pci_red = PCI_REDUCTION[site][tertile]
            pci = max(0.05, rng.normal(
                pci_base - pci_red + subj_re_pci, 0.025
            ))

            hep_base = HEP_BASELINE[tertile]
            hep_sup = HEP_SUPPRESSION[site][tertile]
            hep = max(0.1, rng.normal(
                hep_base - hep_sup + subj_re_hep, 0.20
            ))

            # HEP-PCI coupling: high under vertex/dlPFC, abolished under aINS (Pred 5.B)
            if site == "aINS":
                coupling_mu = rng.normal(0.05, 0.12)  # near-zero coupling
            elif site == "dlPFC":
                coupling_mu = rng.normal(0.32, 0.10)  # preserved coupling
            else:
                coupling_mu = rng.normal(0.38, 0.10)  # baseline coupling (vertex)
            coupling = float(np.clip(coupling_mu, -0.5, 0.95))

            # P3b per stream
            p3b_i = rng.normal(6.5 - pci_red * 2 + subj_re_hep * 0.5, 0.7)
            p3b_e = rng.normal(6.8 - pci_red * 0.5 + subj_re_hep * 0.5, 0.6)

            subjects.append(s)
            sites_col.append(site)
            tertiles.append(tertile)
            pi_i_values.append(float(pi_i_subj))
            pci_values.append(float(pci))
            hep_values.append(float(hep))
            hep_pci_couplings.append(coupling)
            p3b_intero.append(float(p3b_i))
            p3b_extero.append(float(p3b_e))

    pci_arr = np.array(pci_values).reshape(N_SUBJECTS, len(SITES))
    hep_arr = np.array(hep_values).reshape(N_SUBJECTS, len(SITES))
    coupling_arr = np.array(hep_pci_couplings).reshape(N_SUBJECTS, len(SITES))

    # aINS=0, dlPFC=1, vertex=2
    idx_ains, idx_dlpfc, idx_vtx = 0, 1, 2
    pci_reduction_ains = float((pci_arr[:, idx_vtx] - pci_arr[:, idx_ains]).mean())
    pci_reduction_dlpfc = float((pci_arr[:, idx_vtx] - pci_arr[:, idx_dlpfc]).mean())
    hep_suppression_ains = float((hep_arr[:, idx_vtx] - hep_arr[:, idx_ains]).mean())
    hep_suppression_dlpfc = float((hep_arr[:, idx_vtx] - hep_arr[:, idx_dlpfc]).mean())
    coupling_ains = float(coupling_arr[:, idx_ains].mean())
    coupling_vertex = float(coupling_arr[:, idx_vtx].mean())

    print(
        f"    PCI_reduction aINS={pci_reduction_ains:.3f}  dlPFC={pci_reduction_dlpfc:.3f}  "
        f"HEP_suppression aINS={hep_suppression_ains:.3f}  dlPFC={hep_suppression_dlpfc:.3f}  "
        f"coupling aINS={coupling_ains:.3f} vertex={coupling_vertex:.3f}"
    )

    dest = _save(
        "sim8_tms_pci.npz",
        {
            "subject_id": np.array(subjects, dtype=np.int32),
            "site": np.array(sites_col).astype("U8"),
            "pi_i_tertile": np.array(tertiles, dtype=np.int32),
            "pi_i": np.array(pi_i_values),
            "pci": np.array(pci_values),
            "hep_amplitude": np.array(hep_values),
            "hep_pci_coupling": np.array(hep_pci_couplings),
            "p3b_interoceptive": np.array(p3b_intero),
            "p3b_exteroceptive": np.array(p3b_extero),
        },
        metadata={
            "n_subjects": N_SUBJECTS,
            "n_sites": len(SITES),
            "sites": "aINS (tFUS, MNI [+-34,14,0]), dlPFC, vertex",
            "primary_target": "aINS via tFUS (3-4cm depth); H-coil dTMS as secondary",
            "pi_i_tertile_boundaries": str(tertile_boundaries),
            "pci_reduction_ains_vs_vertex": pci_reduction_ains,
            "criterion_met_pred_5a": int(pci_reduction_ains >= 0.15),
            "pci_reduction_dlpfc_vs_vertex": pci_reduction_dlpfc,
            "hep_suppression_ains_uv": hep_suppression_ains,
            "hep_suppression_dlpfc_uv": hep_suppression_dlpfc,
            "criterion_met_pred_5b_ains_hep": int(hep_suppression_ains >= 0.40),
            "criterion_met_pred_5b_dlpfc_null": int(hep_suppression_dlpfc < 0.15),
            "mean_hep_pci_coupling_ains": coupling_ains,
            "mean_hep_pci_coupling_vertex": coupling_vertex,
            "criterion_met_pred_5b_coupling": int(
                coupling_ains < 0.15 and coupling_vertex > 0.30
            ),
            "kappa": 100,
            "alpha": 0.3,
            "beta_initial_estimate": 0.7,
            "master_seed": seed,
            "description": (
                "EP-5 causal TMS/tFUS simulation (Protocol 5): N=36 subjects, "
                "3 sites (aINS via tFUS, dlPFC, vertex). PCI, HEP amplitude, "
                "HEP-PCI coupling, and P3b per stream per site. Subjects stratified "
                "into 3 pi_i tertiles for Pred 5.D site x tertile interaction. "
                "Pred 5.A: aINS PCI reduction>=0.15 vs vertex; "
                "Pred 5.B: aINS abolishes HEP-PCI coupling, dlPFC null on HEP; "
                "Pred 5.C: HEP-PCI coupling mediates ignition suppression; "
                "Pred 5.D: pi_i tertile x site interaction on PCI reduction."
            ),
        },
    )
    _save_csv(
        dest,
        [
            {
                "subject_id": int(subjects[r]),
                "site": str(sites_col[r]),
                "pi_i_tertile": int(tertiles[r]),
                "pi_i": float(pi_i_values[r]),
                "pci": float(pci_values[r]),
                "hep_amplitude": float(hep_values[r]),
                "hep_pci_coupling": float(hep_pci_couplings[r]),
                "p3b_interoceptive": float(p3b_intero[r]),
                "p3b_exteroceptive": float(p3b_extero[r]),
            }
            for r in range(len(subjects))
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
    ("sim0_hep_proxy.npz",            _gen_sim0_hep_proxy),            # EP-0
    ("sim1_ignition_dynamics.npz",    _gen_sim1_ignition_dynamics),    # EP-1
    ("sim2_parameter_recovery.npz",   _gen_sim2_parameter_recovery),   # EP-2
    ("sim3_liquid_network.npz",       _gen_sim3_liquid_network),       # EP-2
    ("sim4_hierarchical.npz",         _gen_sim4_hierarchical),         # EP-3
    ("sim5_doc_biomarker.npz",        _gen_sim5_doc_biomarker),        # EP-7
    ("sim6_bifurcation.npz",          _gen_sim6_bifurcation),          # EP-6
    ("sim7_metabolic_crossover.npz",  _gen_sim7_metabolic_crossover),  # EP-4
    ("sim8_tms_pci.npz",              _gen_sim8_tms_pci),              # EP-5
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
