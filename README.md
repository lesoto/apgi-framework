# APGI Framework

**Adaptive Predictive Global Integration** — a computational framework for consciousness research.

[![CI](https://github.com/lesoto/apgi-framework/actions/workflows/test.yml/badge.svg)](https://github.com/lesoto/apgi-framework/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## Manuscript

> **Pesochin, D. (2026).** Adaptive Predictive Global Integration: A Computational Framework for Consciousness. *Manuscript under review.*  
> Preprint: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) — **replace with real arXiv ID before submission**

This repository is the companion code release for the paper series above.
All figures in Papers 1–3 are reproduced by scripts in `figures/` and `scripts/`.
Pre-registered protocols are in `protocols/` and archived on OSF at [osf.io/XXXXXX](https://osf.io/XXXXXX).

To reproduce all figures from archived data in one command:

```bash
bash reproduce_all.sh
```

## Overview

The APGI framework implements the core equations from the APGI paper series:

| Symbol | Formula | Meaning |
|--------|---------|---------|
| Πⁱ_eff | Πⁱ · exp(−C/κ) | Metabolically gated inhibitory precision |
| Sₜ | Πᵉ·\|zᵉ\| + Πⁱ_eff·\|zⁱ\| | Global integration signal |
| θₜ | α·C + β·V | Adaptive ignition threshold |

**Paper 1** — core APGI equations; EEG interoceptive gating, TMS-insular gating, active inference simulations, DoC biomarkers, fMRI anticipation, and iEEG ignition-dynamics protocols  
**Paper 2** — Liquid Neural Network extension (`apgi.extensions.liquid_network`)  
**Paper 3** — Five-level hierarchical architecture (`apgi.extensions.hierarchical`)

## Repository structure

```
apgi-framework/
├── src/apgi/              # pip-installable library
│   ├── core.py            # Sₜ, θₜ, Πⁱ_eff equations
│   ├── normalizer.py      # APGINormalizer
│   ├── integration.py     # APGICoreIntegration (stateful session)
│   ├── clinical.py        # EnhancedClinicalInterpreter
│   ├── parameter_recovery.py
│   └── extensions/        # optional
│       ├── liquid_network.py
│       └── hierarchical.py
├── protocols/             # pre-registered experimental protocols
│   ├── schemas/           # JSON Schema validation
│   └── *.json
├── tests/                 # pytest suite
├── notebooks/             # worked examples
├── figures/               # figure scripts + output PDFs
├── scripts/               # standalone CLI helpers
└── data/                  # local data cache (Zenodo downloads)
```

## Installation

```bash
git clone https://github.com/lesoto/apgi-framework.git
cd apgi-framework
pip install -e ".[dev]"
```

### Conda (HPC / clean-machine replication)

```bash
conda env create -f environment.yml
conda activate apgi
pip install -e ".[dev]"
```

### Optional: Paper 2 LNN extension (requires PyTorch)

```bash
pip install -e ".[lnn]"
```

## Quick start

```python
import apgi

# Core equations — available directly from the package
pi_i_eff = apgi.compute_pi_i_eff(pi_i=1.0, C_metabolic=50.0, kappa=100.0)
S_t      = apgi.compute_S_t(pi_e=1.2, z_e=0.8, pi_i_eff=pi_i_eff, z_i=0.5)
theta_t  = apgi.compute_theta_t(C_metabolic=1.0, V_information=0.5, alpha=0.3, beta=0.7)
print(f"Sₜ={S_t:.4f}  θₜ={theta_t:.4f}  ignition={apgi.ignition_criterion(S_t, theta_t)}")
```

### Stateful session integration

```python
from apgi.integration import APGICoreIntegration
import numpy as np

rng   = np.random.default_rng(0)
integ = APGICoreIntegration(alpha=0.3, beta=0.7, gamma=0.9)
integ.run_sequence(
    pi_e=rng.uniform(0.8, 1.5, 200), z_e=rng.uniform(0.2, 1.0, 200),
    pi_i=rng.uniform(0.5, 1.5, 200), z_i=rng.uniform(0.1, 0.8, 200),
    C_metabolic=rng.uniform(0.5, 2.0, 200), V_information=rng.uniform(0.1, 1.0, 200),
)
print(f"Ignition rate: {integ.ignition_rate():.3f}")
```

### Paper 2 / 3 extensions (not pulled in by default)

```python
from apgi.extensions.liquid_network import LiquidNeuralNetwork
from apgi.extensions.hierarchical  import APGIHierarchy
```

## Notebooks

Interactive walkthroughs are in `notebooks/`:

| Notebook | Content |
|----------|---------|
| `01_quick_start.ipynb` | Core equations, `APGICoreIntegration`, signal normalisation |
| `02_protocol1_cardiac_eeg.ipynb` | Protocol 1 — Cardiac-EEG: interoceptive precision gating (Pred 1.a–Pred 1.c) |
| `03_protocol2_somatic_agent_sim.ipynb` | Protocol 2 — Somatic-AgentSim: active inference agent simulations (Pred 2.a–Pred 2.e) |
| `04_protocol4_insula_tms.ipynb` | Protocol 4 — Insula-TMS: causal disruption of Πⁱ_eff (Pred 4.a–Pred 4.c) |
| `05_protocol3_anticipation_fmri.ipynb` | Protocol 3 — Anticipation-fMRI: somatic marker vs. prediction error (Pred 3.a–Pred 3.d) |
| `06_protocol6_doc_biomarker.ipynb` | Protocol 6 — DoC-Biomarker: joint HEP+PCI model (Pred 6.a–Pred 6.d, 6.S) |
| `07_protocol5_ignition_ieeg.ipynb` | Protocol 5 — Ignition-iEEG: all-or-none dynamics + pharmacology (Pred 5.a–Pred 5.g) |

```bash
jupyter lab notebooks/
```

## Reproducing figures

```bash
python figures/generate_figure1.py   # ignition dynamics
python figures/generate_figure2.py   # parameter recovery scatter
python figures/generate_figure3.py   # Protocol 1 — Cardiac-EEG: HEP cardiac-phase detection
python figures/generate_figure4.py   # Protocol 4 — Insula-TMS: TMS-induced PCI reduction
python figures/generate_figure5.py   # Protocol 2 — Somatic-AgentSim: somatic marker agent advantage
python figures/generate_figure6.py   # Protocol 6 — DoC-Biomarker: DoC joint biomarker model
python figures/generate_figure7.py   # Protocol 3 — Anticipation-fMRI: vmPFC–insula anticipatory coupling
python figures/generate_figure8.py   # Protocol 5 — Ignition-iEEG: iEEG bimodality + AC1 slowing
```

Output PDFs land in `figures/output/`.

## Running tests

```bash
pytest tests/ -v
```

The full suite (213 tests, 100% coverage) covers core equations, normaliser, integration,
clinical interpreter, parameter recovery, LNN, bifurcation analysis, and five-level hierarchy.

## Data

Simulation outputs are hosted on Zenodo (DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)).
Download pre-computed results with the installed CLI:

```bash
apgi-fetch            # download all datasets
apgi-fetch --list     # list available datasets
apgi-fetch --dataset sim1_ignition_dynamics
```

Checksums are recorded in `data/checksums.sha256`.

## Experimental protocols

Pre-registered protocols are in `protocols/` and on OSF: [osf.io/XXXXXX](https://osf.io/XXXXXX).
All protocol files are validated against `protocols/schemas/protocol.schema.json` in CI.

| File | Paradigm | Status |
|------|---------|--------|
| `protocol_1_cardiac_eeg.json` | Protocol 1 — Cardiac-EEG: interoceptive precision gating (Pred 1.a–Pred 1.c) | specified |
| `protocol_2_somatic_agent_sim.json` | Protocol 2 — Somatic-AgentSim: somatic marker advantage (Pred 2.a–Pred 2.e) | validated |
| `protocol_3_anticipation_fmri.json` | Protocol 3 — Anticipation-fMRI: somatic marker vs. prediction error (Pred 3.a–Pred 3.d) | specified |
| `protocol_4_insula_tms.json` | Protocol 4 — Insula-TMS: causal disruption of Πⁱ_eff (Pred 4.a–Pred 4.c) | specified\_with\_caveat |
| `protocol_5_ignition_ieeg.json` | Protocol 5 — Ignition-iEEG: all-or-none dynamics + pharmacology (Pred 5.a–Pred 5.g) | validated |
| `protocol_6_doc_biomarker.json` | Protocol 6 — DoC-Biomarker: joint HEP+PCI model (Pred 6.a–Pred 6.d, 6.S) | specified |

## Citation

```bibtex
@software{apgi_framework_2026,
  author  = {APGI Research Team},
  title   = {APGI Framework},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/lesoto/apgi-framework}
}
```

## License

MIT — see [LICENSE](LICENSE).
