# APGI Framework

**Adaptive Predictive Global Integration** — a computational framework for consciousness research.

[![CI](https://github.com/lesoto/apgi-framework/actions/workflows/test.yml/badge.svg)](https://github.com/lesoto/apgi-framework/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

The APGI framework implements the core equations from the APGI paper series:

| Symbol | Formula | Meaning |
|--------|---------|---------|
| Πⁱ_eff | Πⁱ · exp(−C/κ) | Metabolically gated inhibitory precision |
| Sₜ | Πᵉ·\|zᵉ\| + Πⁱ_eff·\|zⁱ\| | Global integration signal |
| θₜ | α·C + β·V | Adaptive ignition threshold |

**Paper 1** — core APGI equations, cardiac-phase and TMS-insula protocols  
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
│   └── extensions/        # optional — Paper 2 & 3 (heavy deps)
│       ├── liquid_network.py
│       └── hierarchical.py
├── protocols/             # pre-registered experimental protocols
│   ├── schemas/           # JSON Schema validation
│   └── *.json
├── tests/                 # pytest suite (76 tests)
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
| `02_protocol1_windowing.ipynb` | Cardiac-phase windowing matching Protocol 1 parameters |

```bash
jupyter lab notebooks/
```

## Reproducing figures

```bash
PYTHONPATH=. python figures/generate_figure1.py   # ignition dynamics
PYTHONPATH=. python figures/generate_figure2.py   # parameter recovery scatter
```

Output PDFs land in `figures/output/`.

## Running tests

```bash
pytest tests/ -v
```

The full suite (76 tests) covers core equations, normaliser, integration,
clinical interpreter, parameter recovery, LNN, and five-level hierarchy.

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

| File | Paradigm |
|------|---------|
| `protocol_1_cardiac_phase.json` | Cardiac-phase gating paradigm |
| `protocol_2_tms_insula.json` | TMS-insula disruption of Πⁱ_eff |

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
