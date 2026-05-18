# APGI Framework

**Adaptive Predictive Global Integration** — a computational framework for consciousness research.

[![CI](https://github.com/apgi-research/apgi-framework/actions/workflows/test.yml/badge.svg)](https://github.com/apgi-research/apgi-framework/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

The APGI framework implements the core equations from the APGI paper series:

- **Sₜ** — global integration signal: `Sₜ = Πᵉ·|zᵉ| + Πⁱ_eff·|zⁱ|`
- **θₜ** — adaptive ignition threshold
- **Πⁱ_eff** — effective inhibitory precision
- Liquid Neural Network (LNN) implementation (Paper 2)
- Five-level hierarchical architecture (Paper 3)
- Parameter recovery simulation (Appendix A.4)

## Installation

```bash
# Clone repository
git clone https://github.com/apgi-research/apgi-framework.git
cd apgi-framework

# Install (editable mode — recommended for replication)
pip install -e ".[dev]"
```

### Conda (HPC / clean-machine replication)

```bash
conda env create -f environment.yml
conda activate apgi
pip install -e ".[dev]"
```

## Quick Start

```python
import apgi
from apgi import apgi_core

# Compute global integration signal
S_t = apgi_core.compute_S_t(pi_e=1.2, z_e=0.8, pi_i_eff=0.9, z_i=0.5)
print(f"Sₜ = {S_t:.4f}")

# Check ignition
theta_t = apgi_core.compute_theta_t(C_metabolic=1.0, V_information=0.5, alpha=0.3, beta=0.7)
fired = apgi_core.ignition_criterion(S_t, theta_t)
print(f"Ignition: {fired}")
```

## Reproducing Figures

Each figure has a standalone script under `figures/`:

```bash
python figures/generate_figure1.py
python figures/generate_figure2.py
```

Figures are written to `figures/output/`.

## Running Tests

```bash
pytest tests/ -v
```

## Large Data

Simulation outputs are hosted on Zenodo (DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)).
To download pre-computed results locally:

```bash
python data/fetch_data.py
```

## Experimental Protocols

Pre-registered protocols are in `protocols/` and on OSF: [osf.io/XXXXXX](https://osf.io/XXXXXX)

- `protocol_1_cardiac_phase.json` — cardiac-phase gating paradigm
- `protocol_2_tms_insula.json` — TMS-insula stimulation

## Citation

```bibtex
@software{apgi_framework_2026,
  author  = {APGI Research Team},
  title   = {APGI Framework},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/apgi-research/apgi-framework}
}
```

## License

MIT — see [LICENSE](LICENSE).
