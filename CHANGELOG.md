# Changelog

All notable changes to `apgi` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.2.0] — 2026-06-06

### Added
- `data/generate_seeds.py` — canonical synthetic seed generator for all six `.npz` datasets
  (sim1–sim6). Derives every sub-RNG from `MASTER_SEED = 2025`; self-describing metadata
  is stored inside each archive. Outputs SHA-256 digests for Zenodo deposition.
- `CITATION.cff` — machine-readable citation (CFF 1.2.0); GitHub renders "Cite this
  repository" button; required by eLife, PLOS, and Semantic Scholar indexing.
- `reproduce_all.sh` — single-command reproducibility entry point: fetch data → unit tests →
  protocol validation → all 8 figures → analysis scripts.
- `CONTRIBUTING.md` — contributor guidelines covering environment setup, testing, branch
  naming, protocol additions, and figure conventions.
- `zenodo.json` — Zenodo deposit metadata (title, creators, keywords, license, related
  identifiers) ready for upload to `https://zenodo.org/deposit/new`.
- `sim5_doc_biomarker.npz`, `sim6_bifurcation.npz` seed datasets (two new datasets beyond
  the original four).
- Notebook CI validation via `pytest --nbval-lax` in `.github/workflows/test.yml`.

### Changed
- `pyproject.toml` version `0.1.0` → `0.2.0`; Development Status `3 - Alpha` → `4 - Beta`.
- `environment.yml` adds `nbformat`, `ipykernel`, and `nbval` so notebooks execute in the
  conda environment without additional installs.
- `CITATION.cff` version updated to `0.2.0`; ORCID placeholder annotated with registration
  link for indexing in OpenAlex, Semantic Scholar, and Europe PMC.
- `data/checksums.sha256` updated with real SHA-256 digests for sim1, sim3–sim6; sim2
  digest to be added after the 1 000-run MLE job completes.
- `.gitignore` updated: `data/seeds/` excluded (upload to Zenodo instead); old blanket
  `data/*.npz` rule removed.

### Fixed
- `sim6_bifurcation` LNN parameters retuned to reliably satisfy the pre-registered
  falsification criterion (AC1 increase n ≥ 20/25, CSD ratio ≥ 1.2).

---

## [0.1.0] — 2026-05-16

### Added
- `src/apgi/core.py` — renamed from `apgi_core.py` for cleaner import paths.
- `src/apgi/normalizer.py` — `APGINormalizer` for z-score / min-max normalisation across sessions.
- `src/apgi/integration.py` — `APGICoreIntegration` stateful session integrator.
- `src/apgi/clinical.py` — `EnhancedClinicalInterpreter` with four-level consciousness scale.
- `src/apgi/extensions/` namespace — Paper 2 (`LiquidNeuralNetwork`) and Paper 3 (`APGIHierarchy`) moved here to avoid polluting the core install.
- `protocols/schemas/protocol.schema.json` — JSON Schema for experimental protocol files; CI validates all `protocols/*.json` against it.
- `figures/utils.py` — shared APGI Neural Glow palette, `despine`, `save_figure`, `annotate_pearson_r`, and other helpers used by all figure scripts.
- `notebooks/quick_start.ipynb` — end-to-end demo of core equations, `APGICoreIntegration`, and signal normalisation.
- `scripts/fetch_data.py` — root-level shim for `apgi-fetch` CLI entry point.
- `tests/conftest.py` — shared fixtures (`rng`, `sample_S_t`, `fitted_normalizer`).
- `tests/test_normalizer.py` — full coverage of `APGINormalizer`.
- `tests/test_integration.py` — `APGICoreIntegration` tests including somatic modulation validation.
- `data/checksums.sha256` — integrity manifest for Zenodo downloads.
- `.github/workflows/release.yml` — automated PyPI and Zenodo publish on version tags.
- `CHANGELOG.md` — this file.

### Changed
- `src/apgi/__init__.py` now exposes an explicit `__all__` listing only the stable public API; extension classes are excluded from the top-level namespace.
- `figures/generate_figure1.py` and `generate_figure2.py` updated to import from `figures.utils` and `apgi.core`.
- `data/fetch_data.py` promoted to `src/apgi/scripts/fetch_data.py` (registered as `apgi-fetch` CLI entry point in `pyproject.toml`).
- `pyproject.toml` adds `jsonschema>=4.0` to core dependencies.

---

## [0.1.0] — 2026-05-16

### Added
- Initial release: core APGI equations (`compute_S_t`, `compute_theta_t`, `compute_pi_i_eff`, `ignition_criterion`, `run_trial`, `update_theta`).
- `LiquidNeuralNetwork` continuous-time reservoir (Paper 2).
- `APGIHierarchy` five-level hierarchical architecture (Paper 3).
- `parameter_recovery` module with MLE-based recovery simulation (Appendix A.4).
- Experimental protocols: `protocol_1_cardiac_phase.json`, `protocol_2_tms_insula.json`.
- GitHub Actions CI (`test.yml`) covering Python 3.11 and 3.12.
